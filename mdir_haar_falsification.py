import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    raise SystemExit("Please install torch (PyTorch) first: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu")

try:
    from huggingface_hub import hf_hub_download  # noqa: F401
except Exception:
    raise SystemExit("Please install huggingface_hub: pip install huggingface_hub")

try:
    from safetensors import safe_open  # noqa: F401
except Exception:
    raise SystemExit("Please install safetensors: pip install safetensors")

try:
    from transformers import AutoTokenizer  # noqa: F401
except Exception:
    raise SystemExit("Please install transformers: pip install transformers>=4.43")

try:
    from scipy import stats  # noqa: F401
except Exception:
    stats = None  # KS-test optional

LEADING_MARKERS = ("▁", "Ġ", "Ċ")
SPECIAL_TOKEN_NAMES = {
    "bos_token", "eos_token", "unk_token", "pad_token", "sep_token",
    "cls_token", "mask_token", "additional_special_tokens"
}

def normalize_token(t: str) -> str:
    if not t:
        return t
    while any(t.startswith(m) for m in LEADING_MARKERS):
        for m in LEADING_MARKERS:
            if t.startswith(m):
                t = t[len(m):]
                break
    return t

def build_token_to_id(tokenizer) -> Dict[str, int]:
    token_to_id = {}
    vocab_size = tokenizer.vocab_size
    exclude_ids = set()
    for name in SPECIAL_TOKEN_NAMES:
        tok = getattr(tokenizer, name, None)
        if tok is None:
            continue
        if isinstance(tok, list):
            for x in tok:
                try:
                    exclude_ids.add(tokenizer.convert_tokens_to_ids(x))
                except Exception:
                    pass
        else:
            try:
                exclude_ids.add(getattr(tokenizer, f"{name}_id"))
            except Exception:
                pass
    for i in range(vocab_size):
        if i in exclude_ids:
            continue
        try:
            t = tokenizer.convert_ids_to_tokens(i)
        except Exception:
            continue
        if t is None:
            continue
        token_to_id[t] = i
    return token_to_id

def intersect_tokens(tokA, tokB, max_common: Optional[int] = None, use_normalize: bool = True) -> List[Tuple[int, int]]:
    t2iA = build_token_to_id(tokA)
    t2iB = build_token_to_id(tokB)

    if use_normalize:
        normA = {}
        for t, i in t2iA.items():
            normA.setdefault(normalize_token(t), []).append(i)
        normB = {}
        for t, i in t2iB.items():
            normB.setdefault(normalize_token(t), []).append(i)

        common = []
        for nt, idsA in normA.items():
            idsB = normB.get(nt)
            if not idsB:
                continue
            common.append((idsA[0], idsB[0]))
    else:
        tokens = set(t2iA.keys()) & set(t2iB.keys())
        common = [(t2iA[t], t2iB[t]) for t in tokens]

    random.seed(0)
    random.shuffle(common)
    if max_common is not None:
        common = common[:max_common]
    return common

def _find_embed_key_and_shard(repo_id: str, revision: Optional[str], cache_dir: str, hf_token: Optional[str]) -> Tuple[str, str]:
    idx_path = hf_hub_download(repo_id, filename="model.safetensors.index.json",
                               revision=revision, token=hf_token, cache_dir=cache_dir, local_dir_use_symlinks=False)
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    weight_map = idx.get("weight_map", {})
    candidates = []
    for k in weight_map.keys():
        lk = k.lower()
        if ("embed" in lk and "weight" in lk and "position" not in lk):
            candidates.append(k)
        if re.search(r"(tok(_)?embeddings|embed_tokens)\.weight$", k):
            candidates.append(k)
    seen = set()
    cand_unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            cand_unique.append(c)

    if not cand_unique:
        shapes = idx.get("metadata", {}).get("tensors", {})
        for k, meta in shapes.items():
            if k in weight_map and isinstance(meta, dict):
                shape = meta.get("shape")
                if isinstance(shape, list) and len(shape) == 2:
                    cand_unique.append(k)
                    break

    if not cand_unique:
        raise RuntimeError(f"[{repo_id}] Could not locate input embedding weight key via index metadata.")

    key = cand_unique[0]
    shard = weight_map[key]
    return key, shard

def load_input_embeddings(repo_id: str, cache_dir: str, hf_token: Optional[str] = None,
                          revision: Optional[str] = None, dtype: str = "float32") -> Tuple[np.ndarray, Dict[str, int]]:
    key, shard = _find_embed_key_and_shard(repo_id, revision, cache_dir, hf_token)
    shard_path = hf_hub_download(repo_id, filename=shard, revision=revision,
                                 token=hf_token, cache_dir=cache_dir, local_dir_use_symlinks=False)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise RuntimeError(f"[{repo_id}] Embedding key not found in shard {shard}; composite tensors not handled in this simple path.")
        t = f.get_tensor(key)

    if dtype == "float32":
        t = t.float()
    elif dtype == "float16":
        t = t.half()
    elif dtype == "bfloat16":
        t = t.bfloat16()

    emb = t.cpu().numpy()
    info = {"vocab_size": emb.shape[0], "dim": emb.shape[1], "key": key, "shard": shard}
    return emb, info

def ortho_part(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return (U @ Vt).astype(np.float32, copy=False)

def haar_rowmax_expected(n: int) -> float:
    return math.sqrt(2.0 * math.log(n) / n)

def build_cross_gram(embA: np.ndarray, embB: np.ndarray,
                     pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(pairs) == 0:
        raise ValueError("No common tokens found; cannot build cross Gram matrix.")
    idsA = np.array([p[0] for p in pairs], dtype=np.int64)
    idsB = np.array([p[1] for p in pairs], dtype=np.int64)
    EA = embA[idsA, :]
    EB = embB[idsB, :]
    # center + row-normalize (robustness)
    EA = EA - EA.mean(axis=0, keepdims=True)
    EB = EB - EB.mean(axis=0, keepdims=True)
    EA /= (np.linalg.norm(EA, axis=1, keepdims=True) + 1e-8)
    EB /= (np.linalg.norm(EB, axis=1, keepdims=True) + 1e-8)
    A = EB.T @ EA   # (d, d)
    return A.astype(np.float32, copy=False)

def rowmax_diagnostic(U: np.ndarray) -> Dict[str, float]:
    n = U.shape[1]
    row_max = np.max(np.abs(U), axis=1)
    stats_dict = {
        "n": n,
        "row_max_mean": float(row_max.mean()),
        "row_max_median": float(np.median(row_max)),
        "row_max_q95": float(np.quantile(row_max, 0.95)),
        "row_max_q99": float(np.quantile(row_max, 0.99)),
        "haar_expected_row_max": float(haar_rowmax_expected(n)),
        "rows_exceed_5x_expected": int((row_max > 5.0 * haar_rowmax_expected(n)).sum()),
    }
    return stats_dict

def entry_distribution_diagnostic(U: np.ndarray, sample: int = 200000) -> Dict[str, float]:
    n = U.shape[0]
    data = U.flatten()
    if sample is not None and sample < data.size:
        rng = np.random.default_rng(0)
        data = rng.choice(data, size=sample, replace=False)
    mu = float(np.mean(data))
    var = float(np.var(data))
    target_var = 1.0 / n
    res = {
        "n": n,
        "entry_mean": mu,
        "entry_var": var,
        "target_var_1_over_n": target_var,
        "var_ratio_emp_over_target": var / target_var if target_var > 0 else float("nan"),
    }
    if stats is not None and var > 0:
        z = (data - mu) / (math.sqrt(var) + 1e-12)
        ks_stat, ks_p = stats.kstest(z, "norm")
        res["ks_z_stat"] = float(ks_stat)
        res["ks_z_pvalue"] = float(ks_p)
    return res

def bootstrap_trace_diagnostic(embA, embB, all_pairs, subset: int, trials: int = 20, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(trials):
        pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=subset, replace=False)]
        A = build_cross_gram(embA, embB, pairs)
        U = ortho_part(A)
        traces.append(float(np.trace(U)))
    traces = np.array(traces, dtype=np.float64)
    res = {
        "num_trials": trials,
        "subset_size": subset,
        "trace_mean": float(traces.mean()),
        "trace_var": float(traces.var(ddof=1)),
    }
    if stats is not None and traces.std(ddof=1) > 0:
        ks_stat, ks_p = stats.kstest((traces - traces.mean()) / (traces.std(ddof=1) + 1e-12), "norm")
        res["ks_stat_vs_N01"] = float(ks_stat)
        res["ks_p_vs_N01"] = float(ks_p)
    return res

def permutation_null_diagnostic(embA, embB, pairs, permutes: int = 50, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    idsA = np.array([p[0] for p in pairs], dtype=np.int64)
    idsB = np.array([p[1] for p in pairs], dtype=np.int64)

    traces = []
    rowmax_means = []
    for _ in range(permutes):
        shuffled = rng.permutation(idsB)
        perm_pairs = list(zip(idsA.tolist(), shuffled.tolist()))
        A = build_cross_gram(embA, embB, perm_pairs)
        U = ortho_part(A)
        traces.append(float(np.trace(U)))
        rm = np.max(np.abs(U), axis=1).mean()
        rowmax_means.append(float(rm))

    return {
        "permutes": permutes,
        "null_trace_mean": float(np.mean(traces)),
        "null_trace_std": float(np.std(traces, ddof=1)),
        "null_rowmax_mean_mean": float(np.mean(rowmax_means)),
        "null_rowmax_mean_std": float(np.std(rowmax_means, ddof=1)),
    }

def analyze_pair(modelA: str, modelB: str, max_common: int, subset: Optional[int],
                 hf_token: Optional[str], cache_dir: str, revisionA: Optional[str] = None,
                 revisionB: Optional[str] = None, permutes: int = 50, bootstraps: int = 0,
                 seed: int = 0, normalize_tokens: bool = True) -> Dict[str, object]:
    print(f"\n=== Analyzing pair ===\nA: {modelA}\nB: {modelB}\n")
    tokA = AutoTokenizer.from_pretrained(modelA, use_fast=True, token=hf_token)
    tokB = AutoTokenizer.from_pretrained(modelB, use_fast=True, token=hf_token)

    pairs_all = intersect_tokens(tokA, tokB, max_common=max_common, use_normalize=normalize_tokens)
    print(f"Common tokens (after normalization): {len(pairs_all)}")

    embA, infoA = load_input_embeddings(modelA, cache_dir=cache_dir, hf_token=hf_token, revision=revisionA)
    embB, infoB = load_input_embeddings(modelB, cache_dir=cache_dir, hf_token=hf_token, revision=revisionB)
    print(f"EmbA shape: {embA.shape}  key={infoA['key']} | EmbB shape: {embB.shape}  key={infoB['key']}")

    if subset is not None and subset < len(pairs_all):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs_all), size=subset, replace=False)
        pairs = [pairs_all[i] for i in idx]
    else:
        pairs = pairs_all

    A = build_cross_gram(embA, embB, pairs)
    U = ortho_part(A)

    diag_rowmax = rowmax_diagnostic(U)
    diag_entry = entry_distribution_diagnostic(U)

    diag_boot = None
    if bootstraps and subset is not None:
        diag_boot = bootstrap_trace_diagnostic(embA, embB, pairs_all, subset=subset, trials=bootstraps, seed=seed)

    diag_perm = permutation_null_diagnostic(embA, embB, pairs, permutes=permutes, seed=seed)

    return {
        "models": {"A": modelA, "B": modelB},
        "embedding_info": {"A": infoA, "B": infoB},
        "counts": {"common_tokens": len(pairs_all), "used_pairs": len(pairs)},
        "rowmax_diag": diag_rowmax,
        "entry_diag": diag_entry,
        "bootstrap_trace_diag": diag_boot,
        "perm_null_diag": diag_perm,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="Space-separated list of modelA,modelB pairs (comma separated).")
    ap.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.cache/huggingface/mdir_haar"))
    ap.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None))
    ap.add_argument("--max-common", type=int, default=20000, help="Max number of common tokens to use.")
    ap.add_argument("--subset", type=int, default=12000, help="Subset of common tokens per analysis (for compute).")
    ap.add_argument("--permutes", type=int, default=50, help="Permutation null repeats.")
    ap.add_argument("--bootstraps", type=int, default=0, help="Optional bootstrap trials for Tr(U~).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-normalize-tokens", action="store_true",
                    help="Disable token string normalization when matching cross-token vocab.")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    reports = []
    for p in args.pairs:
        if "," not in p:
            raise SystemExit(f"Bad --pairs item: '{p}'. Expected 'modelA,modelB'.")
        a, b = p.split(",", 1)
        rep = analyze_pair(
            modelA=a.strip(), modelB=b.strip(),
            max_common=args.max_common,
            subset=args.subset,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            permutes=args.permutes,
            bootstraps=args.bootstraps,
            seed=args.seed,
            normalize_tokens=not args.no_normalize_tokens,
        )
        reports.append(rep)

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=120, sort_dicts=False)
    for r in reports:
        print("\n================ RESULTS ================\n")
        pp.pprint(r)

if __name__ == "__main__":
    main()
