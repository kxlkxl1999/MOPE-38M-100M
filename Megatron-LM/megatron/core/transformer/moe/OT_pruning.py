import numpy as np
import torch

import numpy as np

def ot_based_ensemble_pruning(prob, sigma, k: int):
    """
    prob: (E,)  —— 单个 token 的专家概率向量（mu_i）
    sigma: (E,E) —— 全局专家协方差
    k: 选多少个专家
    返回: 0/1 向量 (E,)
    """
    p = np.asarray(prob,  dtype=np.float64)     # 统一到 numpy
    Sx = np.asarray(sigma, dtype=np.float64)
    E = Sx.shape[0]
    eps = 1e-12

    # 选第一个：argmax p_i^2 / Sigma[i,i]
    diag = np.clip(np.diag(Sx), eps, None)
    scores0 = (p * p) / diag
    S = [int(np.argmax(scores0))]

    # R 的初始化（逆 Cholesky 的对角块）
    R = np.array([[1.0 / np.sqrt(diag[S[0]])]], dtype=np.float64)

    for t in range(1, k):
        # print("====== k=%d =====" % t)
        cand = list(set(range(E)) - set(S))
        best_score = -np.inf
        best_idx = -1
        best_R = None

        # 预取已选的列（避免重复索引开销）
        for j in cand:
            beta_j = Sx[S, j]                    # shape: (t,)
            b_jj  = Sx[j, j]

            alpha_j = R @ beta_j                 # (t,)
            s = float(b_jj - alpha_j @ alpha_j)  # Schur 补
            if s <= eps:                         # 数值稳定
                s = eps
            r_j = 1.0 / np.sqrt(s)               # 标量

            gamma_j = (-r_j * alpha_j.reshape(1, -1)) @ R   # (1, t)

            # 组装扩展后的 R（(t+1)×(t+1)）
            R_j = np.block([
                [R,                      np.zeros((R.shape[0], 1), dtype=np.float64)],
                [gamma_j.astype(np.float64),      np.array([[r_j]], dtype=np.float64)]
            ])

            # 计算分数：|| R_j @ p_sub ||^2
            idx = S + [j]
            p_sub = p[idx].reshape(-1, 1)        # (t+1, 1)
            sc = float((R_j @ p_sub).T @ (R_j @ p_sub))
            if sc > best_score:
                best_score, best_idx, best_R = sc, j, R_j

        S.append(best_idx)
        R = best_R

    out = np.zeros(E, dtype=int)
    out[S] = 1
    # print("final score:", out * prob)
    return out


@torch.no_grad()
def otep_batched(scores: torch.Tensor, sigma: torch.Tensor, k: int, eps: float = 1e-12) -> torch.Tensor:
    """
    批量版 OTEP 选择
    scores: [T, E]  —— 每个 token 的专家概率向量
    sigma : [E, E]  —— 全局专家协方差（对称、半正定，建议已加 jitter）
    k     : 需要选择的专家个数
    返回  : [T, E] 的 bool 路由 mask（每行恰有 k 个 True）
    """
    assert scores.dim() == 2 and sigma.dim() == 2
    T, E = scores.shape
    assert sigma.shape == (E, E)
    device = scores.device
    dtype  = scores.dtype

    # 预计算对角（数值稳定）
    diag = torch.diag(sigma).clamp_min(eps)                     # [E]
    inv_sqrt_diag = diag.rsqrt()                                # 1/sqrt(diag)

    # 结果容器
    selected_mask = torch.zeros(T, E, dtype=torch.bool, device=device)
    selected_idx  = torch.full((T, k), -1, dtype=torch.long, device=device) # 全是-1的矩阵

    # 逆 Cholesky 因子 R 与 z=R p_S 的“打包”缓冲（仅用左上 t×t）
    R = torch.zeros(T, k, k, dtype=dtype, device=device)        # 每个 token 一份 R
    z = torch.zeros(T, k, dtype=dtype, device=device)       # 每个 token 一份 z
    # z_norm_sq = torch.zeros(T, dtype=dtype, device=device)      # ||z||^2

    # ====== 第 0 轮：一次性选第一个专家 j0（向量化） ======
    score0 = (scores ** 2) / diag.unsqueeze(0)                  # [T,E]
    j0 = torch.argmax(score0, dim=1)                            # [T] 列与列之间进行比较，所以返回每一行的最大值
    selected_idx[:, 0] = j0
    selected_mask.scatter_(1, j0.unsqueeze(1), True)

    # 初始化 R、z、||z||^2
    R[:, 0, 0] = inv_sqrt_diag.gather(0, j0)                    # R00 = 1/sqrt(Σ[j0,j0]) torch.gather:根据指定的索引从输入张量中收集元素
    z[:, 0]    = R[:, 0, 0] * scores.gather(1, j0.unsqueeze(1)).squeeze(1)  # z0 = R00 * p_j0 score的第i行都取j0的第i个元素
    z_norm_sq  = z[:, 0] ** 2

    # 到这没问题
    # ====== 其余 k-1 轮：每轮对 (T,E) 一次性打分 ======
    for t in range(1, k): # 1,...,7
        # 取已选索引 S（每个 token 一行，长度 t）
        S_t = selected_idx[:, :t]                               # [T,t]

        # Σ[S, :] ：用高级索引获得 [T,t,E]（每个 token 的 t 行）
        # 注意：sigma[S_t, :] 要求 S_t 的两维是 LongTensor
        beta = sigma[S_t, :]                                    # [T,t,E]

        # α = R * β  （batched： [T,t,t] @ [T,t,E] -> [T,t,E]）
        R_tt = R[:, :t, :t]                                     # [T,t,t]
        alpha = torch.bmm(R_tt, beta)                           # [T,t,E] 把所有列的aj都算了一下

        # s_j = Σ[j,j] - ||α_j||^2
        a2 = alpha.pow(2).sum(dim=1)                            # [T,E]
        s  = diag.unsqueeze(0) - a2                             # [T,E]
        s  = torch.clamp(s, min=eps)
        r  = s.rsqrt()                                          # [T,E]

        # a^T z  （z: [T,k]；alpha: [T,t,E]）
        # 先扩展 z 以匹配 alpha 的 t 维，再在 t 上做内积             ???对吗, 对
        aTz = (alpha * z[:, :t].unsqueeze(2)).sum(dim=1)        # [T,E]

        # 每个候选 j 的得分：||z||^2 + (r*(p_j - a^T z))^2
        delta = scores - aTz                                    # [T,E]
        cand_score = z_norm_sq.unsqueeze(1) + (r * delta).pow(2)  # [T,E]

        # 屏蔽已选过的专家
        cand_score = cand_score.masked_fill(selected_mask, float('-inf'))

        # 选本轮的最佳 j*
        j_star = torch.argmax(cand_score, dim=1)                # [T]
        selected_idx[:, t] = j_star
        selected_mask.scatter_(1, j_star.unsqueeze(1), True)

        # 取出所选 j* 对应的 α*, s*, r*
        arng = torch.arange(T, device=device)
        alpha_sel = alpha[arng, :, j_star]                      # [T,t]
        s_sel     = s[arng, j_star]                             # [T]
        r_sel     = s_sel.rsqrt()                               # [T]

        # 更新 z 的新分量：z_new = r*(p_j - α^T z)
        delta_sel = delta[arng, j_star]                         # [T]
        z_new     = r_sel * delta_sel                           # [T]
        z[:, t]   = z_new
        z_norm_sq = z_norm_sq + z_new.pow(2)

        # 更新 R 的新行/新列（块构造）：
        # gamma = - r * α^T R    （1×t），批量化： [T,1,t] = [T,1,t] @ [T,t,t]
        gamma = - r_sel.view(T, 1, 1) * torch.bmm(alpha_sel.unsqueeze(1), R_tt)  # [T,1,t]

        # 写入 R 的新块
        R[:, t, :t] = gamma.squeeze(1)                          # 下三角新行
        R[:, :t, t] = 0                                         # 上三角新列（全 0）
        R[:, t, t]  = r_sel                                     # 右下角标量
        # 其他元素保持不变（左上 t×t）

    return selected_mask



@torch.no_grad()
def otep_batched_chunked(
    scores: torch.Tensor,     # [T, E]，每个 token 的专家概率
    sigma:  torch.Tensor,     # [E, E]，全局专家协方差
    k: int,
    chunk_size: int = 256,
    eps: float = 1e-12,
) -> torch.BoolTensor:
    """
    低显存版批量 OTEP：按专家维分块（E 方向 chunk）。
    返回 [T, E] 的 bool 路由 mask（每行正好 k 个 True）。
    """
    assert scores.dim() == 2 and sigma.dim() == 2
    T, E = scores.shape
    assert sigma.shape == (E, E)
    device, dtype = scores.device, scores.dtype

    scores = scores.to(device=device, dtype=dtype)
    sigma  = sigma.to(device=device, dtype=dtype)

    # 结果容器
    selected_mask = torch.zeros(T, E, dtype=torch.bool, device=device)
    selected_idx  = torch.full((T, k), -1, dtype=torch.long, device=device)

    # 预计算对角
    diag = torch.diag(sigma).clamp_min(eps)        # [E]
    inv_sqrt_diag = diag.rsqrt()                   # [E]

    # 逆 Cholesky 与 z=R p_S 的打包缓冲（仅用左上 t×t / 前 t 元素）
    R = torch.zeros(T, k, k, dtype=dtype, device=device)   # [T,k,k]
    z = torch.zeros(T, k,     dtype=dtype, device=device)  # [T,k]
    z_norm_sq = torch.zeros(T, dtype=dtype, device=device) # [T]

    # ==================== 第 0 轮：分块在 E 上做 argmax ====================
    best0 = torch.full((T,), float("-inf"), dtype=dtype, device=device)
    j0    = torch.full((T,), -1,           dtype=torch.long, device=device)

    for start in range(0, E, chunk_size):
        J = torch.arange(start, min(start + chunk_size, E), device=device)
        # (scores[:,J]**2) / diag[J]
        s_chunk = scores[:, J]                                # [T,C]
        d_chunk = diag[J]                                     # [C]
        sc0 = (s_chunk * s_chunk) / d_chunk.unsqueeze(0)      # [T,C]
        # 按行更新 argmax
        local_best, local_idx = torch.max(sc0, dim=1)         # [T], [T]
        update = local_best > best0
        j0[update]    = J[local_idx[update]]
        best0[update] = local_best[update]

    # 记录第一个专家
    selected_idx[:, 0] = j0
    selected_mask.scatter_(1, j0.unsqueeze(1), True)

    # 初始化 R、z、||z||^2
    R[:, 0, 0] = inv_sqrt_diag.gather(0, j0)  # R00 = 1/sqrt(Σ[j0,j0])
    z0 = R[:, 0, 0] * scores.gather(1, j0.unsqueeze(1)).squeeze(1)
    z[:, 0] = z0
    z_norm_sq = z0 * z0

    # ==================== 后续 k-1 轮 ====================
    for t in range(1, k):
        S_t = selected_idx[:, :t]                       # [T,t]
        R_tt = R[:, :t, :t]                             # [T,t,t]
        z_t  = z[:, :t]                                 # [T,t]

        best = torch.full((T,), float("-inf"), dtype=dtype, device=device)
        j_star = torch.full((T,), -1, dtype=torch.long, device=device)

        for start in range(0, E, chunk_size):
            J = torch.arange(start, min(start + chunk_size, E), device=device)
            C = J.numel()

            # beta = Σ[S, J]  → 通过列切 + 行 gather 实现： [T,t,C]
            sigma_J = sigma.index_select(1, J)                  # [E,C]
            sigma_J_exp = sigma_J.unsqueeze(0).expand(T, -1, -1)  # [T,E,C]
            row_index = S_t.unsqueeze(-1).expand(T, t, C)       # [T,t,C]
            beta = torch.gather(sigma_J_exp, 1, row_index)      # [T,t,C]

            # alpha = R * beta  （batched）: [T,t,t] @ [T,t,C] -> [T,t,C]
            alpha = torch.bmm(R_tt, beta)                       # [T,t,C]

            # s_j = Σ[j,j] - ||α_j||^2
            a2 = alpha.pow(2).sum(dim=1)                        # [T,C]
            s = diag[J].unsqueeze(0) - a2                       # [T,C]
            s = torch.clamp(s, min=eps)
            r = s.rsqrt()                                       # [T,C]

            # a^T z
            aTz = (alpha * z_t.unsqueeze(2)).sum(dim=1)         # [T,C]

            # cand_score = ||z||^2 + (r*(p_j - a^T z))^2
            p_chunk = scores[:, J]                               # [T,C]
            delta = p_chunk - aTz                                # [T,C]
            sc = z_norm_sq.unsqueeze(1) + (r * delta).pow(2)     # [T,C]

            # 屏蔽已选专家
            sc = sc.masked_fill(selected_mask[:, J], float('-inf'))

            # 在本 chunk 内更新全局最优
            local_best, local_idx = torch.max(sc, dim=1)         # [T]
            update = local_best > best
            j_star[update] = J[local_idx[update]]
            best[update]   = local_best[update]

        # 记录本轮选择
        selected_idx[:, t] = j_star
        selected_mask.scatter_(1, j_star.unsqueeze(1), True)

        # 取出所选 j* 的 alpha*, s*, r* 以更新 R 与 z
        arng = torch.arange(T, device=device)

        # 为了得到 alpha_sel: 需要再次取 Σ[S, j*] 并做一次 alpha = R_tt @ beta_sel
        # 先取 beta_sel: [T,t]
        sigma_j = sigma.index_select(1, j_star)                 # [E,T]
        sigma_j = sigma_j.transpose(0,1)                        # [T,E]
        beta_sel = torch.gather(sigma_j, 1, S_t)                # [T,t]
        alpha_sel = torch.bmm(R_tt, beta_sel.unsqueeze(2)).squeeze(2)  # [T,t]

        # s_sel, r_sel
        s_sel = (diag.gather(0, j_star) - (alpha_sel * alpha_sel).sum(dim=1)).clamp_min(eps)  # [T]
        r_sel = s_sel.rsqrt()                                                                        # [T]

        # z_new
        p_sel = scores.gather(1, j_star.unsqueeze(1)).squeeze(1)                       # [T]
        aTz_sel = (alpha_sel * z_t).sum(dim=1)                                         # [T]
        z_new = r_sel * (p_sel - aTz_sel)                                              # [T]
        z[:, t] = z_new
        z_norm_sq = z_norm_sq + z_new.pow(2)

        # 更新 R 的新行/列
        gamma = - r_sel.view(T, 1, 1) * torch.bmm(alpha_sel.unsqueeze(1), R_tt)        # [T,1,t]
        R[:, t, :t] = gamma.squeeze(1)                                                 # 下三角
        R[:, :t, t] = 0                                                                # 上三角
        R[:, t, t]  = r_sel

    return selected_mask
