# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import torch.distributed as dist
import os
from pathlib import Path
from functools import partial
import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MoESubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher
        # 四种选择只影响数据通讯过程，不影响返回结果
        # 默认 allgather 官方同时推荐在启用 Expert Parallelism（EP）时用 alltoall。
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # 输入 hidden_states: shape = [num_tokens, hidden_size] 或 [batch_size * seq_len, hidden_size]；
        # 返回 output: 与输入 shape 相同，是专家处理后的 token 表示
        # process MoE
        def custom_forward(hidden_states):
            # probs: shape = [num_tokens, num_experts] 每个专家的概率
            # routing_map: shape = [num_tokens, num_experts] 每个 token 被分配给的专家的0-1矩阵
            probs, routing_map = self.router(hidden_states)

            # capture the activated expert ids
            if not self.training and self.config.test_mode:
                if not hasattr(self, "cnts") or not hasattr(self, "rank"):
                    self.cnts, self.rank = 0, torch.distributed.get_rank()
                    self.dump = Path(os.environ["EACT_SAVE"], str(self.layer_number))
                    self.dump.mkdir(parents=True, exist_ok=True)
                values, indices = torch.topk(probs, k=self.config.moe_router_topk)
                torch.save((values, indices), Path(self.dump, f"{self.cnts}-{self.rank}.pt"))
                self.cnts += 1
            # dispatched_input: shape = [num_experts, capacity, hidden_size]；
            # tokens_per_expert: 长度为 num_experts 的列表，表示每个专家获得的 token 数量。
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            # expert_output: shape = [num_experts, capacity, hidden_size]
            # mlp_bias: 专家输出中的 bias（用于残差/融合）
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # output 恢复为 [num_tokens, hidden_size] 或 [batch_size * seq_len, hidden_size]
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            # 作用等价于 PyTorch 的 torch.utils.checkpoint：前向不保存中间激活，反向再把 custom_forward 重跑一遍取回激活，从而显著降低激活显存
            # 被 checkpoint 的 custom_forward 在反向会再执行一次。因此，里面若有“带副作用”的代码（如日志计数、把路由结果写盘、累加 aux loss），
            # 需要小心避免重复。Megatron 的 issue 就报告过在 --moe-layer-recompute 下，负载均衡损失会被累计两次的问题
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


class MoPELayer(BaseMoELayer):
    def __init__(self, config: TransformerConfig, submodules: MoESubmodules = None, layer_number: int = None):
        self.submodules = submodules
        super(MoPELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # ========= NEW: 统计相关成员 =========
        # 每经过多少个 batch 后，在“下一”个 batch 上记录统计；0/None 表示关闭
        self.moe_otep_step: int = int(getattr(config, "moe_otep_step", 0) or 0)
        # 统计产物：均值矩阵/协方差矩阵
        # mu ∈ [E_global, H]；sigma ∈ [E_global, E_global]
        E_local = self.num_local_experts
        try:
            EP_size = parallel_state.get_expert_model_parallel_world_size()
        except AttributeError:
            EP_size = getattr(self.config, "expert_model_parallel_size", 1)
        E_global = E_local * EP_size
        H = self.config.hidden_size
        # 全 0 初始化（CPU，FP32）
        self.moe_mu = torch.zeros(E_global, H, dtype=torch.float32, device="cpu")
        self.moe_sigma = torch.zeros(E_global, E_global, dtype=torch.float32, device="cpu")
        self.moe_mu_decay_rate: float = 0 # 对之前的遗忘率，0代表完全不遗忘，1代表完全遗忘
        self.moe_sigma_decay_rate: float = 0
        # 内部计数与触发标志
        self._moe_batch_count: int = 0
        self._moe_capture_next: bool = False
        self._cov_eps: float = 1e-12
        # ====================================

        # Initialize router
        self.router = TopKRouter(config=self.config)
        # self.router.moe_mu = self.moe_mu.to(self.router.expert_bias.device, non_blocking=True)
        # self.router.moe_sigma = self.moe_sigma.to(self.router.expert_bias.device, non_blocking=True)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def _get_ep_group(self):
        """兼容不同版本的 EP 组 API。"""
        get_group = getattr(parallel_state, "get_expert_model_parallel_group",
                            getattr(parallel_state, "get_expert_parallel_group", None))
        get_rank = getattr(parallel_state, "get_expert_model_parallel_rank",
                           getattr(parallel_state, "get_expert_parallel_rank", None))
        get_size = getattr(parallel_state, "get_expert_model_parallel_world_size",
                           getattr(parallel_state, "get_expert_parallel_world_size", None))
        if get_group is None or get_rank is None or get_size is None:
            raise RuntimeError("Expert-parallel group API not found in megatron.core.parallel_state")
        return get_group(), get_rank(), get_size()
    def forward(self, hidden_states: torch.Tensor):
        if (
                self.training
                and self.config.tensor_model_parallel_size > 1
                and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # ===== NEW: 步进触发逻辑（在第 N,2N,... 个 batch 之后的“下一个” batch 记录） =====
        if self.moe_otep_step and not self._moe_capture_next:
            self._moe_batch_count += 1
            if (self._moe_batch_count % self.moe_otep_step) == 0:
                self._moe_capture_next = True

        # =====================================================================
        def custom_forward(hidden_states):
            # 常规路由 → 常规 MoE 前向（不动）
            probs, routing_map = self.router(hidden_states)

            # 推理测试导出（保持你的原逻辑）
            if not self.training and self.config.test_mode:
                if not hasattr(self, "cnts") or not hasattr(self, "rank"):
                    self.cnts, self.rank = 0, torch.distributed.get_rank()
                    self.dump = Path(os.environ["EACT_SAVE"], str(self.layer_number))
                    self.dump.mkdir(parents=True, exist_ok=True)
                # values, indices = torch.topk(probs, k=self.config.moe_router_topk)
                # torch.save((values, indices), Path(self.dump, f"{self.cnts}-{self.rank}.pt"))
                # self.cnts += 1
                # === 新逻辑：基于 routing_map 而不是 top-k(probs) ===
                # 对每个 token 取 mask==1 的列索引（专家 id）
                selected_indices = routing_map.nonzero(as_tuple=False)  # [N_sel, 2]  (token_idx, expert_idx)
                # 同步保存对应概率，便于后期分析（可选）
                selected_values = probs[selected_indices[:, 0], selected_indices[:, 1]]

                torch.save(
                    (selected_values.cpu(), selected_indices.cpu()),
                    Path(self.dump, f"{self.cnts}-{self.rank}.pt"),
                )
                self.cnts += 1

            # 1) 常规的 token → expert 分发与专家计算（保持不变）
            dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

            # ===== NEW: 仅在触发批次，构造“密集版 dispatched_input_all”做统计 =====
            # “每个 token 经过所有专家”的关键：对每个本地专家都喂入全部 tokens。
            # 为了不影响主路径，这里单独做一次 no_grad 前向（且暂时切 eval 以避免 dropout 噪声）。
            need_capture = self._moe_capture_next and (
                    (not self.moe_layer_recompute) or (self.moe_layer_recompute and (not torch.is_grad_enabled()))
            )
            if need_capture:
                with (torch.no_grad()):
                    E_local = self.num_local_experts
                    T, H = hidden_states.shape

                    # 让每个本地专家看到“全部 token”的密集输入：[E_local, T, H]
                    dense_in = hidden_states.unsqueeze(0).expand(E_local, T, H).contiguous()
                    dense_tpe = [T] * E_local

                    # 暂时切到 eval，避免 dropout 干扰统计；结束后恢复
                    restore_flag = False
                    if self.experts.training:
                        restore_flag = True
                        self.experts.eval()
                    try:
                        dense_out_local, _ = self.experts(dense_in, dense_tpe)  # [E_local, T, H]
                    finally:
                        if restore_flag:
                            self.experts.train(True)

                    ep_group, ep_rank, ep_size = self._get_ep_group()
                    E_local, T, H = dense_out_local.shape
                    E_global = E_local * ep_size

                    # 用 all_gather_into_tensor 一次性收集（要求各 rank 形状/类型一致，且在同一设备上）
                    gather_buf = torch.empty(ep_size, E_local, T, H,
                                             device=dense_out_local.device, dtype=torch.float32)
                    dist.all_gather_into_tensor(gather_buf, dense_out_local.to(torch.float32),
                                                group=ep_group)  # 每 rank 都得到完整数据
                    dense_out_global = gather_buf.view(E_global, T, H)  # [E_global, T, H]

                    # === 计算 mu_global（按 token 平均），并做按-token 的协方差后在 token 上平均 ===
                    mu_global = dense_out_global.mean(dim=1)  # [E_global, H]
                    Xc = dense_out_global - mu_global.unsqueeze(1)  # [E_global, T, H] 逐 token 中心化

                    # Σ = (1 / (T*(H-1))) * sum_{t,h} X[:,t,h] X[:,t,h]^T
                    denom = float(max(H - 1, 1) * T)
                    sigma_buf = torch.empty(E_global, E_global,
                                            device=dense_out_global.device, dtype=dense_out_global.dtype)
                    if ep_rank == 0:
                        sigma_tmp = torch.einsum('eth,fth->ef', Xc, Xc) / denom
                        sigma_tmp.diagonal().add_(self._cov_eps)  # 数值稳定
                        sigma_buf.copy_(sigma_tmp)
                    # 广播给 EP 组所有 rank
                    dist.broadcast(sigma_buf, src=0, group=ep_group)

                    # 存到 CPU（全局）
                    self.moe_mu = (mu_global * T / (T + T * self._moe_batch_count) * (1 + self.moe_mu_decay_rate * self._moe_batch_count)
                    + self.moe_mu * T * self._moe_batch_count/(T + T * self._moe_batch_count) * (1 - self.moe_mu_decay_rate)).detach().cpu()
                    self.moe_sigma = (sigma_buf * denom / (denom + denom * self._moe_batch_count) * (1 + self.moe_sigma_decay_rate * self._moe_batch_count)
                    + self.moe_sigma * denom * self._moe_batch_count/(denom + denom * self._moe_batch_count) * (1 - self.moe_sigma_decay_rate)).detach().cpu()
                    # 注入到本层的路由器，让下一次 forward 用最新统计
                    self.router.moe_mu = self.moe_mu.to(self.router.expert_bias.device, non_blocking=True)
                    self.router.moe_sigma = self.moe_sigma.to(self.router.expert_bias.device, non_blocking=True)
                # 完成本次统计，清除触发标志
                self._moe_capture_next = False
                # ===============================================================

                # expert → token 还原
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, expert计算已在 dispatcher 中重叠
                output = output + self.shared_experts(hidden_states)
            return output, mlp_bias

            # 激活重计算：前向不存激活，反向重跑 custom_forward（注意上面的 need_capture 条件）

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias