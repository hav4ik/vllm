# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch import nn

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction,
    composed_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

logger = init_logger(__name__)

# Module-level flag so the PC + spec-decode startup warning is logged
# at most once per process, no matter how many MambaMixer2 layers exist.
_PC_SPEC_BOUNDARY_WARNED = False

# Added by the IBM Team, 2024


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
# --8<-- [start:mixer2_gated_rms_norm]
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):
    # --8<-- [end:mixer2_gated_rms_norm]

    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        use_rms_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert self.full_hidden_size % self.tp_size == 0, (
            "Tensor parallel world size must divide hidden size."
        )

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            return x * nn.functional.silu(gate.to(torch.float32)).to(input_dtype)

        if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:
            return self.forward_native(x, gate)

        return rms_norm_gated(
            x,
            self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            norm_before_gate=False,
        )


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures that all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
# --8<-- [start:mamba_mixer2]
@PluggableLayer.register("mamba_mixer2")
class MambaMixer2(MambaBase, PluggableLayer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    # --8<-- [end:mamba_mixer2]

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num heads."
        )

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_groups, "
            "then num_groups must equal 1."
        )

        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            groups = MambaStateShapeCalculator.extra_groups_for_head_shards(
                n_groups, self.tp_size
            )
            self.n_groups = n_groups + groups

        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = intermediate_size + 2 * self.groups_ssm_state_size

        if n_groups % self.tp_size == 0:
            self.conv1d = MergedColumnParallelLinear(
                input_size=conv_kernel_size,
                output_sizes=[
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                ],
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            self.in_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[
                    intermediate_size,
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ],
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
        else:
            # This is the n_groups == 1 case,
            # where we need to duplicate groups if TP>1.

            self.conv1d = ColumnParallelLinear(
                input_size=conv_kernel_size,
                output_size=self.conv_dim,
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            self.in_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=intermediate_size + self.conv_dim + self.num_heads,
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )

            # - because in_proj is a concatenation of 3 weights, we
            #   need to interleave them before sharding
            # - use the custom weight loader mamba_v2_sharded_weight_loader
            #   for conv1d.bias, covn1d.weight and in_proj.weight
            # - need to set these settings, to assign the groups
            #   to the head shards
            group_shard_settings = (
                self.groups_ssm_state_size,  # expected model size
                (self.n_groups - n_groups) * self.ssm_state_size,  # extra dims assigned
                n_groups == 1,  # if there was only one group
            )
            intermediate_settings = (intermediate_size, 0, False)
            head_settings = (self.num_heads, 0, False)

            # - the weight already has a "weight_loader" attribute
            #   which set_weight_attrs will raise if we do not
            #   delete before trying to override it
            # - ditto for the other two weights below
            delattr(self.conv1d.bias, "weight_loader")
            set_weight_attrs(
                self.conv1d.bias,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

            delattr(self.conv1d.weight, "weight_loader")
            set_weight_attrs(
                self.conv1d.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

            # Create the custom weight loader for Mamba sharding with group
            # replication. This handles the interleaved projections correctly.
            mamba_loader = mamba_v2_sharded_weight_loader(
                [
                    intermediate_settings,  # for gate
                    intermediate_settings,
                    group_shard_settings,
                    group_shard_settings,
                    head_settings,  # for dt
                ],
                self.tp_size,
                tp_rank,
            )

            # Apply the custom weight loader to in_proj.weight
            # Works for both non-quantized (Parameter) and quantized
            # (ModelWeightParameter which extends BasevLLMParameter)
            if isinstance(self.in_proj.weight, BasevLLMParameter):
                # For BasevLLMParameter subclasses (quantized layers like FP8)
                # These have a weight_loader property that can be directly set
                self.in_proj.weight.weight_loader = mamba_loader
            else:
                # For standard Parameter (non-quantized layers)
                delattr(self.in_proj.weight, "weight_loader")
                set_weight_attrs(self.in_proj.weight, {"weight_loader": mamba_loader})

        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `MergedColumnParallelLinear`,
        # and `set_weight_attrs` doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.register_buffer("conv_weights", conv_weights, persistent=False)

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float())
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        # - get hidden_states, B and C after depthwise convolution.
        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # The tuple is (conv_state, ssm_state)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.num_spec = vllm_config.num_speculative_tokens

        # ===========================================================
        # PC + Spec decode shadow-slot scratch tensor (NemotronH-only fix).
        #
        # When prefix caching ('all' mode) AND speculative decoding are
        # both enabled, the regular ssm_state pool MUST NOT be used as
        # the kernel's per-token write target — doing so contaminates
        # boundary states cached by the prefix-cache committer when a
        # spec verify step crosses a mamba block boundary.
        #
        # Instead, we allocate a dedicated SCRATCH tensor that is
        # absolutely segregated from the regular pool: it's a separate
        # torch.Tensor on the same device, never registered with
        # BlockPool, never hashed by the radix tree, never visible to
        # MambaManager. The kernel reads/writes to this scratch tensor
        # for spec verify, and after the rejection sampler runs we copy
        # the accepted state back to the canonical block slot in the
        # regular ssm_state pool.
        #
        # Memory cost (per layer):
        #   max_running_seqs * (1 + num_spec) slots * per_slot_size
        # E.g. for max_num_seqs=16, K=4 spec tokens, fp16 ssm state:
        #   16 * 5 * 1.5 MB ≈ 120 MB / layer × 31 layers ≈ 3.7 GB total
        # At fp32 the cost doubles. Allocated lazily on first decode
        # call so we know the device + dtype + actual ssm_state shape.
        #
        # See docs/features/_pc_spec_decode_upstream_status.md for the
        # full design writeup.
        self._spec_scratch_enabled = (
            self.num_spec > 0
            and self.cache_config is not None
            and self.cache_config.mamba_cache_mode == "all"
        )
        self.spec_scratch_ssm_state: torch.Tensor | None = None
        self.spec_scratch_conv_state: torch.Tensor | None = None
        # Per-step pending commit info: tuple of
        #   (canonical_dst_slot_ids, scratch_slot_base)
        # populated at the start of each decode-with-spec-PC forward,
        # consumed by commit_spec_scratch_to_canonical() after the
        # rejection sampler runs.
        self._spec_scratch_pending: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._spec_scratch_enabled:
            scheduler_config = vllm_config.scheduler_config
            self._spec_scratch_max_running_seqs = scheduler_config.max_num_seqs
            self._spec_scratch_slots_per_req = 1 + self.num_spec

            # ONE-TIME startup warning. Log the rough fraction of decode
            # steps that will fall back to greedy (single-token, no spec)
            # because they cross a mamba block boundary. With block size B
            # and K speculative tokens, exactly K+1 consecutive decode
            # steps per block boundary are unsafe (the window where any of
            # the K+1 candidate positions could span the boundary). The
            # disable rate is therefore (K+1) / B.
            #
            # Logged at most once per process via a module-level flag —
            # NemotronH has 31 mamba layers and we'd otherwise spam.
            global _PC_SPEC_BOUNDARY_WARNED
            if not _PC_SPEC_BOUNDARY_WARNED:
                _PC_SPEC_BOUNDARY_WARNED = True
                B = self.cache_config.mamba_block_size
                K1 = self._spec_scratch_slots_per_req
                disable_pct = 100.0 * K1 / B
                enabled_pct = 100.0 - disable_pct
                logger.warning(
                    "PC + speculative decoding enabled with "
                    "mamba_block_size=%d and num_speculative_tokens=%d. "
                    "Speculative decoding will be active for ~%.2f%% of "
                    "decode steps; the remaining ~%.2f%% (within K+1=%d "
                    "steps of each block boundary) will fall back to "
                    "single-token greedy decoding for correctness. See "
                    "docs/features/_pc_spec_decode_three_approaches.md "
                    "for the design rationale.",
                    B,
                    self.num_spec,
                    enabled_pct,
                    disable_pct,
                    K1,
                )

        # Pre-compute sizes for forward pass
        self.tped_intermediate_size = self.intermediate_size // self.tp_size
        self.tped_conv_size = self.conv_dim // self.tp_size
        self.tped_dt_size = self.num_heads // self.tp_size

        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.tped_intermediate_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        # Check if running on Blackwell (SM100+) for kernel tuning
        self.is_blackwell = current_platform.is_device_capability_family(100)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ):
        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        # 2. Prepare inputs for conv + SSM
        ssm_output = torch.empty(
            [
                hidden_states.shape[0],
                (self.num_heads // self.tp_size) * self.head_dim,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 3. conv + SSM
        # (split `projected_states` into hidden_states_B_C, dt in the custom op to
        # ensure it is not treated as an intermediate tensor by torch compile)
        torch.ops.vllm.mamba_mixer2(
            projected_states,
            ssm_output,
            self.prefix,
        )

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        gate = projected_states[..., : self.tped_intermediate_size]
        hidden_states = self.norm(ssm_output, gate)

        # 5. Final linear projection
        output, _ = self.out_proj(hidden_states)

        return output

    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ):
        hidden_states_B_C, dt = torch.split(
            projected_states[..., self.tped_intermediate_size :],
            [self.tped_conv_size, self.tped_dt_size],
            dim=-1,
        )

        forward_context = get_forward_context()
        # attn_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, Mamba2AttentionMetadata)
            # conv_state must be (..., dim, width-1) for the conv kernels.
            # DS layout stores it that way directly; SD layout needs a
            # transpose (which keeps dim contiguous via stride tricks).
            conv_state = (
                self.kv_cache[0]
                if is_conv_state_dim_first()
                else self.kv_cache[0].transpose(-1, -2)
            )
            ssm_state = self.kv_cache[1]
            has_initial_states_p = attn_metadata.has_initial_states_p
            prep_initial_states = attn_metadata.prep_initial_states
            chunk_size = attn_metadata.chunk_size
            seq_idx_p = attn_metadata.seq_idx_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            num_accepted_tokens = attn_metadata.num_accepted_tokens
            query_start_loc_d = attn_metadata.query_start_loc_d
            num_decodes = attn_metadata.num_decodes
            num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata is None:
            # profile run
            hidden_states_B_C = (
                hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)
            ).contiguous()
            hidden_states, _B, _C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
            return hidden_states

        num_prefills = attn_metadata.num_prefills
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        # Split along token dimension
        hidden_states_B_C_d, hidden_states_B_C_p = torch.split(
            hidden_states_B_C[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )
        dt_d, dt_p = torch.split(
            dt[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )

        if is_mamba_cache_all:
            # If prefix caching is enabled, retrieve the relevant variables
            # for prefill and decode
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            # Prefill-only variables:
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
            # Decode-side block_idx_first_scheduled_token: split out from
            # the full per-request tensor that the metadata builder added
            # for the PC + spec decode disable-at-boundary fix. May be None
            # if the metadata builder didn't populate it (e.g., legacy align
            # mode); in that case the spec scratch path will not need it
            # because the safety check (block_idx_last_computed ==
            # block_idx_last_scheduled) is also computed locally and only
            # the unsafe branch references first_scheduled.
            if attn_metadata.block_idx_first_scheduled_token is not None:
                block_idx_first_scheduled_token_d, _ = torch.split(
                    attn_metadata.block_idx_first_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            else:
                block_idx_first_scheduled_token_d = None
        else:
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_computed_token_d = None
            block_idx_first_scheduled_token_d = None
            num_computed_tokens_p = None

        preallocated_ssm_out_d, preallocated_ssm_out_p = torch.split(
            output[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - It will read the initial states for every sequence,
            #   that has "has_initial_states_p" == True,
            #   from "cache_indices", using "state_indices_tensor_p".
            # - It updates the "conv_state" cache in positions pointed
            #   to by "state_indices_tensor_p".
            #   In particular, it will always write the state at the
            #   sequence end.
            #   In addition, "block_idx_first_scheduled_token_p" and
            #   "block_idx_last_scheduled_token_p"
            #   are provided (which are pointers into
            #   "state_indices_tensor_p"), it will write additional cache
            #   states aligned at "block_size_to_align".
            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see
            hidden_states_B_C_p = causal_conv1d_fn(
                x,
                self.conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_p
            )

            # 3. State Space Model sequence transformation
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                kernel_ssm_indices = state_indices_tensor_p
                if is_mamba_cache_all:
                    kernel_ssm_indices = state_indices_tensor_p.gather(
                        1, block_idx_last_computed_token_p.unsqueeze(1)
                    ).squeeze(1)
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[kernel_ssm_indices],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            assert preallocated_ssm_out_p is not None
            varlen_states = mamba_chunk_scan_combined_varlen(
                hidden_states_p.view(
                    num_prefill_tokens, self.num_heads // self.tp_size, self.head_dim
                ),
                dt_p,
                self.A,
                B_p.view(num_prefill_tokens, self.n_groups // self.tp_size, -1),
                C_p.view(num_prefill_tokens, self.n_groups // self.tp_size, -1),
                chunk_size=chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=seq_idx_p,
                cu_seqlens=query_start_loc_p,
                cu_chunk_seqlens=cu_chunk_seqlen_p,
                last_chunk_indices=last_chunk_indices_p,
                initial_states=initial_states,
                return_intermediate_states=is_mamba_cache_all,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(num_prefill_tokens, -1, self.head_dim),
                state_dtype=ssm_state.dtype,
            )

            if is_mamba_cache_all:
                # The chunk_stride is the number of chunks per mamba block
                # e.g., if mamba_block_size = 512 and chunk_size = 256,
                # then chunk_stride = 2
                chunk_stride = mamba_block_size // chunk_size

                # Save state for sequences with more than just final state
                for seq_idx in range(num_prefills):
                    # Block index for the first scheduled token
                    block_idx_first_scheduled_token = block_idx_first_scheduled_token_p[
                        seq_idx
                    ]

                    # Block index for the last scheduled token
                    block_idx_last_scheduled_token = block_idx_last_scheduled_token_p[
                        seq_idx
                    ]

                    # Number of blocks that need to be written
                    n_blocks_to_fill = (
                        block_idx_last_scheduled_token - block_idx_first_scheduled_token
                    )

                    # Skip sequences that don't have any blocks to fill
                    if n_blocks_to_fill == 0:
                        continue

                    # Look up the state indices
                    cache_blocks_to_fill = state_indices_tensor_p[
                        seq_idx,
                        block_idx_first_scheduled_token:block_idx_last_scheduled_token,
                    ]

                    # First chunk index for this sequence
                    if seq_idx == 0:
                        first_chunk = 0
                    else:
                        first_chunk = 1 + last_chunk_indices_p[seq_idx - 1]

                    # First chunk that is aligned on the mamba block boundary
                    first_aligned_chunk = first_chunk + chunk_stride - 1

                    # Calculate the number of computed tokens that were not
                    # already cached
                    num_unaligned_computed_tokens = (
                        num_computed_tokens_p[seq_idx] % mamba_block_size
                    )

                    if num_unaligned_computed_tokens > 0:
                        # If the number of computed tokens is not block aligned,
                        # then we need to shift the index accordingly
                        first_aligned_chunk -= (
                            num_unaligned_computed_tokens // chunk_size
                        )

                    # Get states to write
                    from_where = varlen_states[
                        first_aligned_chunk : first_aligned_chunk
                        + n_blocks_to_fill * chunk_stride : chunk_stride
                    ]

                    # Write the states
                    ssm_state[cache_blocks_to_fill] = from_where

                # For all seqs, store the last state (note: might be partial):
                ssm_state[
                    state_indices_tensor_p.gather(
                        1, block_idx_last_scheduled_token_p.unsqueeze(1)
                    ).squeeze(1)
                ] = varlen_states[last_chunk_indices_p]

            else:
                # update ssm states
                # - varlen state is a (num_prefills, nheads, headdim, dstate)
                #   tensor
                ssm_state[state_indices_tensor_p] = varlen_states

        # Process decode requests
        if has_decode:
            # ================================================================
            # PC + Spec decode shadow-slot scratch path (NemotronH-only fix).
            #
            # When prefix caching is enabled AND speculative decoding is
            # active, we MUST NOT let the kernel write spec-verify
            # intermediate states to slots in the regular ssm_state pool —
            # those slots are committed verbatim to the prefix cache by
            # the radix-tree committer when blocks become full, and would
            # contaminate cached boundary states for future requests.
            #
            # Instead, route reads/writes through self.spec_scratch_ssm_state
            # (a pre-allocated tensor that is absolutely segregated from the
            # regular pool, never visible to BlockPool / MambaManager / the
            # radix tree). After the rejection sampler runs, the worker
            # copies the accepted state back to the canonical block slot in
            # the regular pool — that copy lives in
            # gpu_model_runner._update_states_after_model_execute and uses
            # the per-layer commit_spec_scratch_to_canonical helper below.
            #
            # See docs/features/_pc_spec_decode_upstream_status.md for the
            # full design writeup.
            use_spec_scratch_path = (
                is_mamba_cache_all
                and self._spec_scratch_enabled
                and num_accepted_tokens is not None
            )
            if use_spec_scratch_path:
                # Scratch tensor is eagerly allocated by gpu_model_runner
                # right after bind_kv_cache (so it lives outside any
                # cudagraph capture region). If it's still None here we
                # were skipped — fall through to a NoneType error so
                # the bug is loud, not silent.
                assert self.spec_scratch_ssm_state is not None, (
                    "spec_scratch_ssm_state was not eagerly allocated by "
                    "gpu_model_runner.initialize_kv_cache_tensors. Check "
                    "that the post-bind_kv_cache hook is firing for this "
                    "layer."
                )
                assert block_idx_first_scheduled_token_d is not None, (
                    "block_idx_first_scheduled_token_d is None inside the "
                    "spec scratch path; the metadata builder did not "
                    "populate the full block_idx_first_scheduled_token "
                    "tensor. The disable-at-boundary fix needs it — check "
                    "mamba_attn.py:_compute_common_metadata."
                )
                # ----------------------------------------------------------
                # Cudagraph-safe path with disable-at-boundary handling.
                #
                # Per-request safety check:
                #   unsafe[req] = block_idx_last_computed_token[req]
                #               != block_idx_last_scheduled_token[req]
                # Equivalently: the K+1 candidate positions span a mamba
                # block boundary, so naively writing K+1 post-states to one
                # block's slot would corrupt the prefix-cache contract.
                # Unsafe requests get forced num_accepted=1 in the runner
                # (so only token 0 is accepted) and only token 0's
                # post-state is committed, written to the slot for token
                # 0's block (= block_idx_first_scheduled).
                #
                # All tensors used below are SLICES of pre-allocated layer
                # buffers, sized for max_running_seqs and pre-computed in
                # _init_spec_scratch_ssm_state. No torch.arange, no
                # torch.zeros, no .contiguous() copies, no per-call .long()
                # conversions. The cross-graph-boundary stashed pending
                # tensors all live in stable storage.
                #
                # See docs/features/_pc_spec_decode_three_approaches.md for
                # the off-by-one analysis.
                # ----------------------------------------------------------

                # Per-row safety mask, computed into a STABLE pre-allocated
                # bool buffer (not a fresh allocation) so that:
                #   (a) the cudagraph capture sees a constant tensor address
                #   (b) the runner can read this buffer eagerly AFTER the
                #       captured forward returns to clamp num_accepted to 1
                #       for unsafe requests (the runner-side
                #       _update_states_after_model_execute path).
                unsafe_d_buf = self._spec_scratch_unsafe_mask_bool[:num_decodes]
                torch.ne(
                    block_idx_last_computed_token_d,
                    block_idx_last_scheduled_token_d,
                    out=unsafe_d_buf,
                )
                unsafe_d = unsafe_d_buf

                # ===========================================================
                # 1. Compute the canonical INPUT slot (the slot we read FROM).
                #    For both safe and unsafe, the input slot is the OLD
                #    canonical slot at block_idx_last_computed_token. (For
                #    unsafe case 1, where num_computed_tokens is at a block
                #    boundary, this is the previous block's boundary state.)
                # ===========================================================
                canonical_in_slot_int32_buf = (
                    self._spec_scratch_canonical_in_slot_int32[:num_decodes]
                )
                torch.gather(
                    state_indices_tensor_d,
                    1,
                    block_idx_last_computed_token_d.unsqueeze(1),
                    out=canonical_in_slot_int32_buf,
                )
                canonical_in_slot_long_buf = (
                    self._spec_scratch_canonical_in_slot_long[:num_decodes]
                )
                canonical_in_slot_long_buf.copy_(
                    canonical_in_slot_int32_buf.squeeze(1)
                )

                # ===========================================================
                # 2. Compute the canonical DESTINATION slot (the slot we
                #    commit BACK TO). For SAFE requests this equals
                #    canonical_in_slot. For UNSAFE requests this is
                #    state_indices[req][block_idx_first_scheduled_token[req]]
                #    — i.e., the slot for token 0's block.
                #
                #    Implementation: torch.where to pick the right block
                #    index per request, then a single gather. Both ops are
                #    fully tensorized (no Python branches inside the
                #    captured region).
                # ===========================================================
                canonical_dst_block_buf = (
                    self._spec_scratch_canonical_dst_block_int32[:num_decodes]
                )
                torch.where(
                    unsafe_d,
                    block_idx_first_scheduled_token_d,
                    block_idx_last_computed_token_d,
                    out=canonical_dst_block_buf,
                )
                canonical_dst_slot_int32_buf = (
                    self._spec_scratch_canonical_dst_slot_int32[:num_decodes]
                )
                torch.gather(
                    state_indices_tensor_d,
                    1,
                    canonical_dst_block_buf.unsqueeze(1),
                    out=canonical_dst_slot_int32_buf,
                )
                canonical_dst_slot_long_buf = (
                    self._spec_scratch_canonical_dst_slot_long[:num_decodes]
                )
                canonical_dst_slot_long_buf.copy_(
                    canonical_dst_slot_int32_buf.squeeze(1)
                )

                # ===========================================================
                # 3. Per-batch-position scratch slot base IDs (constant
                #    pre-computed buffers). Long form is used as the index
                #    arg to index_copy_ below.
                # ===========================================================
                scratch_slot_base_long = (
                    self._spec_scratch_slot_base_long[:num_decodes]
                )

                # ===========================================================
                # 4. Pre-step copy: ssm_state[canonical_in_slot] →
                #    scratch[scratch_slot_base + 0].
                #    The kernel reads from scratch_slot_base + init_token_idx
                #    (where init_token_idx comes from num_accepted of the
                #    previous step). Because state_indices_tensor_d_input is
                #    set to scratch_slot_base broadcast across all K+1
                #    columns, the kernel always reads from the +0 entry.
                # ===========================================================
                self.spec_scratch_ssm_state.index_copy_(
                    0,
                    scratch_slot_base_long,
                    ssm_state.index_select(0, canonical_in_slot_long_buf),
                )

                # ===========================================================
                # 5. Stash for the worker's post-verify commit.
                #    The first entry is the DESTINATION slot (per request),
                #    the second is the source slot base (per request). Both
                #    are LONG-typed slices of layer-attribute buffers, so
                #    the storage is stable across the
                #    captured-forward → eager-commit boundary.
                # ===========================================================
                self._spec_scratch_pending = (
                    canonical_dst_slot_long_buf,
                    scratch_slot_base_long,
                )

                # ===========================================================
                # 6. Pre-computed kernel state indices.
                #    Input: all K+1 entries point to scratch_slot_base + 0.
                #    Output: per request, either
                #      [base+0, base+1, ..., base+K]   (safe)
                #    or
                #      [base+0, NULL,   ..., NULL  ]   (unsafe — kernel
                #                                       skips writes for
                #                                       NULL_BLOCK_ID slots)
                #    Combined via torch.where row-wise.
                # ===========================================================
                state_indices_tensor_d_input = (
                    self._spec_scratch_state_indices_input_int32[:num_decodes]
                )
                output_combined_buf = (
                    self._spec_scratch_state_indices_output_combined_int32[
                        :num_decodes
                    ]
                )
                torch.where(
                    unsafe_d.unsqueeze(1),
                    self._spec_scratch_state_indices_unsafe_output_int32[
                        :num_decodes
                    ],
                    self._spec_scratch_state_indices_output_int32[:num_decodes],
                    out=output_combined_buf,
                )
                state_indices_tensor_d_output = output_combined_buf

                # Use the scratch tensor as the "ssm_state" for the kernel
                # call below.
                ssm_state_for_kernel = self.spec_scratch_ssm_state
            elif is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
                # for decode:
                #   block_idx_first_scheduled_token_d ==
                #       block_idx_last_scheduled_token_d
                # at block boundaries:
                #   block_idx_first_scheduled_token_d >
                #       block_idx_last_computed_token_d
                ssm_state_for_kernel = ssm_state
            else:
                # Without caching, read and write in-place to the same blocks:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d
                ssm_state_for_kernel = ssm_state

            # 2. Convolution sequence transformation
            # When the spec scratch path is active, route conv_state
            # through the scratch tensor too: pre-copy canonical →
            # scratch, run the kernel with scratch (non-APC mode so the
            # widened state + num_accepted offset logic handles per-
            # candidate rollback), then commit in the post-step hook.
            # Conv state: use the regular pool directly (no scratch).
            # With prefix cache hits at 0% (upstream issue #38182),
            # contamination of cached conv boundary states doesn't
            # matter — nobody reads them. This eliminates the conv
            # round-trip (pool→scratch→pool) which was suspected of
            # causing the accuracy degradation from 92% → 50%.
            # The SSM state still uses scratch (needed for K+1 slots).
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=(
                    block_idx_last_scheduled_token_d
                ),
                initial_state_idx=block_idx_last_computed_token_d,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=query_start_loc_d,
                max_query_len=state_indices_tensor_d.size(-1),
            )

            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_d
            )

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            assert preallocated_ssm_out_d is not None
            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - When use_spec_scratch_path is True, ssm_state_for_kernel is
            #   self.spec_scratch_ssm_state and the slot indices point into
            #   it. Otherwise it's the regular ssm_state.
            # NOTE: final output is an in-place update of out tensor
            selective_state_update(
                ssm_state_for_kernel,
                hidden_states_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=preallocated_ssm_out_d.view(num_decode_tokens, -1, self.head_dim),
                num_accepted_tokens=num_accepted_tokens,
                cu_seqlens=query_start_loc_d,
                is_blackwell=self.is_blackwell,
                enable_stochastic_rounding=self.cache_config.enable_mamba_cache_stochastic_rounding,
                cache_philox_rounds=self.cache_config.mamba_cache_philox_rounds,
            )

    def _init_spec_scratch_states(
        self,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> None:
        """Eagerly allocate spec-decode scratch tensors for BOTH conv_state
        and ssm_state, plus the shared per-step index buffers.

        Called from gpu_model_runner.initialize_kv_cache_tensors right after
        bind_kv_cache, BEFORE any cudagraph capture has started. This is
        load-bearing: every tensor allocated here must live in the main
        allocator pool (not the cudagraph private pool) so that storage
        addresses are stable across cudagraph captures and replays.

        Both conv_state and ssm_state share the same page in the regular
        mamba pool (stride[0] = page_size_bytes / dtype_size). The scratch
        tensors match this stride so that index_copy_ / index_select between
        scratch and the regular pool preserve the per-slot data layout.

        The conv_state scratch is essential because the causal_conv1d kernel
        (in IS_APC_ENABLED mode) writes the FULL widened conv state — which
        includes speculative candidate inputs — directly to the block's
        canonical slot. Without the scratch, those speculative writes
        contaminate the prefix cache's boundary snapshot and corrupt future
        requests that hit the cache.
        """
        assert self._spec_scratch_enabled
        per_slot_shape = ssm_state.shape[1:]
        # +1 for the unused slot 0 (NULL_BLOCK_ID sentinel — the kernel
        # silently skips reads/writes to slot 0). See the +1 offset in
        # the scratch_slot_base computation in conv_ssm_forward.
        total_slots = (
            self._spec_scratch_max_running_seqs * self._spec_scratch_slots_per_req
            + 1
        )
        # Allocate via raw bytes so we can match the strided layout exactly.
        # The ssm_state stride[0] in elements gives us the page size.
        page_size_elements = ssm_state.stride(0)
        # Compute inner strides from the per_slot_shape (contiguous within
        # a slot — only the leading slot dim has the page-padded stride).
        inner_stride = torch.empty(per_slot_shape).stride()
        target_stride = (page_size_elements, *inner_stride)
        # Total raw element count = total_slots * page_size_elements
        # (this includes the per-slot padding region between
        # consecutive slots, which is wasted but lets us share strides
        # with the regular pool).
        raw = torch.zeros(
            total_slots * page_size_elements,
            dtype=ssm_state.dtype,
            device=ssm_state.device,
        )
        self.spec_scratch_ssm_state = torch.as_strided(
            raw,
            size=(total_slots, *per_slot_shape),
            stride=target_stride,
            storage_offset=0,
        )

        # ===========================================================
        # Conv state scratch (same segregation principle as SSM state).
        # The conv_state in the regular pool has shape
        #   (num_blocks, conv_dim, state_len)
        # where state_len = conv_kernel - 1 + num_spec (widened for spec).
        #
        # IMPORTANT: we use CONTIGUOUS allocation for conv scratch
        # (NOT the page-padded stride used by the regular pool). The
        # page padding is huge (~2.3MB/slot for NemotronH) because it's
        # shared with the SSM state segment, but the conv data is only
        # ~196KB/slot. Page-padded scratch would waste ~5.8GB across 31
        # layers, causing OOM. Contiguous scratch is ~481MB total.
        #
        # PyTorch's index_copy_ / index_select handle the stride
        # mismatch between page-padded pool and contiguous scratch
        # correctly: they copy element-by-element, not raw bytes.
        # ===========================================================
        conv_per_slot_shape = conv_state.shape[1:]
        self.spec_scratch_conv_state = torch.zeros(
            (total_slots, *conv_per_slot_shape),
            dtype=conv_state.dtype,
            device=conv_state.device,
        )

        # ===========================================================
        # Pre-allocated, persistent per-step index buffers.
        #
        # These are sized for the worst-case batch (max_running_seqs)
        # and the forward path slices into them via [:num_decodes]
        # instead of allocating fresh tensors per call. The point is
        # cudagraph compatibility: every replay must reuse the same
        # storage addresses, otherwise we risk capturing stale pointers
        # into freed memory and getting silent corruption (the symptom
        # we observed in the cudagraph-mode AIME-25 regression where
        # accuracy collapsed from ~95% eager → ~74% cudagraph).
        #
        # All buffers live on the same device/stream as ssm_state.
        # ===========================================================
        device = ssm_state.device
        M = self._spec_scratch_max_running_seqs
        K1 = self._spec_scratch_slots_per_req

        # arange(M) — used as the row index for "request i in the running
        # batch". Pre-allocated as int32 because state_indices_tensor_d
        # is int32 in vLLM's attention metadata.
        req_batch_indices = torch.arange(M, device=device, dtype=torch.int32)
        # 1 + arange(M) * (K+1) — the +1 reserves slot 0 for NULL_BLOCK_ID
        # (the kernel skips reads/writes to slot 0; see the long comment in
        # conv_ssm_forward where this used to be computed per-call).
        self._spec_scratch_slot_base_int32 = (
            1 + req_batch_indices * K1
        ).contiguous()
        # Long-typed view of slot_base (used as the dim-0 index argument
        # to index_copy_ / index_select inside the forward and the commit
        # helper). Materializing it once avoids per-call .long() copies.
        self._spec_scratch_slot_base_long = (
            self._spec_scratch_slot_base_int32.long().contiguous()
        )

        # state_indices_input[i, j] = slot_base[i] (broadcast over j).
        # The kernel reads the input state from this slot for every
        # candidate token j ∈ [0, K]. Shape (M, K+1), int32.
        self._spec_scratch_state_indices_input_int32 = (
            self._spec_scratch_slot_base_int32.unsqueeze(1)
            .expand(-1, K1)
            .contiguous()
        )

        # state_indices_output[i, j] = slot_base[i] + j. Shape (M, K+1),
        # int32. The kernel writes the post-token-j state to this slot.
        offsets = torch.arange(K1, device=device, dtype=torch.int32)
        self._spec_scratch_state_indices_output_int32 = (
            self._spec_scratch_slot_base_int32.unsqueeze(1) + offsets.unsqueeze(0)
        ).contiguous()

        # state_indices_output for boundary-crossing ("unsafe") requests:
        # only token 0 gets a real scratch slot; tokens 1..K get
        # NULL_BLOCK_ID so the kernel SKIPS those writes (see
        # mamba_ssm.py:260 — `if token_dst_idx != null_block_id`).
        # We don't want speculative junk written to ANY slot for unsafe
        # requests, because we'll force num_accepted=1 in the runner and
        # only commit token 0's post-state to the canonical pool.
        # Shape (M, K+1), int32. Constant; computed once at startup.
        unsafe_output = torch.zeros(
            (M, K1), device=device, dtype=torch.int32
        )  # NULL_BLOCK_ID == 0 everywhere
        unsafe_output[:, 0] = self._spec_scratch_slot_base_int32  # token 0 → base
        self._spec_scratch_state_indices_unsafe_output_int32 = (
            unsafe_output.contiguous()
        )

        # Buffers to receive the per-call canonical "current"-slot ids that
        # we gather out of state_indices_tensor_d. We need both an int32
        # form (because state_indices_tensor_d is int32 and gather's `out=`
        # tensor must match dtype) and a long form (because index_select /
        # index_copy_ require int64 indices). Pre-allocated to keep storage
        # stable across cudagraph replays.
        #   shape (M, 1) — matches gather output (index has unsqueeze(1))
        self._spec_scratch_canonical_in_slot_int32 = torch.empty(
            (M, 1), device=device, dtype=torch.int32
        )
        #   shape (M,) — long form for use as an index_select / index_copy_ arg
        self._spec_scratch_canonical_in_slot_long = torch.empty(
            (M,), device=device, dtype=torch.long
        )

        # Parallel buffers for the per-call canonical DESTINATION slot ids:
        # the slot we commit token 0's (or token N-1's) post-state TO. For
        # SAFE requests this equals canonical_in_slot — the OLD slot stays
        # live. For UNSAFE (boundary-crossing) requests this is
        # state_indices_tensor_d gathered at block_idx_first_scheduled,
        # i.e., the slot for token 0's block. The long version is what gets
        # stashed into _spec_scratch_pending and read by the commit hook
        # outside the cudagraph capture region — it MUST live in stable
        # storage (this layer attribute), not in some fresh per-call tensor
        # whose backing memory could be reused across cudagraph replays.
        self._spec_scratch_canonical_dst_slot_int32 = torch.empty(
            (M, 1), device=device, dtype=torch.int32
        )
        self._spec_scratch_canonical_dst_slot_long = torch.empty(
            (M,), device=device, dtype=torch.long
        )

        # Buffer for the per-call combined state_indices_tensor_d_output
        # (the result of torch.where mixing safe vs unsafe per-row).
        # Pre-allocated so the kernel input has stable storage. Shape
        # matches both input buffers: (M, K+1), int32.
        self._spec_scratch_state_indices_output_combined_int32 = torch.empty(
            (M, K1), device=device, dtype=torch.int32
        )

        # Buffer for the per-row "destination block index" used as the
        # gather argument when computing canonical_dst_slot. For each row:
        #   safe[req]   → block_idx_last_computed_token[req]
        #   unsafe[req] → block_idx_first_scheduled_token[req]
        # The block-index gather argument has to be int32 (matches
        # state_indices_tensor_d.dtype) to satisfy gather's index dtype
        # constraint.
        self._spec_scratch_canonical_dst_block_int32 = torch.empty(
            (M,), device=device, dtype=torch.int32
        )

        # Per-row "is this request boundary-crossing?" mask. Computed by
        # the layer's forward (inside the captured cudagraph region) using
        # torch.ne(..., out=...) into this stable buffer. The runner reads
        # it back AFTER the captured forward to clamp num_accepted_tokens
        # to 1 for unsafe requests (so commit_spec_scratch_to_canonical
        # picks scratch_slot_base + 0 instead of scratch_slot_base + N - 1,
        # AND so the runner emits exactly 1 output token for those reqs).
        # All mamba layers see the same metadata so the runner only needs
        # to read this buffer from ONE layer per step. Shape (M,) bool.
        self._spec_scratch_unsafe_mask_bool = torch.empty(
            (M,), device=device, dtype=torch.bool
        )

        # Long buffer for the per-step "scratch source slot" indices used
        # by commit_spec_scratch_to_canonical:
        #   src_idx[i] = scratch_slot_base[i] + num_accepted[i] - 1
        # This runs outside cudagraph (in
        # gpu_model_runner._update_states_after_model_execute) so the
        # cudagraph-stability argument doesn't apply, but pre-allocating
        # still avoids allocator churn on the hot path. Shape (M,) long.
        self._spec_scratch_src_indices_long = torch.empty(
            (M,), device=device, dtype=torch.long
        )

    def commit_spec_scratch_to_canonical(
        self,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        """Copy accepted spec scratch slots back to the canonical pool.

        Called by the worker (gpu_model_runner) after the rejection sampler
        determines per-request accept counts AND after the runner has
        already clamped num_accepted_tokens to 1 for boundary-crossing
        ("unsafe") requests. See the disable-at-boundary fix in
        gpu_model_runner._update_states_after_model_execute and
        docs/features/_pc_spec_decode_three_approaches.md option 1.

        Uses the per-step pending info stashed in self._spec_scratch_pending
        by the most recent forward() call. If no spec-scratch path was
        taken (e.g. this layer is in a prefill-only step), this is a no-op.

        This MUST run before the scheduler's cache_blocks() commits any
        block to the prefix cache, otherwise the cache committer will read
        the stale (pre-step) state from the canonical slot. The worker is
        responsible for ordering this correctly.

        Args:
            num_accepted_tokens: int32 tensor (num_decodes,) of accepted
                token counts. For safe requests this is the rejection
                sampler's output (in [1, K+1]). For unsafe requests it has
                been clamped to 1 by the runner. Either way, the slot
                committed is `scratch_slot_base + num_accepted - 1`, and
                the canonical destination slot was pre-computed per
                request inside the forward pass (so unsafe requests
                commit to the FIRST scheduled block's slot, not the OLD
                canonical slot).
        """
        if not self._spec_scratch_enabled:
            return
        if self._spec_scratch_pending is None:
            # No spec-scratch path was taken this step (e.g. prefill-only)
            return
        # Both entries in _spec_scratch_pending are pre-allocated long-typed
        # SLICES of layer-attribute buffers (see _init_spec_scratch_ssm_state).
        # No .long() conversion needed.
        canonical_dst_long, scratch_slot_base_long = self._spec_scratch_pending
        num_decodes = canonical_dst_long.shape[0]

        # src_idx[i] = scratch_slot_base[i] + num_accepted[i] - 1
        # This is the slot containing the state after the i-th request's
        # last accepted token of this step's spec verify. We materialize
        # the result in a pre-allocated long buffer (_spec_scratch_src_indices_long)
        # to avoid per-call allocations.
        src_idx_long = self._spec_scratch_src_indices_long[:num_decodes]
        # Compute into the buffer in-place: copy_ accepts a broadcasted
        # arithmetic expression and handles the int32→int64 cast.
        src_idx_long.copy_(
            scratch_slot_base_long + num_accepted_tokens[:num_decodes] - 1
        )

        assert self.spec_scratch_ssm_state is not None
        ssm_state = self.kv_cache[1]
        # Use index_copy_ instead of fancy-indexed assignment so the dst
        # dim is explicit and no fancy-indexing semantics ambiguity.
        ssm_state.index_copy_(
            0,
            canonical_dst_long,
            self.spec_scratch_ssm_state.index_select(0, src_idx_long),
        )

        # Conv state: NO scratch commit needed. Conv state goes directly
        # through the regular pool (no scratch round-trip). See the
        # comment in the forward path above the causal_conv1d_update call.

        # Clear the pending so a subsequent call (or a no-spec step) is a no-op
        self._spec_scratch_pending = None

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=self.intermediate_size,
            tp_world_size=get_tensor_model_parallel_world_size(),
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
            num_spec=self.num_spec,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba2"


def mamba_mixer2(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.conv_ssm_forward(projected_states=projected_states, output=output)


def mamba_mixer2_fake(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer2",
    op_func=mamba_mixer2,
    mutates_args=["output"],
    fake_impl=mamba_mixer2_fake,
)
