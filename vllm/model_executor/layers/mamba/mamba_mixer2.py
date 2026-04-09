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
_PC_SPEC_V2_WARNED = False

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
        # v2 PC + Spec decode: persistent scratch SSM slots.
        #
        # When mamba_cache_mode="all" (PC) AND spec decode is active,
        # the SSM kernel needs K+1 dedicated candidate slots per request
        # (for its native init_token_idx rollback). The block table
        # only has 1 slot per block — wrong for spec decode indexing.
        #
        # We allocate a separate scratch SSM tensor with K+1 slots per
        # request. States persist in scratch across decode steps (no
        # per-step pool↔scratch round-trip). The pool is only written
        # at block boundaries (to provide correct boundary states for
        # the prefix cache).
        #
        # Conv state uses the regular pool with IS_APC_ENABLED=True
        # (the native APC path handles conv correctly for the boundary
        # writes; spec decode contamination of conv boundary states is
        # handled by the kernel's built-in widened-state rollback).
        # ===========================================================
        self._pc_spec_v2_enabled = (
            self.num_spec > 0
            and self.cache_config is not None
            and self.cache_config.mamba_cache_mode == "all"
        )
        self.spec_scratch_ssm_state: torch.Tensor | None = None
        if self._pc_spec_v2_enabled:
            scheduler_config = vllm_config.scheduler_config
            self._pc_spec_max_seqs = scheduler_config.max_num_seqs
            self._pc_spec_slots_per_req = 1 + self.num_spec
            global _PC_SPEC_V2_WARNED
            if not _PC_SPEC_V2_WARNED:
                _PC_SPEC_V2_WARNED = True
                logger.warning(
                    "PC + spec decode v2 enabled: using persistent scratch "
                    "SSM slots with boundary-only commit. K=%d, block_size=%d.",
                    self.num_spec,
                    self.cache_config.mamba_block_size,
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
        else:
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_computed_token_d = None
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
            # =============================================================
            # v2 PC + Spec decode path: persistent scratch SSM slots.
            #
            # When PC (mamba_cache_mode=all) AND spec decode are both
            # active, the SSM kernel uses persistent scratch slots with
            # K+1 candidates per request (native init_token_idx rollback).
            # The conv kernel uses the regular pool with APC mode.
            #
            # The scratch slots are NEVER round-tripped to the pool per
            # step. States persist in scratch across decode steps. Only
            # at block boundaries (detected post-step in the runner's
            # commit_boundary_states hook) do we copy the boundary state
            # from scratch to the pool for the prefix cache.
            #
            # On the first decode step after prefill (when the scratch
            # is empty), we initialize scratch[base+0] from the pool.
            # We detect this by always copying pool → scratch[base+0];
            # on subsequent steps the kernel reads from base+init_tok_idx
            # (typically > 0), so the copy to base+0 is a wasted no-op.
            # =============================================================
            use_v2_scratch = (
                is_mamba_cache_all
                and self._pc_spec_v2_enabled
                and num_accepted_tokens is not None
            )

            if use_v2_scratch:
                assert self.spec_scratch_ssm_state is not None

                # Canonical pool slot (for init copy + boundary commit)
                canonical_in_slot = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1).long()

                slot_base_long = self._pc_spec_slot_base_long[:num_decodes]

                # Init copy: pool[canonical] → scratch[base+0].
                # This initializes scratch on the FIRST decode step
                # (when scratch[base+0] is zeros). On subsequent steps,
                # it overwrites base+0 with stale pool data — but we
                # fix that immediately below with the in-scratch copy.
                self.spec_scratch_ssm_state.index_copy_(
                    0,
                    slot_base_long,
                    ssm_state.index_select(0, canonical_in_slot),
                )

                # In-scratch promotion: copy the PREVIOUS step's accepted
                # state to scratch[base+0]. This overwrites the stale pool
                # data from the init copy above.
                #
                # On the first step: num_accepted_prev = 1, so this copies
                # scratch[base+0] to scratch[base+0] — a no-op. The init
                # copy's data (from the pool) is preserved. ✓
                #
                # On subsequent steps: num_accepted_prev = N > 0, copies
                # scratch[base+N-1] to scratch[base+0]. The correct
                # accepted state replaces the stale pool data. ✓
                if num_accepted_tokens is not None:
                    prev_accepted_offset = (
                        num_accepted_tokens[:num_decodes].long() - 1
                    ).clamp(min=0)
                    prev_accepted_slot = slot_base_long + prev_accepted_offset
                    self.spec_scratch_ssm_state.index_copy_(
                        0,
                        slot_base_long,
                        self.spec_scratch_ssm_state.index_select(
                            0, prev_accepted_slot
                        ),
                    )

                # SSM kernel uses K+1 scratch slot IDs (native rollback)
                state_indices_tensor_d_input = (
                    self._pc_spec_slot_ids_int32[:num_decodes]
                )
                state_indices_tensor_d_output = (
                    self._pc_spec_slot_ids_int32[:num_decodes]
                )
                ssm_state_for_kernel = self.spec_scratch_ssm_state

                # Stash for boundary commit (block table + old block idx)
                self._pc_spec_pending = (
                    state_indices_tensor_d,
                    block_idx_last_computed_token_d,
                )

            elif is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
                ssm_state_for_kernel = ssm_state
            else:
                # Without caching, read and write in-place to the same blocks:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d
                ssm_state_for_kernel = ssm_state

            # 2. Convolution sequence transformation
            # Conv always uses the regular pool + APC mode (no scratch).
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
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
            # - When use_v2_scratch is True, ssm_state_for_kernel is
            #   the scratch tensor and indices point to scratch slots.
            #   Otherwise it's the regular pool.
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

    def _init_pc_spec_v2_scratch(self, ssm_state: torch.Tensor) -> None:
        """Allocate the persistent scratch SSM tensor for v2 PC+spec.

        Called from gpu_model_runner.initialize_kv_cache_tensors after
        bind_kv_cache, BEFORE any cudagraph capture. The scratch tensor
        matches the pool's stride so index_copy_ works correctly.
        """
        M = self._pc_spec_max_seqs
        K1 = self._pc_spec_slots_per_req
        # +1 for NULL_BLOCK_ID sentinel at slot 0
        total_slots = M * K1 + 1

        per_slot_shape = ssm_state.shape[1:]
        page_size_elements = ssm_state.stride(0)
        inner_stride = torch.empty(per_slot_shape).stride()
        target_stride = (page_size_elements, *inner_stride)
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

        # Pre-computed slot index table: shape (M, K+1) int32.
        # spec_slot_ids[i][j] = 1 + i * K1 + j (skip slot 0 for NULL)
        device = ssm_state.device
        bases = 1 + torch.arange(M, device=device, dtype=torch.int32) * K1
        offsets = torch.arange(K1, device=device, dtype=torch.int32)
        self._pc_spec_slot_ids_int32 = (
            bases.unsqueeze(1) + offsets.unsqueeze(0)
        ).contiguous()
        # Long version for index_copy_/index_select
        self._pc_spec_slot_base_long = bases.long().contiguous()

    def commit_boundary_states(
        self,
        num_accepted_tokens: torch.Tensor,
        num_computed_tokens_d: torch.Tensor,
        block_size: int,
    ) -> None:
        """Commit SSM boundary states to the pool for prefix cache.

        Called from gpu_model_runner after the rejection sampler. For
        each request whose accepted tokens span a block boundary, copy
        the boundary-position state from scratch to the pool.

        Args:
            num_accepted_tokens: (num_decodes,) int32 — accepted count
            num_computed_tokens_d: (num_decodes,) int32 — position before
                this step (= n_done for each request)
            block_size: mamba block size (e.g. 512)
        """
        if not self._pc_spec_v2_enabled or self.spec_scratch_ssm_state is None:
            return

        ssm_state = self.kv_cache[1]
        num_decodes = num_accepted_tokens.shape[0]

        n_done = num_computed_tokens_d[:num_decodes]
        n_accepted = num_accepted_tokens[:num_decodes]

        # Block index before and after this step
        block_before = (n_done - 1).clamp(min=0) // block_size
        block_after = (n_done + n_accepted - 1) // block_size

        # Requests that crossed a boundary
        crossed = (block_before != block_after)
        if not crossed.any():
            return

        # For each crossing request: the boundary position is
        # (block_before + 1) * block_size - 1 = last position of old block.
        # The candidate index for this position is:
        # j = boundary_pos - n_done
        boundary_pos = (block_before + 1) * block_size - 1
        candidate_idx = (boundary_pos - n_done).clamp(min=0, max=self.num_spec)

        # Scratch slot for the boundary state
        slot_base = self._pc_spec_slot_base_long[:num_decodes]
        scratch_slot = (slot_base + candidate_idx.long())

        # Pool slot for the old block (from block table)
        # We need the block table to look up the physical slot.
        # The block table is state_indices_tensor_d from the metadata.
        # We stash it in _pc_spec_pending during forward.
        if not hasattr(self, '_pc_spec_pending') or self._pc_spec_pending is None:
            return
        state_indices_tensor_d, block_idx_old = self._pc_spec_pending

        pool_slot = state_indices_tensor_d.gather(
            1, block_idx_old.unsqueeze(1)
        ).squeeze(1).long()

        # Only commit for crossing requests
        cross_indices = crossed.nonzero(as_tuple=True)[0]
        if cross_indices.numel() == 0:
            return

        scratch_src = scratch_slot[cross_indices]
        pool_dst = pool_slot[cross_indices]

        ssm_state.index_copy_(
            0,
            pool_dst,
            self.spec_scratch_ssm_state.index_select(0, scratch_src),
        )
        self._pc_spec_pending = None

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
