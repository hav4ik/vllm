# SPDX-License-Identifier: Apache-2.0
"""
Triton kernel for batched commit of mamba spec-slot state to pool
across all layers in a single kernel launch.

Replaces 23 × 2 = 46 individual index_copy_ calls with 2 kernel launches
(one for SSM, one for conv).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_commit_kernel(
    src_ptrs,       # [num_layers] int64 — base data_ptr of each src tensor
    dst_ptrs,       # [num_layers] int64 — base data_ptr of each dst tensor
    src_slot_ptr,   # [num_commits] int64 — which row to read
    dst_slot_ptr,   # [num_commits] int64 — which row to write
    src_row_bytes,  # bytes between rows in src tensors (stride(0) * element_size)
    dst_row_bytes,  # bytes between rows in dst tensors
    copy_bytes,     # bytes to copy per row (min of src, dst row size)
    BLOCK_SIZE: tl.constexpr,  # elements per thread block (in uint16 = 2 bytes)
):
    layer_idx = tl.program_id(0)
    commit_idx = tl.program_id(1)

    # Load base pointers (as raw byte pointers via uint16)
    src_base = tl.load(src_ptrs + layer_idx).to(tl.pointer_type(tl.uint16))
    dst_base = tl.load(dst_ptrs + layer_idx).to(tl.pointer_type(tl.uint16))

    src_row = tl.load(src_slot_ptr + commit_idx)
    dst_row = tl.load(dst_slot_ptr + commit_idx)

    # Byte offsets to the start of each row
    src_off = src_row * src_row_bytes // 2  # uint16 elements
    dst_off = dst_row * dst_row_bytes // 2

    num_u16 = copy_bytes // 2  # number of uint16 elements to copy

    offsets = tl.arange(0, BLOCK_SIZE)
    for start in tl.range(0, num_u16, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < num_u16
        data = tl.load(src_base + src_off + idx, mask=mask)
        tl.store(dst_base + dst_off + idx, data, mask=mask)


def batch_commit_states(
    layers: list,
    src_slot: torch.Tensor,
    pool_slot: torch.Tensor,
):
    """Commit spec-slot state to pool for ALL layers in one or two
    kernel launches (SSM + conv).

    Args:
        layers: list of MambaMixer2 layers
        src_slot: [num_commits] int64 — spec slot indices to read
        pool_slot: [num_commits] int64 — pool slot indices to write
    """
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        is_conv_state_dim_first,
    )

    num_commits = src_slot.shape[0]
    if num_commits == 0:
        return

    num_layers = len(layers)
    device = src_slot.device

    # --- SSM state commit ---
    spec_ssm_0 = layers[0].spec_ssm
    pool_ssm_0 = layers[0].kv_cache[1]

    # Cache pointer tensors (allocated once, reused every step)
    if not hasattr(batch_commit_states, '_ssm_ptrs'):
        batch_commit_states._ssm_ptrs = (
            torch.tensor([l.spec_ssm.data_ptr() for l in layers],
                         dtype=torch.int64, device=device),
            torch.tensor([l.kv_cache[1].data_ptr() for l in layers],
                         dtype=torch.int64, device=device),
        )
    ssm_src_ptrs, ssm_dst_ptrs = batch_commit_states._ssm_ptrs

    src_row_bytes = spec_ssm_0.stride(0) * spec_ssm_0.element_size()
    dst_row_bytes = pool_ssm_0.stride(0) * pool_ssm_0.element_size()
    # Copy the smaller of the two (spec slots may be smaller than pool)
    copy_bytes = min(src_row_bytes, dst_row_bytes)

    # One-time log
    if not hasattr(batch_commit_states, '_logged'):
        batch_commit_states._logged = True
        import logging
        _log = logging.getLogger(__name__)
        _log.info(
            "batch_commit_states: SSM src_stride=%d dst_stride=%d "
            "copy_bytes=%d match=%s shape_src=%s shape_dst=%s",
            src_row_bytes, dst_row_bytes, copy_bytes,
            src_row_bytes == dst_row_bytes,
            spec_ssm_0.shape, pool_ssm_0.shape,
        )

    # BLOCK_SIZE in uint16 elements
    num_u16 = copy_bytes // 2
    BLOCK = triton.next_power_of_2(min(num_u16, 2048))

    grid = (num_layers, num_commits)
    _batch_commit_kernel[grid](
        ssm_src_ptrs, ssm_dst_ptrs,
        src_slot, pool_slot,
        src_row_bytes=src_row_bytes,
        dst_row_bytes=dst_row_bytes,
        copy_bytes=copy_bytes,
        BLOCK_SIZE=BLOCK,
    )
    # Triton kernel launched for SSM

    # --- Conv state commit ---
    if is_conv_state_dim_first():
        spec_conv_0 = layers[0].spec_conv
        pool_conv_0 = layers[0].kv_cache[0]

        if not hasattr(batch_commit_states, '_conv_ptrs'):
            batch_commit_states._conv_ptrs = (
                torch.tensor([l.spec_conv.data_ptr() for l in layers],
                             dtype=torch.int64, device=device),
                torch.tensor([l.kv_cache[0].data_ptr() for l in layers],
                             dtype=torch.int64, device=device),
            )
        conv_src_ptrs, conv_dst_ptrs = batch_commit_states._conv_ptrs

        src_row_bytes = spec_conv_0.stride(0) * spec_conv_0.element_size()
        dst_row_bytes = pool_conv_0.stride(0) * pool_conv_0.element_size()
        copy_bytes = min(src_row_bytes, dst_row_bytes)
        num_u16 = copy_bytes // 2
        BLOCK = triton.next_power_of_2(min(num_u16, 2048))

        _batch_commit_kernel[grid](
            conv_src_ptrs, conv_dst_ptrs,
            src_slot, pool_slot,
            src_row_bytes=src_row_bytes,
            dst_row_bytes=dst_row_bytes,
            copy_bytes=copy_bytes,
            BLOCK_SIZE=BLOCK,
        )
    else:
        # Transposed conv: can't batch easily, fallback per-layer
        for layer in layers:
            pool_conv = layer.kv_cache[0].transpose(-1, -2)
            spec_conv = layer.spec_conv.transpose(-1, -2)
            pool_conv.index_copy_(
                0, pool_slot,
                spec_conv.index_select(0, src_slot))
