# SPDX-License-Identifier: Apache-2.0
"""Triton kernel for batched mamba state commit across all layers."""
import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def _batch_commit_kernel(
    src_ptrs, dst_ptrs, src_slot_ptr, dst_slot_ptr,
    needs_commit_ptr, src_stride_u16, dst_stride_u16, num_u16,
    BLOCK_SIZE: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    if tl.load(needs_commit_ptr + req_idx) == 0:
        return
    src_base = tl.load(src_ptrs + layer_idx).to(tl.pointer_type(tl.uint16))
    dst_base = tl.load(dst_ptrs + layer_idx).to(tl.pointer_type(tl.uint16))
    src_off = tl.load(src_slot_ptr + req_idx) * tl.load(src_stride_u16)
    dst_off = tl.load(dst_slot_ptr + req_idx) * tl.load(dst_stride_u16)
    n = tl.load(num_u16)
    offs = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    for s in tl.range(0, n, BLOCK_SIZE):
        idx = s + offs
        m = idx < n
        tl.store(dst_base + dst_off + idx, tl.load(src_base + src_off + idx, mask=m), mask=m)

_c = None
def _cache(layers, dev):
    global _c
    if _c is not None: return _c
    s0, p0, c0, pc0 = layers[0].spec_ssm, layers[0].kv_cache[1], layers[0].spec_conv, layers[0].kv_cache[0]
    def u(t): return t.stride(0) * t.element_size() // 2
    def d(t): return t[0].numel() * t.element_size() // 2
    T = lambda v: torch.tensor(v, dtype=torch.int64, device=dev)
    _c = {
        'ss': T([l.spec_ssm.data_ptr() for l in layers]), 'sd': T([l.kv_cache[1].data_ptr() for l in layers]),
        'cs': T([l.spec_conv.data_ptr() for l in layers]), 'cd': T([l.kv_cache[0].data_ptr() for l in layers]),
        'sss': T([u(s0)]), 'sds': T([u(p0)]), 'sn': T([d(s0)]),
        'css': T([u(c0)]), 'cds': T([u(pc0)]), 'cn': T([d(c0)]),
        'sb': triton.next_power_of_2(min(d(s0), 2048)),
        'cb': triton.next_power_of_2(min(d(c0), 2048)),
    }
    logger.info("batch_commit: %d layers, SSM=%d bytes", len(layers), s0[0].numel()*s0.element_size())
    return _c

def batch_commit_states(layers, src_slot, pool_slot, needs_commit):
    N = needs_commit.shape[0]
    if N == 0: return
    c = _cache(layers, needs_commit.device)
    mask = needs_commit.to(torch.int8).contiguous()
    g = (len(layers), N)
    _batch_commit_kernel[g](c['ss'],c['sd'],src_slot,pool_slot,mask,c['sss'],c['sds'],c['sn'],BLOCK_SIZE=c['sb'])
    _batch_commit_kernel[g](c['cs'],c['cd'],src_slot,pool_slot,mask,c['css'],c['cds'],c['cn'],BLOCK_SIZE=c['cb'])
