# FULL cudagraph for mixed prefill+decode batches (hybrid mamba)

## Current state

FULL_AND_PIECEWISE mode: FULL for pure decode, PIECEWISE for mixed.
Mamba runs eagerly during PIECEWISE (splitting op). This is correct
but slightly slower than FULL for the mamba layers during mixed steps.

## Why FULL doesn't work for mixed batches

`conv_ssm_forward()` has Python-level branching baked at capture time:

    if use_spec_slots:        # decode: spec slot buffers
        state = self.spec_ssm
        indices = spec_ids
    elif is_mamba_cache_all:  # prefill: pool buffers
        state = ssm_state
        indices = pool_indices

FULL capture bakes one branch. Replay with the other batch type fails.

## Approach A: Dual kernel calls (no kernel changes)

Emit BOTH calls in the captured graph. Zero-length positions early-exit.

    ssm_kernel(self.spec_ssm, decode_qsl, spec_ids, ...)   # decode
    ssm_kernel(ssm_state, prefill_qsl, pool_ids, ...)       # prefill

Metadata in persistent buffers controls which call does real work.
Cost: ~46 extra (mostly no-op) kernel launches across 23 mamba layers.

## Approach B: Unified state buffer (no kernel changes if stride matches)

Allocate spec slots as additional rows at the end of the pool tensor
instead of separate `spec_ssm`/`spec_conv` tensors. Then a single
`state_indices` tensor can mix pool indices and spec-slot indices.

    pool:       rows [0, N_pool)         — block-table indexed
    spec slots: rows [N_pool, N_pool+M*(K+1)+1) — spec_ids indexed

Both paths use the same state tensor and the same kernel call. The
kernels already support arbitrary per-position indices — no changes.

Obstacle: pool rows use LCM-padded stride (block alignment), spec
slots use real state size (contiguous). Must reconcile by allocating
spec slots with padded stride. Changes needed:
  - mamba_mixer2.py: init_spec_slots allocation + index computation
  - mamba_attn.py: state_indices construction for unified buffer

## Approach C: Kernel modification (most flexible)

Modify `selective_state_update` and `causal_conv1d_update` Triton
kernels to accept two state pointers + per-position selector flag.
Most invasive but zero runtime overhead.

## Practical note

FULL_AND_PIECEWISE is already near-optimal for tool-call workloads.
Decode steps (95%+ of wall time) use FULL. Prefill steps are rare
and prefill-dominated. The approaches above only matter if rapid
tool-call turns make PIECEWISE overhead measurable.

## TODO: chunked prefill ablation

Chunked prefill (`max_num_batched_tokens=8192`) increases the number
of PIECEWISE steps (more mixed batches). Benchmark with and without
`--no-chunked-prefill` to quantify impact on throughput.
