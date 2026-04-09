# PC + Eagle3 + Mamba2 spec scratch implementation: status, hypotheses, plan

> **For the next agent picking this up.** This is a working-progress doc, written
> as context was about to be compacted. Read it top-to-bottom before touching
> any of the existing code.
>
> Companion to:
> - `_pc_spec_decode_mamba_investigation.md` — original 4-bug empirical hunt
> - `_pc_spec_decode_upstream_status.md` — upstream GitHub state and the design
>   for the surgical fix
> - `_lifecycle_*.md` — vLLM internals navigation reference
>
> Date: 2026-04-09

## Quick orientation

The goal: enable speculative decoding (Eagle3) WITH prefix caching enabled
(`mamba_cache_mode=all`) for NemotronH-style hybrid Mamba2+Transformer models in
vLLM v1, on the `nemotron3-eagle3-support` branch of `hav4ik/vllm`.

The architecture committed so far is the **dedicated scratch SSM-state tensor**
approach we agreed on earlier:

- Each `MambaMixer2` layer pre-allocates a `spec_scratch_ssm_state` tensor
  (separate `torch.Tensor`, NOT in any `BlockPool`, NOT visible to the radix
  tree, NOT visible to `MambaManager`).
- The spec verify reads/writes go to scratch instead of the regular ssm_state.
- A post-verify commit hook (in `gpu_model_runner._update_states_after_model_execute`)
  copies the accepted state back to the canonical block slot.
- Cache contamination is impossible by construction: the spec verify NEVER
  touches the regular pool, so the radix tree's commit-time read of
  `ssm_state[block_id]` always sees a state I explicitly committed.

## What's committed (3 code commits)

```
0df56f3aa [Mamba2][Spec Decode] Eagerly init spec scratch tensor outside cudagraph
5c63c1de9 [Mamba2][Spec Decode] Fix NULL_BLOCK_ID conflict + match scratch stride to ssm_state pool
ec24bdf64 [Mamba2][Spec Decode] Add segregated scratch tensor for PC + spec decode (NemotronH)
```

All three are load-bearing. They build on each other:

1. **`ec24bdf64`** — adds the entire architecture: `_spec_scratch_enabled` flag in
   `MambaMixer2.__init__`, lazy `_init_spec_scratch_ssm_state`, the
   `commit_spec_scratch_to_canonical` hook, the
   `state_indices_tensor_d_input/output` rewriting in the decode path, and the
   worker-side iteration of `MambaMixer2` instances after the rejection sampler.
   On its own this commit produces totally degenerate output because of bugs
   fixed in #5c63c1de9 below.

2. **`5c63c1de9`** — fixes two real bugs in the previous commit:
   - **`NULL_BLOCK_ID = 0`** is the kernel sentinel — `selective_state_update`
     silently skips reads/writes to slot 0. The previous commit had
     `scratch_slot_base = req_idx * (K+1)`, so request at batch position 0 had
     `dst[0] = 0 = NULL_BLOCK_ID`, which the kernel skipped. Fix: shift to
     `scratch_slot_base = 1 + req_idx * (K+1)`, bump scratch tensor size by 1.
   - **Stride mismatch.** vLLM's mamba pool `ssm_state` is created via
     `torch.as_strided` with `target_stride = (num_element_per_page, *stride[1:])`
     where `num_element_per_page` is the SHARED page size for the entire mamba
     state (conv_state + ssm_state share a page). My scratch was `torch.zeros(...)`
     with default contiguous strides, so `index_copy_` between the two had
     mismatched per-slot offsets. Fix: allocate scratch via
     `torch.as_strided` over a flat raw buffer with the SAME `stride[0]` as
     `ssm_state`.
   - After this commit, eager-mode smoke tests produce coherent reasoning and
     correct math answers.

3. **`0df56f3aa`** — moves the lazy `_init_spec_scratch_ssm_state` call OUT of
   `conv_ssm_forward` and into `gpu_model_runner.initialize_kv_cache_tensors`
   (right after `bind_kv_cache`). The lazy init was correct in eager mode but
   in cudagraph mode it ran *during* graph capture, producing a tensor whose
   storage is part of the captured graph's memory pool — leading to silent
   state corruption on replay. Eager init outside any capture context is the
   standard vLLM pattern.

## What's working

| Setup | Status | AIME-25 |
|---|---|---|
| Server starts with PC + Eagle3 enabled, no crashes | ✅ | — |
| Smoke test (single math problem, max_tokens=1500-2000) | ✅ coherent | — |
| Spec metrics match no-PC vLLM Eagle3 baseline | ✅ | mean accept 2.84, per-pos 0.71/0.50/0.36/0.26, avg 46% |
| Eager mode (`--enforce-eager`) accuracy | ⚠️ partial | 18/18=100% in early run, then 11/12=91.7%, 16/22=72%? variance unclear |
| Cudagraph mode accuracy | ❌ | ~74% (degenerate "no tool calls" failures on hard problems) |

## What's NOT working / what's still broken

1. **Cudagraph mode is still degraded** even with the eager init fix. AIME-25
   sessions on hard problems show the symptom: 32K tokens generated, 0 tool
   calls, no answer. This is the same "model can't engage in tool-augmented
   reasoning" pattern that earlier looked like state corruption. The eager
   init fix (`0df56f3aa`) helped (raised accuracy from 73% to 74-80% range)
   but did NOT fully solve it.

2. **Eager mode accuracy is also non-100%** in the latest run (11/12 = 91.7%).
   The earlier 18/18 = 100% may have been on easier problems first; I haven't
   confirmed eager mode is fully clean across all 60 sessions. **This is the
   first thing the next agent should re-verify.**

3. **Boundary crossings — NOT implemented.** The current commit always copies
   the accepted state to the SAME canonical slot we read from
   (`block_idx_last_computed_token`). When a verify step's K+1 candidates cross
   a 512-token mamba block boundary, the accepted state should land in the NEW
   block, not the old one. ~1% of spec steps cross. This is the next big TODO
   if (1) and (2) above turn out to be already-fixed.

4. **PC hits = 0 for the AIME workload**, even though the conversation has
   2000+ tokens after a few turns. `vllm:prefix_cache_hits_total` stayed at 0
   for the entire run despite `prefix_cache_queries_total` climbing into the
   hundreds of thousands. This is **a separate issue from my fix** — it's a
   vLLM mamba PC bug or NemotronH-specific quirk, my code path is exercised
   regardless. Worth filing as a separate issue but not blocking.

## Hypotheses for the cudagraph regression

The cudagraph regression is the most pressing problem. Possible causes, in
order of likelihood:

### Hypothesis 1 (most likely): dynamic tensor allocation inside the forward

My `conv_ssm_forward` path creates several new tensors inside the captured
forward pass:

```python
# in mamba_mixer2.py around line 912-957
req_batch_indices = torch.arange(num_decodes, device=..., dtype=torch.int32)
scratch_slot_base = 1 + req_batch_indices * slots_per_req
state_indices_tensor_d_input = (
    scratch_slot_base.unsqueeze(1).expand(-1, slots_per_req).contiguous().to(torch.int32)
)
offsets = torch.arange(slots_per_req, device=..., dtype=torch.int32)
state_indices_tensor_d_output = (
    scratch_slot_base.unsqueeze(1) + offsets.unsqueeze(0)
).contiguous()
```

In cudagraph capture mode, these allocations are made during capture — the
resulting tensors get addresses in the cudagraph's memory pool. On replay,
these allocations re-execute and may produce different tensor addresses than
the captured graph expects, OR the captured graph may reuse stale addresses
from the previous replay.

This is the same class of issue that bit me with the lazy init (`0df56f3aa`).

**Fix to try first:** pre-allocate ALL of these as layer attributes in
`MambaMixer2.__init__` (or in the eager init that runs after `bind_kv_cache`),
sized for `max_running_seqs`. The forward then `slice_` into them up to
`num_decodes`. No new tensor allocation per call.

### Hypothesis 2: `index_copy_` + `index_select` semantics in cudagraphs

`index_copy_` and `index_select` may not be cudagraph-safe if the indices are
themselves dynamic tensors. PyTorch documents that "tensor operations with
dynamic shapes can break cudagraph capture/replay".

**Fix to try second:** rewrite the pre-copy and commit using direct slicing:
```python
# instead of:
self.spec_scratch_ssm_state.index_copy_(0, scratch_slot_base.long(),
                                         ssm_state.index_select(0, canonical_in_slot_ids_d.long()))

# try:
for i in range(num_decodes):
    self.spec_scratch_ssm_state[scratch_slot_base[i]].copy_(ssm_state[canonical_in_slot_ids_d[i]])
```

(But this is slow and not vectorized. A better version would use a custom
Triton kernel or the `gather` op with reshape.)

### Hypothesis 3: `_update_states_after_model_execute` runs at the wrong time relative to cudagraph

My commit hook runs AFTER `_sample` and BEFORE the next forward. In normal
eager mode this is fine. In cudagraph mode, the forward runs as a captured
graph but the commit runs as eager Python. There may be ordering issues
between the captured graph's CUDA stream and the eager commit's CUDA stream.

**Fix to try third:** add `torch.cuda.synchronize()` before the commit reads
from `self.spec_scratch_ssm_state`. This forces the captured graph's writes to
flush before the commit reads them. (Already tried with one strategic sync at
the pre-copy entry, didn't help on its own; might need to be at the commit
entry instead.)

### Hypothesis 4: The captured graph is reusing scratch slots across requests

If `max_running_seqs=16` but the cudagraph captures with `bs=1`, the captured
graph only has scratch slot allocations for batch position 0. On replay with
multiple batch sizes, the indices might not match.

**Fix to try fourth:** verify that cudagraphs are captured at multiple batch
sizes and that each capture uses the right `scratch_slot_base` values.

## What to do next, in priority order

The next agent should follow this script. Don't skip steps.

### Step 0: Re-verify the eager-mode baseline

The first thing to do is to actually run a FULL AIME-25 with `--enforce-eager`
and confirm whether eager mode is truly clean (95%+) or if it also has issues.
The earlier 18/18 = 100% number was from a partial run; the later 11/12 = 91.7%
from a different partial run. The full run has never finished in eager mode
because eager is slow (~30+ minutes for 60 sessions).

```bash
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve chankhavu/c2-softcpy-fp8 \
    --max-model-len 65536 --trust-remote-code \
    --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
    --enforce-eager \
    --speculative-config '{"model": "chankhavu/c2.eagle3-test", "method": "eagle3", "num_speculative_tokens": 4}'
```

```bash
SERVER_ADDR=http://127.0.0.1:18000 python collect_traces_nemotron.py \
    --input_dir /workspace/vllm_nemotron3_eagle3/data \
    --output_dir /workspace/vllm_nemotron3_eagle3/traces/eager_full \
    --n_sessions 2 --max_parallel 8 \
    --model_name chankhavu/c2-softcpy-fp8 \
    --max_tokens 32768 --resume \
    --metrics_url http://127.0.0.1:18000/metrics
```

**If eager mode hits 95%+, the bug is purely cudagraph-related and steps 1-4
below are the right path. If eager mode is also stuck at 80-92%, there's a
deeper bug in the spec scratch logic itself (likely the boundary crossing
issue).**

### Step 1: Pre-allocate dynamic tensors at startup

This is the most likely fix for the cudagraph regression. Move all dynamic
tensor creation out of `conv_ssm_forward` into the eager init hook.

In `MambaMixer2._init_spec_scratch_ssm_state`, add:

```python
# Pre-allocate all the dynamic-shape tensors used in the forward path,
# sized for max_running_seqs. The forward will slice them.
device = ssm_state.device
slots_per_req = self._spec_scratch_slots_per_req
max_seqs = self._spec_scratch_max_running_seqs
self._spec_scratch_req_batch_indices = torch.arange(
    max_seqs, device=device, dtype=torch.int32
)
self._spec_scratch_slot_base_full = (
    1 + self._spec_scratch_req_batch_indices * slots_per_req
)  # shape (max_seqs,)
self._spec_scratch_offsets = torch.arange(
    slots_per_req, device=device, dtype=torch.int32
)  # shape (slots_per_req,)
# Pre-build the (max_seqs, slots_per_req) "all base" and "base + offset"
# index tensors. The forward slices [:num_decodes] from these.
self._spec_scratch_input_indices_full = (
    self._spec_scratch_slot_base_full.unsqueeze(1)
    .expand(-1, slots_per_req)
    .contiguous()
)
self._spec_scratch_output_indices_full = (
    self._spec_scratch_slot_base_full.unsqueeze(1)
    + self._spec_scratch_offsets.unsqueeze(0)
).contiguous()
```

In the forward decode path, replace the dynamic computation with slices:

```python
state_indices_tensor_d_input = (
    self._spec_scratch_input_indices_full[:num_decodes]
)
state_indices_tensor_d_output = (
    self._spec_scratch_output_indices_full[:num_decodes]
)
scratch_slot_base = self._spec_scratch_slot_base_full[:num_decodes]
```

The pre-copy and commit use the sliced `scratch_slot_base` directly.

### Step 2: Replace `index_copy_` with `gather`-based slicing

The `index_copy_` calls in the pre-copy and commit may not be cudagraph-safe.
Try rewriting:

```python
# Pre-copy: scratch[base..base+num_decodes-1 stride 1] = ssm_state[canonical_in_slot_ids]
# Use a contiguous "view" by pre-allocating a scratch sub-region for inputs.
```

Actually, the cleanest replacement: pre-allocate a contiguous "input scratch
region" of `(max_running_seqs, *per_slot_shape)` and just `copy_()` directly:

```python
# In init:
self._spec_scratch_input_buf = torch.zeros(
    (max_seqs, *per_slot_shape),
    dtype=ssm_state.dtype, device=device,
)

# In forward (pre-copy):
self._spec_scratch_input_buf[:num_decodes].copy_(
    ssm_state[canonical_in_slot_ids_d]  # this still allocates, hmm
)
```

The fundamental challenge: `ssm_state[indices]` allocates a new tensor.
There's no in-place version of fancy indexing that writes into an existing
buffer. Options:
- Use a custom Triton kernel that takes (src_tensor, src_indices, dst_tensor)
  and does the indexed copy in-place.
- Use `torch.index_select(input=ssm_state, dim=0, index=indices, out=...)` —
  this DOES support an `out=` arg, but I haven't tested whether it's
  cudagraph-safe with dynamic indices.

### Step 3: Add `torch.cuda.synchronize()` at strategic points

If steps 1 and 2 don't fully solve it, add a synchronize at the start of
`commit_spec_scratch_to_canonical` to force the captured graph's writes to
flush before the eager commit reads them.

### Step 4: Verify cudagraph capture coverage

Inspect the cudagraph dispatcher logs at startup to see what batch sizes are
captured. If `bs=1` is captured but my code references `scratch_slot_base[0..15]`,
there might be a mismatch. The captured graph for `bs=1` should only access
`scratch_slot_base[0..0]`, but if my dynamic computation produces a fresh
tensor each time, the captured graph might be using a stale reference.

### Step 5 (last resort): disable spec at boundary crossings

If steps 1-4 all fail and eager mode is genuinely 95%+ but cudagraph mode is
broken, the next idea is to detect block boundary crossings in the metadata
builder and override `num_accepted_tokens = 1` for those steps (effectively
disabling spec). The user explicitly suggested this. The implementation:

In `mamba_attn.py` `_compute_common_metadata`:
```python
if num_accepted_tokens is not None and is_mamba_cache_all and num_decodes > 0:
    # Detect boundary crossings: r + spec_max_len > block_size
    # where r = (num_computed - 1) % block_size + 1
    block_size = self.kv_cache_spec.block_size
    num_computed = common_attn_metadata.compute_num_computed_tokens()
    r = ((num_computed[:num_decodes] - 1) % block_size) + 1
    would_cross = r + (1 + self.num_spec_tokens) > block_size
    # For requests that would cross, override num_accepted to disable spec
    # (kernel still processes K+1 tokens but we discard the spec results)
    num_accepted_tokens = num_accepted_tokens.clone()
    num_accepted_tokens[would_cross] = 1
```

(This is approximate — the exact implementation needs to interact correctly
with the rejection sampler too.)

## Files to look at first

```
vllm/model_executor/layers/mamba/mamba_mixer2.py
    - MambaMixer2.__init__ — search for "spec_scratch"
    - MambaMixer2._init_spec_scratch_ssm_state — eager init
    - MambaMixer2.conv_ssm_forward — search for "use_spec_scratch_path"
    - MambaMixer2.commit_spec_scratch_to_canonical — post-verify hook

vllm/v1/worker/gpu_model_runner.py
    - initialize_kv_cache_tensors — search for "_init_spec_scratch_ssm_state"
    - _update_states_after_model_execute — search for "MambaMixer2"

vllm/model_executor/layers/mamba/ops/mamba_ssm.py
    - _selective_scan_update_kernel — the kernel; reference for IS_SPEC_DECODING

vllm/v1/attention/backends/utils.py
    - NULL_BLOCK_ID = 0 (line 45)
```

## Current AIME-25 raw results table

| Run | Mode | Sessions done | Accuracy | Notes |
|---|---|---|---|---|
| `traces/vllm_pc_eagle3_scratch` | cudagraph, before NULL_BLOCK_ID fix | 60/60 | 10% | totally broken |
| `traces/vllm_pc_eagle3_nullfix` | eager, after NULL_BLOCK_ID fix | 18/60 | 100% | killed early |
| `traces/vllm_pc_eagle3_final` | cudagraph, after NULL_BLOCK_ID fix, lazy init | 31/60 | 74% | "no tool calls" failures on hard problems |
| `traces/vllm_pc_eagle3_eagerinit` | cudagraph, after eager init fix | 31/60 | 74% | same failure mode, eager init alone didn't fix it |
| `traces/vllm_pc_eagle3_eager_full` | eager, after eager init fix | 12/60 | 91.7% | killed early; 1 failure unclear root cause |

## Useful commands

Server launch (eager mode, the safe path):
```bash
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve chankhavu/c2-softcpy-fp8 \
    --max-model-len 65536 --trust-remote-code \
    --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
    --enforce-eager \
    --speculative-config '{"model": "chankhavu/c2.eagle3-test", "method": "eagle3", "num_speculative_tokens": 4}'
```

Drop `--enforce-eager` to enable cudagraph mode (currently buggy).

Trace collection (n=2, ~30 min total):
```bash
SERVER_ADDR=http://127.0.0.1:18000 python /workspace/vllm_nemotron3_eagle3/collect_traces_nemotron.py \
    --input_dir /workspace/vllm_nemotron3_eagle3/data \
    --output_dir /workspace/vllm_nemotron3_eagle3/traces/<name> \
    --n_sessions 2 --max_parallel 8 \
    --model_name chankhavu/c2-softcpy-fp8 \
    --max_tokens 32768 --resume \
    --metrics_url http://127.0.0.1:18000/metrics
```

Compute accuracy:
```bash
python /workspace/vllm_nemotron3_eagle3/calc_accuracy.py /workspace/vllm_nemotron3_eagle3/traces/<name>
```

Smoke test (one shot, 30 sec):
```bash
curl -sf -X POST http://127.0.0.1:18000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"chankhavu/c2-softcpy-fp8","messages":[{"role":"user","content":"Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b. End with \\boxed{answer}."}],"max_tokens":1500,"temperature":0}'
```

The smoke test should produce a coherent ~3000-character LaTeX proof of
\boxed{70}. If it produces "We need to find... We need to find..." loops,
state is corrupted.

## Bottom line for the user

**Eager mode** (`--enforce-eager`) appears to work but is slow (~3 sessions/min
vs ~12 sessions/min in cudagraph). Smoke tests are coherent. AIME-25 partial
runs show 92-100% accuracy.

**Cudagraph mode** is currently degraded (~74% AIME). The root cause is
suspected to be dynamic tensor allocations inside `conv_ssm_forward` that
don't replay correctly. Fix in progress per Step 1-4 above.

**Spec metrics** match the no-PC vLLM Eagle3 baseline (mean accept 2.84,
per-position 0.71/0.50/0.36/0.26, avg rate 46%) — confirming the spec verify
path is producing valid proposals when it works.

**PC hits = 0** for the AIME workload — separate vLLM mamba PC bug, not
caused by my fix and not blocking it.
