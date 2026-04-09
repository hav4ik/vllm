# Fix: Prefix Cache 0% Hit Rate with Eagle3 on Hybrid Models

> Branch: `pc-spec-v2-nonpc-slots`
> Date: 2026-04-09
> Fix location: `vllm/v1/core/kv_cache_coordinator.py`

## Problem

When Eagle3 speculative decoding is enabled on NemotronH (hybrid Mamba2
model), prefix cache hits are **exactly 0%** despite `--enable-prefix-caching`
being active.

## Root cause

### The Eagle block drop mechanism

When `use_eagle=True`, `FullAttentionManager.find_longest_cache_hit()` drops
the last matched block to force recomputation of hidden states for Eagle's
drafter head:

```python
# vllm/v1/core/single_type_kv_cache_manager.py:457-460
if use_eagle and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()
```

### Why it's catastrophic for hybrid models

For transformer-only models, `block_size=16` so dropping one block loses
only 16 tokens — negligible. But for hybrid models (NemotronH), the block
alignment logic in `_align_hybrid_block_size()` inflates block_size to the
LCM of all attention types. For NemotronH with `mamba_block_size=512`:

```
block_size = 4608  (= 512 * 9, from LCM alignment)
```

With a block size of 4608, most prompts have only **1-2 cacheable blocks**.
The Eagle drop removes the last (often only) block, reducing cache hits to 0.

### Concrete example (from diagnostic logs)

Second request with 8402 tokens (1 full block of 4608):

```
FullAttentionManager: found=1 blocks
→ Eagle drop: 1 → 0 blocks
→ MambaManager: max_length=0, max_num_blocks=0
→ FINAL hit_length=0
```

## The fix

In `HybridKVCacheCoordinator.find_longest_cache_hit()`, pass
`use_eagle=False` to the individual managers for simple hybrid models
(1 full attention group + 1 other group):

```python
use_eagle_for_managers = False if is_simple_hybrid else self.use_eagle
```

### Why this is safe

`KVCacheManager.get_computed_blocks()` already sets:
```python
max_cache_hit_length = request.num_tokens - 1
```

This guarantees at least the last token is always recomputed during
prefill. That recomputation produces the fresh hidden states Eagle needs
for its drafter. The additional block drop is redundant.

## Verification methodology

### Step 1: Diagnostic instrumentation

Added `logger.info()` calls to three critical functions:

1. **`HybridKVCacheCoordinator.find_longest_cache_hit()`** — logged
   `max_cache_hit_length`, `num_block_hashes`, `lcm_block_size`,
   `use_eagle`, `is_simple_hybrid`, `hash_block_size`, and
   `cached_blocks_count` at entry. Logged each iteration's
   `curr_hit_length` after both the reuse and search branches.
   Logged `FINAL hit_length` at exit.

2. **`FullAttentionManager.find_longest_cache_hit()`** — logged
   `max_length`, `block_size`, `max_num_blocks`, number of found
   blocks, miss index, `use_eagle`, `alignment_tokens`, and
   `group_ids`. Logged before/after Eagle drop and alignment drop
   counts when they changed the block count.

3. **`MambaManager.find_longest_cache_hit()`** — logged
   `max_length`, `block_size`, `max_num_blocks`, `alignment_tokens`,
   `use_eagle`, `num_block_hashes`, and `group_ids`. Logged
   individual block hit/miss/alignment-skip events.

### Step 2: Baseline test (no Eagle, no fix)

Server: GPU 0, `--enable-prefix-caching`, `--mamba-cache-mode all`,
`--mamba-block-size 512`, no speculative decoding.

Sent identical 8402-token prompt twice:
- Request 1: `FINAL hit_length=0` (expected, nothing cached)
- Request 2: `FINAL hit_length=4608` (**cache hit!**)
- Confirmed: `block_size=4608`, `hash_block_size=4608`, `use_eagle=False`

### Step 3: Simulated Eagle test (use_eagle forced True, no fix)

Temporarily forced `self.use_eagle = True` in
`HybridKVCacheCoordinator.__init__()`.

Same test:
- Request 2: `found=1 blocks` → Eagle drop → `0 blocks`
- `FINAL hit_length=0` (**0% hit rate!**)

### Step 4: Simulated Eagle test WITH fix

Applied the fix (`use_eagle_for_managers = False`), kept
`self.use_eagle = True` forced.

Same test:
- Request 2: `found=1 blocks` → **no Eagle drop** → `1 block`
- `FINAL hit_length=4608` (**cache hit restored!**)

### Step 5: Actual Eagle3 test (first attempt — fix too narrow)

Started server with actual Eagle3 drafter (`chankhavu/c2.eagle3-test`).
Fix used `is_simple_hybrid` condition. Logs revealed:

```
is_simple=False, n_attn_groups=3,
groups=[('FullAttentionSpec', [23..28]),
        ('MambaSpec', [0..22]),
        ('SlidingWindowSpec', [29])]
```

Eagle3 drafter adds a **SlidingWindowSpec** group (group 29), making
`len(attention_groups) == 3` → `is_simple_hybrid = False` → fix didn't
activate. Cache hits still 0.

### Step 6: Broadened fix — unconditional for all hybrid models

Changed condition from `False if is_simple_hybrid` to unconditional
`False` for all `HybridKVCacheCoordinator` instances. Restarted server.

Logs confirmed: `eagle=True, eagle_mgr=False`

### Step 7: Actual Eagle3 + broadened fix — SUCCESS

Prometheus metrics after sending identical 8402-token prompt twice:

```
vllm:prefix_cache_queries_total: 16804.0
vllm:prefix_cache_hits_total:    4608.0
```

**4608 tokens cached** — exactly 1 block of the 2nd request was a hit.
This is the first time prefix cache hits have been non-zero with Eagle3
on NemotronH.

### Key findings from logs

| Config | use_eagle | eagle_mgr | Req2 hit_length | Status |
|--------|-----------|-----------|-----------------|--------|
| Baseline (no Eagle) | False | False | 4608 | Working |
| Simulated Eagle (no fix) | True | True | 0 | **Broken** |
| Simulated Eagle + fix v1 | True | False | 4608 | **Fixed** |
| Actual Eagle3 + fix v1 | True | True | 0 | **Fix too narrow** |
| Actual Eagle3 + fix v2 | True | False | 4608 | **Fixed!** |

### Prometheus metrics verification

Verified via `curl /metrics`:
- `vllm:prefix_cache_queries_total` = 16804 (2 requests × 8402 tokens)
- `vllm:prefix_cache_hits_total` = 4608 (1 block hit on 2nd request)
- Previously both were 0 with Eagle3

## Interaction with mamba boundary states (v2 spec slot fix)

The prefix cache hit means the mamba state at the block boundary
(position `block_size - 1 = 4607`) is loaded from the cached block
instead of being freshly computed. This state was saved during the
first request's generation via the v2 boundary commit mechanism.

The v2 fix ensures boundary states are correct:
- After rejection sampler, `commit_boundary_states()` copies the
  accepted candidate's state from the spec slot to the pool's block slot
- This write happens BEFORE `cache_blocks()` is called
- So the cached mamba state reflects the correct accepted token sequence

No additional changes are needed for this fix to be safe with the v2
spec slot mechanism.

## Files changed

| File | Change |
|------|--------|
| `vllm/v1/core/kv_cache_coordinator.py` | +15 lines: skip Eagle block drop for simple hybrid models |
