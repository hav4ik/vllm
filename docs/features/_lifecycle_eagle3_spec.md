# Eagle3 Speculative Decoding Lifecycle in vLLM v1

This is a focused trace of the **Eagle3** speculative decoding path in vLLM v1,
from server-config parsing through per-step propose/verify, the rejection
sampler, the accept-count plumbing, and the broken Mamba2 interaction at the
end. It intentionally ignores the MTP, Medusa, ngram, draft-model, DFlash, and
tree-attention code paths except where they share code with Eagle3.

File root: `/workspace/vllm_nemotron3_eagle3/vllm`.

## 1. Server config -> drafter instantiation

`--speculative-config` (aka `-sc`) is a JSON blob parsed into
`EngineArgs.speculative_config: dict` (`vllm/engine/arg_utils.py:550,1328`).
`EngineArgs.create_speculative_config` (`arg_utils.py:1514`) turns it into a
`SpeculativeConfig` exposing `method` (= `"eagle3"`),
`num_speculative_tokens` (= **K**), `draft_model_config`, and flags
(`disable_padded_drafter_batch`, `parallel_drafting`, ...). K implies every
request produces `num_query_per_req = 1 + K` candidate tokens per step: K
drafts plus one bonus slot.

The drafter is instantiated in `GPUModelRunner.__init__` on the last PP rank
(`vllm/v1/worker/gpu_model_runner.py:514-577`). Eagle branch:

```
elif self.speculative_config.use_eagle():
    self.drafter = EagleProposer(self.vllm_config, self.device, self)
    if self.speculative_config.method == "eagle3":
        self.use_aux_hidden_state_outputs = (
            self.drafter.eagle3_use_aux_hidden_state
        )
```

(line 557-562), and right after, `self.rejection_sampler =
RejectionSampler(self.sampler)` at line 577.

`EagleProposer` (`vllm/v1/spec_decode/eagle.py:60`, subclass of
`SpecDecodeBaseProposer`) owns: the draft `nn.Module` `self.model` (loaded at
`gpu_model_runner.py:4758-4760`), preallocated persistent buffers
(`self.input_ids`, `self.positions`, `self.hidden_states`), the draft
attention metadata builders (`self.draft_attn_groups`), its own cudagraph
dispatcher, and `self.num_speculative_tokens = K` (`eagle.py:79`). The draft
shares the target's paged KV allocator but has its own attention groups.

## 2. Per-step proposal

Each step, the verifier forward runs inside `execute_model`, then sampling
and drafting happen in `sample_tokens` (`gpu_model_runner.py:4111+`). The
closure `propose_draft_token_ids` (line 4176-4190) captures the forward-time
state and calls `GPUModelRunner.propose_draft_token_ids(...)`:

```
self._draft_token_ids = self.propose_draft_token_ids(
    scheduler_output, sampled_token_ids, sampling_metadata,
    hidden_states, sample_hidden_states, aux_hidden_states,
    spec_decode_metadata, spec_decode_common_attn_metadata, slot_mappings)
self._copy_draft_token_ids_to_cpu(scheduler_output)
```

For Eagle with padded-drafter-batch enabled (line 4200-4230), `use_gpu_toks
= True`, and the closure runs immediately with the GPU `sampled_token_ids`
tensor - no D2H round trip. `_draft_token_ids` is cached on the runner;
`_copy_draft_token_ids_to_cpu` ships it to the scheduler, which stages it as
next step's `scheduler_output.scheduled_spec_decode_tokens[req_id]`.

The outer dispatcher `GPUModelRunner.propose_draft_token_ids`
(`gpu_model_runner.py:4599+`) assembles Eagle3 inputs. With
`use_aux_hidden_state_outputs` it concatenates captured aux states along the
hidden dim (`gpu_model_runner.py:4651-4653, 4687-4691`):

```
target_hidden_states = torch.cat(
    [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
```

then invokes `self.drafter.propose(target_token_ids, target_positions,
target_hidden_states, next_token_ids, token_indices_to_sample, ...)` at
line 4703-4714.

`EagleProposer.propose` (`eagle.py:404`) projects the stacked aux states back
to `hidden_size` via `self.model.combine_hidden_states(...)` (line 434-437),
then `set_inputs_first_pass` (line 439-449, body at 646-679) shifts input ids
by one slot and overwrites the last slot per request with `next_token_ids`
(the verifier's just-sampled bonus token). Per-layer/group attn metadata is
built (line 451-453) and the draft model runs once inside `set_forward_context`
(line 463-478); the output is indexed by `token_indices_to_sample` into
`sample_hidden_states` (line 480).

If K == 1 or `parallel_drafting`, one `_greedy_sample` produces the result
(line 483-485). Otherwise the autoregressive loop at line 545 runs K-1 more
iterations: `eagle_step_update_slot_mapping_and_metadata` fuses position +
slot-mapping updates (line 560-569), attn metadata is rebuilt with
`draft_index=token_index+1` (line 598-600), the draft forward runs
(line 623-636), then `_greedy_sample` yields the next token (line 639).
Results are stacked to `[batch_size, K]` at line 643.

Persistent drafter state across steps: `self.model`, draft KV cache, and the
preallocated `input_ids`/`positions`/`hidden_states` buffers that back
cudagraphs.

## 3. Per-step verify

The verifier sees the drafts from the previous step as part of this step's
input batch. The scheduler injected them via
`scheduler_output.scheduled_spec_decode_tokens`, and the input-batch update in
`_update_states` (`gpu_model_runner.py:1185-1253`) stages them into
`InputBatch.spec_decode_tokens`. When `_prepare_inputs` builds this step's
tensors, it calls `_calc_spec_decode_metadata(num_draft_tokens,
cu_num_scheduled_tokens)` (`gpu_model_runner.py:2577-2655`) to produce a
`SpecDecodeMetadata`:

- `logits_indices`: flat indices into the verifier's full logits tensor for
  **every sampled position**, i.e. K+1 indices per request, interleaved.
- `target_logits_indices`: indices of the K draft-aligned positions per
  request (sub-selects from `logits_indices`).
- `bonus_logits_indices`: indices of the +1 "bonus" position per request
  (= last slot per request).
- `cu_num_draft_tokens` / `cu_num_sampled_tokens`: cumulative prefix sums.
- `draft_token_ids`: the K drafts per request, read directly from
  `self.input_ids.gpu[logits_indices]` at an offset of +1 (line 2644-2645).

The worked example in the docstring (line 2583-2590) is worth keeping at
hand. The dataclass is in `vllm/v1/spec_decode/metadata.py` (17 lines total).

During the same `_prepare_inputs` call, the previous step's
`num_accepted_tokens` is fed into the per-attn-group metadata builder via
`extra_attn_metadata_args` (`gpu_model_runner.py:2240-2249`):

```
if use_spec_decode and isinstance(
    builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
):
    extra_attn_metadata_args = dict(
        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
    )
```

Those values were produced by the **previous** step's verifier+rejection
sampler and landed on `self.num_accepted_tokens.gpu` via
`_update_states_after_model_execute` (section 5). They then flow through into
the mamba metadata builder at `vllm/v1/attention/backends/mamba_attn.py:423`:

```
if num_decodes > 0 and self.use_spec_decode and num_accepted_tokens is not None:
    query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
    num_accepted_tokens = num_accepted_tokens[:num_decodes]
```

and eventually end up as `metadata.num_accepted_tokens`
(`mamba_attn.py:469`) and `metadata.state_indices_tensor_d` (line 468). The
latter holds, for each decode request, the K+1 candidate mamba state-cache
slot ids; the former tells the selective_state_update kernel which of those
slots to *read* from and how many consecutive slots to *write*.

## 4. Rejection sampling

`RejectionSampler` (`vllm/v1/sample/rejection_sampler.py:30`) is called from
`_sample` (`gpu_model_runner.py:4156`) when `spec_decode_metadata is not
None`. `RejectionSampler.forward` (line 60-166) takes `SpecDecodeMetadata`,
`draft_probs` (None for Eagle3 since drafts are argmax-sampled with no
recorded probs), the verifier's `logits: [num_tokens + batch_size,
vocab_size]`, and the sampling metadata.

It splits `logits` using the two index tensors:

- `bonus_logits = logits[bonus_logits_indices]`, sampled with
  `predict_bonus_token=True` (line 101-114) - the "free" bonus tokens if the
  entire draft chain is accepted.
- `target_logits = logits[target_logits_indices]`, fp32, passed through
  logits processors + sampling constraints (line 120-139).

Then `rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len,
cu_num_draft_tokens, None, target_logits, bonus_token_ids, ...)` runs
(line 141-150; body at line 350). Because Eagle3 is greedy and
`draft_probs` is None, `rejection_greedy_sample_kernel` (line 388-400) walks
each request's K drafts; accepts drafts that equal the verifier argmax,
replaces the first mismatch with the verifier argmax, and marks all
subsequent slots (and any remaining drafts) as `PLACEHOLDER_TOKEN_ID = -1`.
If all K drafts match, the bonus token lands in slot K. Output shape:
`[batch_size, max_spec_len + 1]` with `-1` sentinels in rejected slots
(line 381-386). This becomes `SamplerOutput.sampled_token_ids`.

## 5. Post-verify state advancement

Immediately after `_sample`, `sample_tokens` calls
`_update_states_after_model_execute(sampler_output.sampled_token_ids, ...)`
(`gpu_model_runner.py:4158-4160`, body at line 1404-1462). For hybrid models
(= NemotronH / Cascade-2) this is where `num_accepted_tokens` is computed
from the `-1` sentinel pattern. Using `argmax` on `(output_token_ids == -1)`
yields the first rejected position per request, which is exactly the accept
count `j` (1 <= j <= K+1):

```
self.num_accepted_tokens.gpu[:num_reqs] = (
    torch.cat([output_token_ids, torch.full((num_reqs, 1), -1, ...)], dim=1)
    == -1
).int().argmax(-1)
```

(line 1423-1440). A trailing `-1` column is appended so that the fully-accept
row (all K drafts good, plus bonus) still finds a `-1` and returns K+1.

Then, depending on `mamba_cache_mode`:

- In `"align"` mode (line 1442-1456): values are synced to CPU and
  `postprocess_mamba` is run to physically move the mamba state. The CPU
  copy is dropped into `self.input_batch.num_accepted_tokens_cpu[i]`.
- In the default path used for our NemotronH setup (line 1457-1462):
  `self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(...,
  non_blocking=True)` and `self.num_accepted_tokens_event.record()`. The
  actual sync happens in the next step during `_prepare_inputs`, guarded by
  `self.num_accepted_tokens_event.synchronize()` at
  `gpu_model_runner.py:1929-1953`.

Three mirrors of `num_accepted_tokens` coexist: (a)
`self.num_accepted_tokens: CpuGpuBuffer` on the runner
(`gpu_model_runner.py:718`), GPU + numpy; GPU filled at line 1423. (b)
`self.input_batch.num_accepted_tokens_cpu_tensor`, pinned host copy D2H'd
at line 1458, read back into (a) at line 1947. (c)
`metadata.num_accepted_tokens`, a slice of (a) passed via
`extra_attn_metadata_args` (line 2245) and landing at `mamba_attn.py:469`.
`num_accepted_tokens_event` is recorded at line 1462 and awaited at
line 1929-1930 of the next step, letting the drafter proceed without
blocking on the previous step's mamba state sync.

**Deferred correction pattern.** `_update_states` (which runs *before* the
step) cannot know the true accept counts yet (previous forward may be
queued async). It optimistically extends `req_state.output_token_ids` with
`-1` placeholders and queues a callback:
`deferred_spec_decode_corrections.append((req_id, optimistic_num_accepted,
req_state))` (`gpu_model_runner.py:1236-1238`). The closure
`correct_spec_decode_token_counts` is built at line 1370-1402 and returned.
Callers stash it as `deferred_state_corrections_fn = self._update_states(...)`
(line 3805), then fire it *after* the current step's forward launch at
`gpu_model_runner.py:4106-4107`. Once the previous forward's
`valid_sampled_token_count` is ready, the closure fixes up
`req_state.num_computed_tokens` and
`self.input_batch.num_computed_tokens_cpu[cur_req_index]` in place
(line 1387-1398).

## 6. Eagle3 specifics

Eagle3 feeds the drafter with **auxiliary hidden states** extracted from
intermediate layers of the verifier, not just the final layer. The flow:

1. At `load_model` time, `use_aux_hidden_state_outputs` is set (runner
   line 559-562), and after loading the drafter, the target model is
   instructed which layers to capture:
   `self.model.set_aux_hidden_state_layers(aux_layers)` at line 4806. The
   `aux_layers` come from either the speculative config
   (`_get_eagle3_aux_layers_from_config`) or the model's default
   (`get_eagle3_default_aux_hidden_state_layers`) at line 4795-4804.
2. During verifier forward, if the model supports Eagle3 it returns
   `(hidden_states, aux_hidden_states)` instead of a bare tensor
   (`gpu_model_runner.py:4031-4038`). `aux_hidden_states` is a list of
   tensors, one per captured layer, each `[num_tokens, hidden_size]`.
3. Before invoking the drafter, those aux states are concatenated along the
   hidden dim: `torch.cat([h[...] for h in aux_hidden_states], dim=-1)` at
   line 4651-4653 / 4687-4691, yielding `[num_tokens, num_aux *
   hidden_size]`.
4. Inside `EagleProposer.propose`, Eagle3 projects that stacked tensor back
   down to `hidden_size` via `self.model.combine_hidden_states(...)`
   (`eagle.py:434-437`). This projector is a trained linear layer inside the
   draft model.
5. The projected hidden states become the drafter's `hidden_states`
   persistent buffer input for its first forward (line 676).
6. Draft forward -> compute_logits -> argmax -> draft_token_ids. Then the
   K-1 autoregressive iterations continue as in section 2.

The `_get_eagle3_use_aux_hidden_state_from_config` helper at `eagle.py:1588`
reads `use_aux_hidden_state` from the draft model config (defaults to True).
Setting it to False falls back to using the verifier's final
`hidden_states` only.

## 7. Key data structures

**`SpecDecodeMetadata`** (`vllm/v1/spec_decode/metadata.py`). A small
dataclass carrying the flattened draft token ids (shape `[num_draft_tokens]`,
sum over batch), the per-request draft count list, cumulative prefix sums
`cu_num_draft_tokens` and `cu_num_sampled_tokens`, and the three index
tensors `target_logits_indices`, `bonus_logits_indices`, `logits_indices`
that gather the right rows out of the verifier logits tensor. `max_spec_len`
is derived in `__post_init__`. Built once per step by
`_calc_spec_decode_metadata`; consumed by the rejection sampler.

**`EagleProposer` state.** Persistent across steps: `self.model` (draft nn),
`self.draft_attn_groups` (list of `AttentionGroup` for drafter layers),
preallocated `self.input_ids`, `self.positions`, `self.hidden_states`,
`self.inputs_embeds`, `self._slot_mapping_buffer`, `self.arange`,
`self.cudagraph_dispatcher`, and the K. For the eagle3/dflash path it also
holds mask/parallel-drafting tensors
(`self.is_rejected_token_mask`, `self.is_masked_token_mask`,
`self.parallel_drafting_hidden_state_tensor`) populated at
`eagle.py:1355-1368`. `self.num_speculative_tokens` and `self.hidden_size`
come from `SpeculativeConfig.draft_model_config`.

**`num_accepted_tokens` buffer.** Three mirrors of one logical per-request
int tensor. (a) `self.num_accepted_tokens: CpuGpuBuffer`
(`gpu_model_runner.py:718`), filled on GPU at line 1423 from the `-1`
sentinel pattern of `sampler_output.sampled_token_ids`, numpy shadow at
`.np`. (b) `self.input_batch.num_accepted_tokens_cpu_tensor` /
`.num_accepted_tokens_cpu`: pinned host copy, D2H at line 1458, H2D'd into
the runner buffer at the start of the next step (line 1947-1950).
(c) `metadata.num_accepted_tokens` inside each Mamba2/GDN
`AttentionMetadata`, a slice of (a) passed via
`extra_attn_metadata_args` at line 2245, used by the kernel at
`mamba_ssm.py:157`. Sync is governed by
`self.num_accepted_tokens_event` (record at line 1462, sync at
line 1929-1930) so async scheduling holds.

## 8. Mamba2 + spec decode interaction (the broken bit)

The `selective_state_update` Triton kernel in
`vllm/model_executor/layers/mamba/ops/mamba_ssm.py` already supports the
Eagle3-style read/write pattern. The constexpr `IS_SPEC_DECODING` autotunes
on `num_accepted_tokens_ptr is not None` (line 56). The init block
(line 155-174):

```
if HAS_STATE_BATCH_INDICES:
    if IS_SPEC_DECODING:
        num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
        init_token_idx = tl.maximum(num_accepted - 1, 0)
    else:
        init_token_idx = 0
    ...
    state_batch_indices_ptr += (
        pid_b * stride_state_indices_batch
        + init_token_idx * stride_state_indices_T
    )
```

Layout contract: `state_batch_indices` is `[batch, K+1]`. Column `i` of row
`pid_b` holds the cache-block id of the mamba state produced after consuming
the i-th candidate token (i=0 = pre-spec, i=K = post-bonus). On step N+1 the
kernel picks column `num_accepted - 1` - accept count `j` means the
verifier actually consumed `j` candidate tokens, so the "current" state sits
at slot `j-1`. The `max(-, 0)` clamp guards the `j == 0` edge.

In the main `seq_len` loop (line 216), the kernel writes K+1 consecutive
output states, one per candidate token (line 257-270):

```
if IS_SPEC_DECODING:
    dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
    token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
    if token_dst_idx != null_block_id:
        token_dst_ptrs = state_ptr_base + token_dst_idx * stride_state_batch + ...
        tl.store(token_dst_ptrs, state.to(...), mask=mask)
```

Full round-trip contract:

1. Metadata builder packs K+1 mamba state-cache slot ids into
   `state_indices_tensor_d[batch, 0:K+1]`.
2. Previous step's accept counts land in `metadata.num_accepted_tokens` and
   drive `init_token_idx = num_accepted - 1` to pick the starting state.
3. Kernel runs the recurrence for all K+1 positions, writing each
   intermediate state into the corresponding column.
4. Next step reads column `j-1` and continues.

**Broken piece:** the metadata-builder side, under `mamba_cache_mode ==
"all"` (prefix caching). `vllm/v1/attention/backends/mamba_attn.py:109-124`:

```
if self.vllm_config.cache_config.mamba_cache_mode == "all":
    max_num_blocks = cdiv(...)
    # Speculative decoding not supported with prefix caching,
    # so keep shape consistent with prefill buffer
    self.state_indices_tensor_d: torch.Tensor = torch.empty(
        (self.decode_cudagraph_max_bs, max_num_blocks),
        dtype=torch.int32, device=device)
```

versus the non-PC branch at line 135-140 which correctly sizes it to
`(bs, 1 + self.num_spec_tokens)`. In `"all"` mode the second dim indexes
sequence blocks, not the K+1 candidate axis. And at `mamba_attn.py:413-417`:

```
if self.vllm_config.cache_config.mamba_cache_mode != "all":
    state_indices_tensor_d = state_indices_tensor_d[
        :, : 1 + self.num_spec_tokens]
```

the `[:, :K+1]` trim only applies in non-PC mode; under `"all"` the builder
hands the kernel a `[batch, max_num_blocks]` tensor where the kernel expects
`[batch, K+1]`. That mismatch is the stub behind the "Speculative decoding
not supported with prefix caching" TODO. Fixing PC+Eagle3 requires (a) a
proper `[batch, K+1]` shadow tensor that holds the *last* K+1 scratch block
slots per decode request, (b) ensuring those slots can be safely
overwritten K+1 times per step, and (c) ensuring step N+1's
`num_accepted_tokens` indexes the same column the kernel wrote at step N.
The kernel already handles the K+1 read/write pattern correctly.
