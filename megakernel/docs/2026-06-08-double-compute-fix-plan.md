# Plan: make the kernel REPLACE PyTorch's backbone (kill the double-compute)

## What we proved (live, on the box)
- Kernel backbone call (`_dfh`): **1.08 ms/step** (matches offline 0.78 ms).
- PyTorch backbone `talker.model.forward`: **42.5 ms/step**, runs every step,
  then the kernel post-hook OVERWRITES its result. The 42 ms is wasted.
- code_predictor.generate() (codes 1–15, 15 inner steps): **~173 ms/step** — the
  real bottleneck, runs in PyTorch, kernel does not touch it.
- Everything is on cuda:0 (talker, backbone, code_predictor, codec). Codec was on
  CPU (the original "hang"); fixed in common.py (move speech_tokenizer to GPU).

## Why the double-compute exists
The kernel runs as a `register_forward_hook` (post-hook) on `talker.model`. A
forward hook fires AFTER the module's forward() already ran all 28 layers. The
hook can only overwrite `last_hidden_state`; it cannot prevent the 42 ms compute.

## The structure of talker.forward (decode step)
```
else:  # decode
    last_id_hidden = get_input_embeddings()(input_ids)
    predictor_result = self.code_predictor.generate(           # ~173 ms (codes 1-15)
        inputs_embeds=cat((past_hidden, last_id_hidden), dim=1),
        max_new_tokens=num_code_groups - 1, ...)
    codec_ids = cat((input_ids, predictor_result.sequences), dim=-1)
    ...
outputs = self.model(inputs_embeds=..., past_key_values=..., use_cache=True, ...)  # 42 ms backbone
hidden_states = outputs.last_hidden_state
logits = self.codec_head(hidden_states)         # code 0 logits
return Qwen3TTSTalkerOutputWithPast(
    logits=logits,
    past_key_values=outputs.past_key_values,    # NEEDED next step
    hidden_states=(outputs.hidden_states, codec_ids),
    past_hidden=hidden_states[:, -1:, :],       # NEEDED next step (feeds code_predictor)
    ...)
```

## The fix (replace, don't hook)
Monkeypatch `talker.model.forward` (only when USE_KERNEL=1) so that on a DECODE
step (inputs_embeds seq len == 1) it:
  1. runs the kernel (`_dfh`) on the inputs_embeds -> kernel `_hidden`,
  2. applies `talker.model.norm` -> last_hidden_state (the faithful path),
  3. ADVANCES the PyTorch KV cache by 1 position WITHOUT running 28 layers, so
     `past_key_values.get_seq_length()` stays correct for HF's loop. The kernel
     keeps its OWN kv cache (dec._k_cache); the PyTorch cache only needs its
     length to advance (its contents are unused once the kernel drives decode).
  4. returns a BaseModelOutputWithPast with last_hidden_state + past_key_values.

On the PREFILL step (seq len > 1) we call the ORIGINAL forward (PyTorch does the
prompt, and we seed the kernel kv cache from it — same as today).

### Risk / unknowns to verify
- How to advance the PyTorch cache length by 1 cheaply. DynamicCache appends on
  each layer's update(); with no layers run, length won't advance. Options:
  a) call cache.update() once with a dummy 1-token k/v of the right shape, OR
  b) keep a tiny "length shim" — but HF reads get_seq_length() from the cache.
  Need to confirm what code_predictor / generate read between steps. (It reads
  past_hidden, which WE supply, and cache_position which HF derives from cache
  length.) -> verify with a 1-step probe before trusting it.
- Must keep numerics identical to the verified post-hook path (kernel _hidden +
  norm). We already proved that path is audibly correct.

### Fallback if cache-advance is fragile
Keep the post-hook (correct, just slow) for the demo, and document the
double-compute as a known overhead with the measured 1 ms vs 42 ms numbers. The
kernel's correctness + speed are already proven offline; the server demo can run
either path. This is honest and low-risk.

## Expected payoff
- Saves ~41 ms/step. Step 227 ms -> ~186 ms. RTF ~3.0 -> ~2.4.
- Does NOT reach realtime (code_predictor ~173 ms/step dominates and is out of
  the kernel's scope). Document this clearly.
