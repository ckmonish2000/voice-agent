# Performance results — streaming TTS server (RTX 5090, measured 2026-06-09)

All numbers measured ON the box (RTX 5090), warmed, on a ~6 s utterance
("The quick brown fox jumps over the lazy dog, and then it runs back home
before the rain begins to fall."), via inference_server/debug_tools/bench_engine.py.

## Kernel vs PyTorch backbone (end-to-end, in the streaming server)

| Metric            | PyTorch backbone | Kernel backbone | Improvement |
|-------------------|------------------|-----------------|-------------|
| TTFC              | 1031 ms          | 837 ms          | ~19% faster |
| RTF overall       | 2.963            | 2.441           | ~18% faster |
| RTF steady        | 2.949            | 2.431           | ~18% faster |
| total wall (~6 s) | 17.54 s          | 14.84 s         | 2.7 s saved |
| chunk gap (mean)  | 917 ms           | 778 ms          | ~139 ms     |
| streaming         | yes (19 chunks)  | yes (19 chunks) | —           |

The kernel makes the whole pipeline ~18% faster end-to-end. It does NOT reach
the RTF<0.15 / TTFC<60ms targets — see the per-step breakdown for why.

### After the first-word optimization (first_hop=1)

Emitting the first codec chunk after 1 frame instead of 4 (StreamConfig.first_hop)
cut TTFC sharply with no change to steady-state speed:

| Metric           | kernel (hop=4) | kernel (first_hop=1) |
|------------------|----------------|----------------------|
| TTFC             | 837 ms         | 312 ms               |
| first chunk gap  | ~745 ms        | 220 ms               |
| RTF overall      | 2.441          | 2.377                |
| RTF steady       | 2.431          | 2.353                |

TTFC is the latency a listener feels first; it dropped ~2.7x. RTF is unchanged
(steady-state is gated by the code_predictor, not by when the first chunk emits).

## Per-step breakdown (kernel path, measured via debug_tools/diag_step.py / debug_tools/diag_double.py)

| Component                       | per step  | share |
|---------------------------------|-----------|-------|
| code_predictor (codes 1-15, 15x)| ~173 ms   | ~76%  |  <- bottleneck, plain PyTorch, OUT of kernel scope
| backbone — PyTorch (BEFORE fix) | 42.5 ms   | —     |  <- was computed then discarded
| backbone — kernel (_dfh)        | 1.08 ms   | <1%   |  <- the megakernel; matches offline 0.78 ms
| codec decode (every 4th step)   | ~12 ms avg| ~6%   |
| whole step (before fix)         | ~227 ms   |       |
| whole step (after fix)          | ~186 ms   |       |

Backbone offline benchmark: 1286 steps/s (0.777 ms/step), ~103x realtime.
So the kernel does its job; the wall is the code_predictor.

## Bugs found and fixed (debugging rigor)

1. **Codec (speech_tokenizer) was running on the CPU.** model.to(device) does not
   move the Qwen3TTSTokenizer wrapper (its net is in .model). Result: codec
   decode ~13 s/call, GPU ~0 ms (proven via torch.profiler: wall 13.7 s, CUDA
   ~0 ms, 4219 ops dominated by aten::copy_ / Memcpy DtoH). This presented as a
   "streaming hang" (each decode slower than the client timeout).
   Fix: move speech_tokenizer.model to cuda in common.load_model().
   Result: codec ~13000 ms -> ~28 ms (~450x).

2. **Backbone computed twice per step.** The kernel ran as a forward HOOK, which
   fires AFTER PyTorch's 28 layers already ran (~42 ms), then overwrote the
   result. Fix: replace talker.model.forward so the kernel runs INSTEAD of the
   PyTorch backbone on decode steps (prefill still PyTorch; PyTorch KV-cache
   length advanced with a dummy token/layer so HF position bookkeeping stays
   correct; kernel uses its own KV cache). Result: ~41 ms/step saved -> 18%
   end-to-end. Verified by debug_tools/parity_kernel_replace.py (first 9 frames bit-identical
   to PyTorch greedy, then inaudible bf16 drift — same signature as the offline
   13/16 parity + ear test).

## Honest conclusion

The megakernel is correct (offline parity + ear test + in-server greedy parity)
and fast (1 ms/step backbone). Integrating it into the streaming server gives a
measured ~18% end-to-end speedup. Real-time targets are not met because the
code_predictor (5-layer transformer x15 per frame, plain PyTorch) is ~76% of each
step and is outside the backbone megakernel's scope. The clear next optimization
to approach real-time would be a megakernel for the code_predictor, not the
backbone.
