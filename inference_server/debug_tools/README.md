# inference_server/debug_tools

Diagnostic, profiling, and benchmark scripts used to build and debug the kernel
integration. **None of these are needed to run the server** — they live here so
the server folder stays clean and so the tools are available later for debugging.

Run them from the repo with the same env the server uses, e.g.:

```bash
cd inference_server
PYTHONPATH=../megakernel/qwen_tts_megakernel \
QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
python debug_tools/<script>.py
```

(They add `inference_server/` to `sys.path` so `from engine import ...` works.)

## Benchmarks (measure performance)

| Script              | What it measures |
|---------------------|------------------|
| `bench_engine.py`   | Steady-state TTFC / RTF / chunk gaps of the engine. Run with USE_KERNEL=1 and =0 to compare kernel vs PyTorch. **The main perf benchmark.** |
| `bench_client.py`   | Client-side benchmark over the WebSocket (TTFC/RTF incl. network). Needs a running server: `python debug_tools/bench_client.py ws://localhost:8000/tts --save out.wav`. |

## Correctness checks

| Script                     | What it verifies |
|----------------------------|------------------|
| `parity_kernel_replace.py` | Kernel-replaces-backbone path matches PyTorch greedy frame-by-frame (first ~9 frames bit-identical, then inaudible bf16 drift). |

## Profiling / root-cause diagnostics (the debugging trail)

These were written to find the two big bugs (codec-on-CPU; backbone double-compute).
Kept as a record and for future use.

| Script                   | What it isolates |
|--------------------------|------------------|
| `diag_hang.py`           | Whole streaming path: TTFC + chunk count + heartbeat (is it streaming or stuck?). |
| `diag_thread.py`         | Generation on main thread vs worker thread (rules out threading deadlock). |
| `diag_codec.py`          | Codec decode in isolation (does it hang, or just slow?). |
| `diag_codec2.py`         | Codec with vs without the kernel having run (kernel/codec coexistence). |
| `diag_codec_speed.py`    | Codec time per call vs input size (warm-up vs per-call cost). |
| `diag_codec_profile.py`  | Per-stage breakdown inside the codec (transformer vs conv decoder). |
| `diag_codec_profiler.py` | torch.profiler: CPU time vs CUDA time — proved the codec ran on CPU. |
| `diag_codec_dtype.py`    | Codec dtype (bf16/fp16/fp32) timing. |
| `diag_step.py`           | Per-step time split: backbone vs codec vs the rest (code_predictor). |
| `diag_double.py`         | Device check (all on cuda?) + the backbone double-compute proof (PyTorch 42ms vs kernel 1ms). |

## Dev inspection helpers (read model internals)

| Script               | What it prints |
|----------------------|----------------|
| `dev_check.py`       | Device (cpu/cuda) of talker / backbone / code_predictor / codec. |
| `dev_inspect.py`     | Dumps talker source to files. |
| `dev_show.py`        | Where code_predictor runs + talker.forward source -> `_talker_dump.txt`. |
| `dev_cache_probe.py` | DynamicCache structure / get_seq_length / update (for the cache-advance fix). |

## The story these tools told

1. Streaming "hang" was the **codec running on CPU** (~13s/decode, GPU idle) —
   found with `diag_codec_profiler.py`, confirmed with `dev_check.py`. Fixed in
   `common.py` (move speech_tokenizer to GPU): ~13000ms -> ~28ms.
2. The kernel's speed was wasted by the **backbone double-compute** (PyTorch
   42ms/step then discarded) — found with `diag_double.py`. Fixed in `engine.py`
   (kernel replaces the backbone forward): ~18% end-to-end.

See `megakernel/docs/2026-06-09-performance-results.md` for the numbers.
