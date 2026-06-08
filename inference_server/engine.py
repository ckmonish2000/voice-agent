"""
StreamingTTSEngine — the streaming seam.

This is the piece the RTX 5090 megakernel replaces later. Everything above it
(WebSocket server, Pipecat service) is model-independent and never changes.

What it does
------------
Turns the existing batched `model.generate()` path (which returns the whole
(T,16) codes tensor at once — "buffered then sent") into a frame-by-frame
generator:

    text  ──▶  per-step (1,16) audio codes  ──▶  per-hop 24 kHz PCM tail

How streaming is achieved without reimplementing HF sampling
------------------------------------------------------------
The talker is a normal `GenerationMixin` model: its per-step `forward` emits one
frame of 16 codes (codebook 0 from `codec_head`, codes 1..15 from the 5-layer
`code_predictor`) and HF's `generate()` loop drives it. We do NOT rewrite that
loop — instead we:

  1. Run `model.generate(...)` in a background thread.
  2. Register a `forward` hook on the talker module. HF invokes talker.forward
     once per generated step; the hook reads that step's `codec_ids` (the full
     16-code frame, returned as `output.hidden_states[1]`) and pushes it onto a
     thread-safe queue in real time.
  3. The foreground thread drains the queue frame-by-frame, runs a sliding-window
     codec decode, and yields the newly revealed PCM tail.

A forward hook is used (not a LogitsProcessor) because qwen_tts's
`model.generate()` does not thread a `logits_processor` through to the talker —
but the talker is a real `nn.Module`, so a forward hook fires every step.

Because the exact same sampling path runs, the streamed codes are identical to
the batched path by construction — that's what `parity_check.py` asserts.

The megakernel seam is the talker decode step. To swap it in later, replace the
generate() call's talker backend; the queue/codec/PCM machinery is untouched.
"""

import os
import sys
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import load_model  # noqa: E402  (shared Qwen3-TTS model loader)

# Voice-clone reference (same clip the stage scripts use).
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)

# Sliding-window codec decode parameters. The 12 Hz codec uses sliding-window
# attention + causal transpose-conv upsampling, so a frame's waveform depends on
# its neighbours. We decode a rolling window and emit only the new tail.
DEFAULT_WINDOW = 16   # frames kept as decode context
DEFAULT_HOP = 4       # decode + emit every HOP new frames
UPSAMPLE = 1920       # samples per frame at 24 kHz (12.5 Hz * 1920 = 24000)
SAMPLE_RATE = 24000

_FRAME_SENTINEL = None  # pushed onto the queue to signal generation finished


@dataclass
class StreamConfig:
    max_new_tokens: int = 512
    window: int = DEFAULT_WINDOW
    hop: int = DEFAULT_HOP
    seed: Optional[int] = None  # set for deterministic parity runs
    # sampling (mirror the wrapper defaults so parity holds against generate())
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50


@dataclass
class StreamMetrics:
    """Filled in as a stream runs; read by metrics.py / the server."""
    num_frames: int = 0
    first_frame_t: Optional[float] = None   # perf_counter at first code frame
    last_frame_t: Optional[float] = None
    start_t: Optional[float] = None
    frame_arrival_ts: list = field(default_factory=list)  # per-frame perf_counter

    @property
    def num_codes(self) -> int:
        return self.num_frames * 16


class StreamingTTSEngine:
    """Loads the model once; streams PCM for each text request."""

    def __init__(self, device: Optional[str] = None):
        self.model, self.processor, self.device = load_model(device)
        self._eos = self.model.config.talker_config.codec_eos_token_id

        # Build the official wrapper once — we reuse ONLY its prompt-building and
        # kwarg-merging helpers, not its (batched) generate path.
        from qwen_tts import Qwen3TTSModel
        self._wrapper = Qwen3TTSModel(
            model=self.model,
            processor=self.processor,
            generate_defaults=self.model.generate_config,
        )
        self._voice = self._build_voice_clone()
        self._warmed = False

        # --- optional megakernel backbone (USE_KERNEL=1) -------------------
        # When enabled, the RTX 5090 megakernel runs the 28-layer talker backbone
        # each decode step in place of PyTorch's layers (code_predictor + codec
        # stay PyTorch). Verified equivalent + audibly identical — see
        # megakernel/README.md. Falls back to PyTorch if kernel init fails.
        self._use_kernel = os.environ.get("USE_KERNEL", "0") == "1"
        self._kdec = None
        if self._use_kernel:
            try:
                from qwen_tts_megakernel.model_tts import build_talker_decoder
                self._kdec, _ = build_talker_decoder(verbose=False)
                self._dfh = torch.ops.qwen_tts_megakernel_C.decode_from_hidden
                print("[engine] USE_KERNEL=1 — megakernel backbone active")
            except Exception as e:
                print(f"[engine] USE_KERNEL=1 but kernel init failed ({e}); "
                      "falling back to PyTorch backbone")
                self._use_kernel = False

    def warmup(self, text: str = "warm up") -> None:
        """Run one throwaway generate so the first real request isn't the cold
        outlier (the first-ever generate() after load primes lazy model state and
        produces slightly different output than steady-state calls)."""
        if self._warmed:
            return
        self.generate_batched_codes(text, StreamConfig(do_sample=False, max_new_tokens=8))
        self._warmed = True

    # ---- one-time voice-clone prompt build (reference voice) ----
    def _build_voice_clone(self):
        prompt_items = self._wrapper.create_voice_clone_prompt(
            ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False
        )
        vc_prompt = self._wrapper._prompt_items_to_voice_clone_prompt(prompt_items)
        ref_tok = self._wrapper._tokenize_texts(
            [self._wrapper._build_ref_text(REF_TEXT)]
        )[0]
        return {
            "vc_prompt": vc_prompt,
            "ref_tok": ref_tok,
            "ref_code": prompt_items[0].ref_code,
        }

    def _text_to_input_ids(self, text: str) -> torch.Tensor:
        # Wrap in the chat template the talker was trained on (same as stage1 /
        # the official wrapper). The generate() path slices input_id[:, :3] and
        # input_id[:, 3:-5], so this exact framing is required.
        wrapped = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        enc = self.processor(text=wrapped, return_tensors="pt", padding=True)
        ids = enc["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)

    # ---- the streaming entry point ----
    def decode_stream(
        self, text: str, cfg: Optional[StreamConfig] = None
    ) -> Iterator[bytes]:
        """Yield 24 kHz mono int16 PCM chunks as the talker generates frames.

        This is the megakernel seam: replace the talker decode inside the
        generate() call and this method's contract is unchanged.
        """
        import time

        cfg = cfg or StreamConfig()
        metrics = self.last_metrics = StreamMetrics()
        metrics.start_t = time.perf_counter()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        input_ids = self._text_to_input_ids(text)
        gen_kwargs = self._wrapper._merge_generate_kwargs(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )

        # PCM queue carries finished audio bytes (NOT raw frames). All GPU work —
        # backbone, code_predictor AND codec decode — runs on the worker thread,
        # serialized by generate()'s own loop (the codec decode happens inside the
        # frame hook, which fires synchronously on the worker thread each step).
        # The foreground does ZERO GPU work; it only drains bytes off the queue.
        #
        # Why: a single process-wide CUDA context is shared by all threads. The
        # earlier design ran generate() on a worker thread and the codec decode on
        # the foreground thread — two threads issuing GPU work concurrently, which
        # deadlocked (worker froze mid-generate, codec blocked, no PCM ever out;
        # see diag_hang.py: 13 frames produced, 0 chunks emitted, both threads
        # alive but stuck). Keeping every CUDA call on ONE thread (the way the
        # verified kernel_in_loop.py does) fixes it while preserving streaming.
        pcm_q: "queue.Queue" = queue.Queue()
        decoder = _SlidingCodecDecoder(
            self.model.speech_tokenizer,
            ref_code=self._voice["ref_code"].to(self.device),
            window=cfg.window,
            hop=cfg.hop,
            device=self.device,
        )
        self._install_frame_hook(pcm_q, metrics, decoder)
        if self._use_kernel:
            self._install_kernel_backbone_hook()

        def _run_generate():
            try:
                with torch.no_grad():
                    self.model.generate(
                        input_ids=[input_ids],
                        ref_ids=[self._voice["ref_tok"]],
                        voice_clone_prompt=self._voice["vc_prompt"],
                        languages=["English"],
                        non_streaming_mode=False,
                        **gen_kwargs,
                    )
                # generation finished: flush the codec tail (still on this thread,
                # the only thread that touches the GPU).
                tail = decoder.flush()
                if tail is not None and len(tail) > 0:
                    pcm_q.put(tail)
            except Exception as e:  # surface worker crashes instead of hanging
                import traceback
                traceback.print_exc()
                pcm_q.put(("__error__", repr(e)))
            finally:
                pcm_q.put(_FRAME_SENTINEL)

        worker = threading.Thread(target=_run_generate, daemon=True)
        worker.start()

        # Foreground: drain finished PCM bytes only — no GPU work here.
        try:
            while True:
                item = pcm_q.get()
                if item is _FRAME_SENTINEL:
                    break
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    raise RuntimeError(f"generation worker failed: {item[1]}")
                if item is not None and len(item) > 0:
                    yield item
        finally:
            metrics.last_frame_t = time.perf_counter()
            worker.join(timeout=5.0)
            self._remove_frame_hook()
            if self._use_kernel:
                self._remove_kernel_backbone_hook()

    # ---- frame hook: decode each step's frame to PCM on the worker thread ----
    def _install_frame_hook(self, pcm_q: "queue.Queue", metrics: StreamMetrics,
                            decoder: "_SlidingCodecDecoder"):
        """Decode each generated frame to PCM as the talker produces it, then
        push the finished bytes onto the queue.

        talker.forward returns hidden_states=(layer_hidden_states, codec_ids).
        On generate steps (seq len 1) codec_ids is the (1,16) frame for this
        step; on the prefill step it is None. We feed each real frame to the
        sliding-window codec decoder *here* — i.e. on the same (worker) thread
        that runs generate(), so every CUDA call is serialized on one thread.
        The EOS frame is skipped.
        """
        talker = self.model.talker
        eos = self._eos

        def _hook(_module, _inputs, output):
            import time
            hs = getattr(output, "hidden_states", None)
            if not (isinstance(hs, tuple) and len(hs) == 2):
                return output
            codec_ids = hs[1]
            if codec_ids is None:
                return output  # prefill step
            frame = codec_ids.detach().view(-1)[:16]
            if int(frame[0]) == eos:
                return output
            now = time.perf_counter()
            if metrics.first_frame_t is None:
                metrics.first_frame_t = now
            metrics.frame_arrival_ts.append(now)
            metrics.num_frames += 1
            # codec decode on THIS (worker) thread — emit PCM tail if a hop is ready
            pcm = decoder.push(frame.to(self.device))
            if pcm is not None and len(pcm) > 0:
                pcm_q.put(pcm)
            return output

        self._hook_handle = talker.register_forward_hook(_hook)

    def _remove_frame_hook(self):
        h = getattr(self, "_hook_handle", None)
        if h is not None:
            h.remove()
            self._hook_handle = None

    # ---- megakernel backbone hooks (USE_KERNEL=1) ----
    def _install_kernel_backbone_hook(self):
        """Run the megakernel 28-layer backbone each decode step in place of
        PyTorch's layers. pre-hook captures inputs_embeds; post-hook seeds the
        kernel KV cache from PyTorch's prefill (first decode step), runs
        decode_from_hidden, and overwrites last_hidden_state with the kernel's
        (pre-norm _hidden + PyTorch final norm — the faithful path). Verified
        reference: megakernel/.../checks/kernel_in_loop.py.
        """
        talker_model = self.model.talker.model
        dec = self._kdec
        st = {"seeded": False, "emb": None}

        def _layer_kv(pkv, L):
            if hasattr(pkv, "layers"):
                return pkv.layers[L].keys, pkv.layers[L].values
            return pkv.key_cache[L], pkv.value_cache[L]

        def _seed(pkv):
            kc, vc = dec._k_cache, dec._v_cache
            for L in range(kc.shape[0]):
                k, v = _layer_kv(pkv, L)
                n = min(k.shape[2], kc.shape[2])
                kc[L, :, :n, :] = k[0, :, :n, :].to(kc.dtype)
                vc[L, :, :n, :] = v[0, :, :n, :].to(vc.dtype)

        def pre_hook(_m, args, kwargs):
            emb = kwargs.get("inputs_embeds")
            if emb is None and args:
                emb = args[0]
            st["emb"] = emb

        def post_hook(_m, _a, _kw, output):
            hs = output.last_hidden_state
            if hs.shape[1] != 1:          # prefill — PyTorch handles it
                return output
            pkv = output.past_key_values
            pos = pkv.get_seq_length() - 1
            if not st["seeded"]:
                _seed(pkv)
                st["seeded"] = True
            emb = st["emb"].detach().to(torch.bfloat16).reshape(-1).contiguous()
            dec._position = pos
            self._dfh(
                dec._out_token, emb,
                dec._embed_weight, dec._layer_weights_packed,
                dec._final_norm_weight, dec._lm_head_weight,
                dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
                dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
                dec._attn_out, dec._mlp_inter, dec._norm_out,
                dec._bmax_vals, dec._bmax_idxs,
                28, pos, dec._k_cache.shape[2], dec._attn_scale,
            )
            k_pre = dec._hidden.detach().to(torch.bfloat16).view(1, 1, -1)
            output.last_hidden_state = talker_model.norm(k_pre)
            return output

        self._k_pre_handle = talker_model.register_forward_pre_hook(
            pre_hook, with_kwargs=True)
        self._k_post_handle = talker_model.register_forward_hook(
            post_hook, with_kwargs=True)

    def _remove_kernel_backbone_hook(self):
        for attr in ("_k_pre_handle", "_k_post_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)

    # ---- batched reference path (used by parity_check) ----
    def generate_batched_codes(self, text: str, cfg: Optional[StreamConfig] = None):
        """Run the original batched generate() — returns (T,16) codes on CPU."""
        cfg = cfg or StreamConfig()
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        input_ids = self._text_to_input_ids(text)
        gen_kwargs = self._wrapper._merge_generate_kwargs(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        with torch.no_grad():
            codes_list, _ = self.model.generate(
                input_ids=[input_ids],
                ref_ids=[self._voice["ref_tok"]],
                voice_clone_prompt=self._voice["vc_prompt"],
                languages=["English"],
                non_streaming_mode=False,
                **gen_kwargs,
            )
        return codes_list[0].detach().cpu()


class _SlidingCodecDecoder:
    """Rolling-window codec decode that emits only newly-revealed PCM.

    The ref-code prefix is prepended once to seed the cloned voice and warm the
    decoder's sliding-window context, but its samples are never emitted. After
    that, every HOP frames we decode the last `window` frames and emit the PCM
    that corresponds to the new hop only, so output is frame-by-frame and
    glitch-free at boundaries.
    """

    def __init__(self, speech_tokenizer, ref_code, window, hop, device):
        self._tok = speech_tokenizer
        self._ref = ref_code            # (R,16) reference voice codes
        self._window = window
        self._hop = hop
        self._device = device
        self._buf: list = []            # generated frames (each (16,))
        self._emitted_frames = 0        # how many generated frames already emitted

    def _decode(self, frames_tensor):
        """codes (N,16) -> waveform samples (CPU, 1-D)."""
        import numpy as np
        codes = torch.cat([self._ref, frames_tensor], dim=0)
        wavs, _sr = self._tok.decode([{"audio_codes": codes}])
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to("cpu").numpy()
        else:
            wav = np.asarray(wav)
        # drop the samples belonging to the ref prefix
        ref_samples = self._ref.shape[0] * UPSAMPLE
        return wav[ref_samples:]

    def push(self, frame) -> Optional[bytes]:
        self._buf.append(frame.view(-1)[:16])
        unemitted = len(self._buf) - self._emitted_frames
        if unemitted < self._hop:
            return None
        # decode the tail window for context, emit only the new hop's samples
        start = max(0, len(self._buf) - self._window)
        window_frames = torch.stack(self._buf[start:], dim=0).to(self._device)
        wav = self._decode(window_frames)
        # samples in this window that precede the new hop are context -> skip
        new_frames = len(self._buf) - self._emitted_frames
        window_len = len(self._buf) - start
        context_frames = window_len - new_frames
        skip = context_frames * UPSAMPLE
        new_pcm = wav[skip:]
        self._emitted_frames = len(self._buf)
        return _to_int16_bytes(new_pcm)

    def flush(self) -> Optional[bytes]:
        if len(self._buf) <= self._emitted_frames:
            return None
        start = max(0, len(self._buf) - self._window)
        window_frames = torch.stack(self._buf[start:], dim=0).to(self._device)
        wav = self._decode(window_frames)
        new_frames = len(self._buf) - self._emitted_frames
        window_len = len(self._buf) - start
        context_frames = window_len - new_frames
        skip = context_frames * UPSAMPLE
        self._emitted_frames = len(self._buf)
        return _to_int16_bytes(wav[skip:])


def _to_int16_bytes(wav) -> bytes:
    import numpy as np
    arr = wav.numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()
