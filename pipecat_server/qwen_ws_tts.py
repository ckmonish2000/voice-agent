"""
QwenWSTTSService — a Pipecat TTS service backed by our WebSocket inference server.

This is the model-independent half of the seam (PLAN.md): it speaks Pipecat's
frame protocol on one side and the server's WS protocol on the other. It does
not know or care whether the talker decode behind the server is stock PyTorch or
the megakernel — that swap happens server-side and never touches this file.

run_tts(text) -> async stream of TTSAudioRawFrame
  opens (or reuses) a WS to the server, sends the text, and yields each PCM
  chunk as a TTSAudioRawFrame the moment it arrives. ttfb metrics mark TTFC.
"""

import json
from typing import AsyncGenerator

import websockets

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings


class QwenWSTTSService(TTSService):
    """Streams audio from the Qwen3-TTS WebSocket inference server."""

    def __init__(self, *, uri: str = "ws://localhost:8000/tts",
                 sample_rate: int = 24000, preroll_secs: float = 0.0, **kwargs):
        # The voice is fixed server-side (the reference-clone clip), so model/
        # voice/language have no per-request meaning here; set them to None to
        # satisfy TTSSettings validation.
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._uri = uri
        self._sample_rate = sample_rate
        # Jitter buffer pre-roll: seconds of audio to accumulate before playback.
        #   0.0  = V2 pure frame-by-frame streaming (no buffering). Honest realtime
        #          streaming, but at RTF>1 (codec slower than realtime) the browser
        #          plays faster than we generate -> audio underruns / stutters.
        #   >0   = V1 buffered: accumulate this many seconds before playback so the
        #          cushion covers the gap and audio is smooth.
        # Converted to bytes: sample_rate * 2 bytes/sample (int16) * 1 channel.
        # Runtime-mutable via set_realtime() so the UI can toggle V1<->V2 live.
        self._buffered_preroll_secs = preroll_secs if preroll_secs > 0 else 8.0
        self._preroll_bytes = int(preroll_secs * sample_rate * 2)

    def set_realtime(self, realtime: bool) -> None:
        """Toggle V2 realtime streaming (no buffer) vs V1 buffered playback.

        realtime=True  -> preroll 0  (stream each chunk as it arrives; may stutter)
        realtime=False -> preroll _buffered_preroll_secs (smooth, buffered start)
        Applies to the NEXT utterance.
        """
        if realtime:
            self._preroll_bytes = 0
        else:
            self._preroll_bytes = int(
                self._buffered_preroll_secs * self._sample_rate * 2)

    @property
    def is_realtime(self) -> bool:
        return self._preroll_bytes == 0

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str
                      ) -> AsyncGenerator[Frame | None, None]:
        self.logger.debug(f"{self}: TTS [{text}]") if hasattr(self, "logger") else None
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            # JITTER BUFFER (pre-roll cushion).
            # The Qwen model decodes ~12x slower than real time on Apple Silicon
            # (MPS): RTF ~11-13. WebRTC plays at 1x real time, so if we stream each
            # chunk the instant it arrives, the output buffer empties faster than
            # the model can refill it and the voice breaks up ("i a m do ing we ll").
            #
            # Instead we accumulate a cushion of `_preroll_bytes` (default ~2.5s of
            # audio) BEFORE emitting anything. Once the cushion is ready we release
            # it as one frame (playback starts), then stream every later chunk as it
            # arrives. While the cushion plays, the model keeps generating, so for
            # short replies (a sentence or two) playback usually finishes the cushion
            # right as the rest arrives — smooth, with less initial delay than
            # buffering the entire reply. For long replies the cushion can still run
            # dry (the model is simply too slow), so this trades some smoothness on
            # long replies for lower latency. Tune with preroll_secs / QWEN_TTS_PREROLL.
            # When the fast GPU kernel lands (RTF < 0.1), set preroll to ~0.
            buf = bytearray()
            started = False  # have we released the pre-roll cushion yet?
            async with websockets.connect(self._uri, max_size=None) as ws:
                await ws.send(json.dumps({"text": text}))
                first = True
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if first:
                            await self.stop_ttfb_metrics()
                            first = False
                        buf.extend(msg)
                        # Hold until the cushion is full; after that, stream each
                        # chunk through immediately.
                        if not started:
                            if len(buf) >= self._preroll_bytes:
                                yield TTSAudioRawFrame(
                                    audio=bytes(buf),
                                    sample_rate=self._sample_rate,
                                    num_channels=1,
                                    context_id=context_id,
                                )
                                buf.clear()
                                started = True
                        else:
                            yield TTSAudioRawFrame(
                                audio=bytes(buf),
                                sample_rate=self._sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )
                            buf.clear()
                    else:
                        evt = json.loads(msg)
                        if evt.get("event") == "done":
                            self._last_server_metrics = evt.get("metrics")
                            break
                        if evt.get("event") == "error":
                            yield ErrorFrame(f"server error: {evt.get('message')}")
                            break

            # Flush whatever is left (the reply ended before the cushion filled, or
            # a final partial chunk after streaming started).
            if buf:
                yield TTSAudioRawFrame(
                    audio=bytes(buf),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )

            yield TTSStoppedFrame(context_id=context_id)
        except Exception as e:
            yield ErrorFrame(f"QwenWSTTSService error: {e}")
