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
                 sample_rate: int = 24000, **kwargs):
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

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str
                      ) -> AsyncGenerator[Frame | None, None]:
        self.logger.debug(f"{self}: TTS [{text}]") if hasattr(self, "logger") else None
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            # The Qwen model decodes ~12x slower than real time on Apple Silicon
            # (MPS): RTF ~11-13 per the project README. A real-time transport like
            # WebRTC drains its output buffer faster than the model can fill it, so
            # streaming each chunk as it arrives underruns and the voice breaks up
            # ("i a m do ing we ll"). Instead we buffer the WHOLE utterance, then
            # emit it as one frame: the output transport re-chunks it into a steady
            # 10ms cadence with no underrun, so playback is gapless. The cost is
            # latency — the listener waits for the full decode before any sound.
            # When the fast GPU kernel lands (RTF < 0.1), switch back to streaming
            # each chunk for low-latency output.
            pcm = bytearray()
            async with websockets.connect(self._uri, max_size=None) as ws:
                await ws.send(json.dumps({"text": text}))
                first = True
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if first:
                            await self.stop_ttfb_metrics()
                            first = False
                        pcm.extend(msg)
                    else:
                        evt = json.loads(msg)
                        if evt.get("event") == "done":
                            self._last_server_metrics = evt.get("metrics")
                            break
                        if evt.get("event") == "error":
                            yield ErrorFrame(f"server error: {evt.get('message')}")
                            break

            if pcm:
                yield TTSAudioRawFrame(
                    audio=bytes(pcm),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )

            yield TTSStoppedFrame(context_id=context_id)
        except Exception as e:
            yield ErrorFrame(f"QwenWSTTSService error: {e}")
