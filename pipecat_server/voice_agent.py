"""
Voice agent pipeline: mic audio -> Deepgram STT -> OpenAI LLM -> Qwen TTS -> speaker.

This is the Pipecat server (docs/2026-06-05-voice-agent-frontend-design.md):
the orchestrator the browser talks to. It owns the Deepgram and OpenAI keys and
calls the inference server via QwenWSTTSService.

Two entrypoints:
  build_pipeline_task(transport) -> PipelineTask   (used by tests + by bot())
  bot(runner_args)               -> async          (used by the runner / server.py)
"""

import os
import sys

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

sys.path.insert(0, os.path.dirname(__file__))
from qwen_ws_tts import QwenWSTTSService  # noqa: E402

SR = 24000

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are spoken aloud by a "
    "text-to-speech model that synthesizes slower than real time, so BREVITY is "
    "critical: answer in ONE short sentence, ideally under 15 words. Never use "
    "markdown, lists, or emoji. Get to the point immediately; no filler like "
    "'Sure!' or 'Great question'."
)

# Populated by build_pipeline_task so tests can inspect processor order.
_LAST_PROCESSORS: list = []


def build_pipeline_task(transport) -> PipelineTask:
    """Assemble the STT->LLM->TTS pipeline around a transport. No network here."""
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    )
    tts = QwenWSTTSService(
        uri=os.environ.get("QWEN_TTS_URI", "ws://localhost:8000/tts"),
        sample_rate=SR,
        # Jitter-buffer pre-roll (seconds) before playback starts. Tune via env:
        # smaller = lower latency but riskier underrun; larger = smoother but more
        # initial delay. Default 2.5s suits short replies on the slow MPS model.
        preroll_secs=float(os.environ.get("QWEN_TTS_PREROLL", "2.5")),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor()

    processors = [
        transport.input(),
        rtvi,
        stt,
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ]

    global _LAST_PROCESSORS
    _LAST_PROCESSORS = processors

    pipeline = Pipeline(processors)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    return task


def _webrtc_params():
    """Transport parameters for the browser connection: audio in + audio out.

    Note: Pipecat 1.3.0's TransportParams has no `vad_analyzer` field — VAD /
    turn-taking is handled by the universal LLM aggregator's default turn
    strategy, not configured here. Passing a vad_analyzer to TransportParams is
    silently ignored, so we don't. This matches the runner's own webrtc factory
    (pipecat/runner/utils.py), which sets only the audio in/out flags.
    """
    from pipecat.transports.base_transport import TransportParams

    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=SR,
        audio_out_channels=1,
    )


async def bot(runner_args):
    """Runner entrypoint. The runner hands us a SmallWebRTC connection; we build a
    transport from it, assemble the pipeline, and run until the browser leaves."""
    from pipecat.runner.utils import create_transport

    transport = await create_transport(
        runner_args,
        {"webrtc": _webrtc_params},
    )

    task = build_pipeline_task(transport)

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        # Greet on connect so the user immediately hears the loop works.
        from pipecat.frames.frames import LLMRunFrame
        await task.queue_frames([LLMRunFrame()])

    from pipecat.pipeline.runner import PipelineRunner

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
