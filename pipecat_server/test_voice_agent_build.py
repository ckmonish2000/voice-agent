"""
Network-free assembly test: build the pipeline with a fake transport and fake
API keys, and assert it produced a runnable PipelineTask with processors in the
expected order. No WebRTC, no Deepgram, no OpenAI calls happen here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


class _FakeTransport:
    def input(self):
        from pipecat.processors.frame_processor import FrameProcessor
        return FrameProcessor(name="fake-input")

    def output(self):
        from pipecat.processors.frame_processor import FrameProcessor
        return FrameProcessor(name="fake-output")


def test_build_pipeline_task_assembles(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-dg")
    monkeypatch.setenv("OPENAI_API_KEY", "test-oai")
    monkeypatch.setenv("QWEN_TTS_URI", "ws://localhost:8000/tts")

    import voice_agent

    task = voice_agent.build_pipeline_task(_FakeTransport())

    from pipecat.pipeline.task import PipelineTask
    assert isinstance(task, PipelineTask)

    names = [type(p).__name__ for p in voice_agent._LAST_PROCESSORS]
    assert "DeepgramSTTService" in names
    assert "OpenAILLMService" in names
    assert "QwenWSTTSService" in names
    assert names.index("DeepgramSTTService") < names.index("OpenAILLMService")
    assert names.index("OpenAILLMService") < names.index("QwenWSTTSService")
