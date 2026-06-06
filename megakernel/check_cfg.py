import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

# Register the custom arch so Auto* knows what "qwen3_tts" is.
# (qwen3_tts is NOT in stock transformers' auto-mapping; the qwen-tts
#  package ships the class, but we must register it manually before load.)
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

print("=== top-level config ===")
print(cfg)

for attr in ["talker_config", "talker", "text_config", "thinker_config", "code_predictor_config"]:
    sub = getattr(cfg, attr, None)
    if sub is not None:
        print("\n=== " + attr + " ===")
        print(sub)
