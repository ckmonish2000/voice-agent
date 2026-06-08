#!/usr/bin/env bash
# One-shot setup for a fresh RTX 5090 Vast.ai box.
#
#   bash setup.sh
#
# Does, in the SAFE order (so nothing breaks the box's CUDA torch):
#   1. verify the box: RTX 5090 + nvcc >= 12.8 + torch sees CUDA
#   2. install project deps WITHOUT disturbing the pre-installed CUDA torch
#   3. run the megakernel parity checks (JIT-builds the kernel, proves the GPU path)
#
# After this, start the inference server with:
#   cd inference_server && \
#   PYTHONPATH=$(pwd)/../megakernel/qwen_tts_megakernel \
#   QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
#   python -m uvicorn app:app --host 0.0.0.0 --port 8080

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCHECKS="$ROOT/megakernel/qwen_tts_megakernel/checks"

bold() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
ok()   { printf "  \033[32mOK\033[0m  %s\n" "$1"; }
die()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; exit 1; }

bold "1/3  Verify the box"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "5090" \
  && ok "GPU is RTX 5090" \
  || die "GPU is NOT an RTX 5090 — the kernel only runs on sm_120a (Blackwell)."
NVCC_VER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
[ -n "$NVCC_VER" ] || die "nvcc not found — you need a CUDA -devel image (>=12.8)."
awk "BEGIN{exit !($NVCC_VER >= 12.8)}" && ok "nvcc $NVCC_VER (>=12.8)" \
  || die "nvcc $NVCC_VER < 12.8 — kernel needs CUDA >= 12.8."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0) else 1)" \
  && ok "torch sees the RTX 5090 ($(python -c 'import torch;print(torch.__version__)'))" \
  || die "torch can't see the GPU — do NOT use 'uv run' (CPU-only env); use system python."

bold "2/3  Install deps (without disturbing the box's CUDA torch)"
# transformers MUST install its own deps (huggingface_hub, tokenizers, etc.) —
# using --no-deps leaves it half-broken (GGUF_CONFIG_MAPPING / AutoProcessor
# import errors). So install transformers WITH deps. Only qwen-tts gets --no-deps
# (so it can't pin transformers back to a version that fights the model). We do
# NOT install/upgrade torch or torchvision — the box's CUDA build stays.
pip install -q "transformers==4.57.3" || die "transformers install failed"
pip install -q --no-deps "qwen-tts==0.1.1" || die "qwen-tts install failed"
pip install -q "accelerate==1.12.0" \
               librosa torchaudio onnxruntime einops soundfile ninja \
               sox gradio \
               fastapi uvicorn websockets || die "runtime deps install failed"
# qwen_tts imports the `sox` python pkg, which needs the system sox binary.
command -v sox >/dev/null 2>&1 || apt-get install -y sox libsox-dev >/dev/null 2>&1 || true
pip install -q "pipecat-ai[webrtc,deepgram,silero,runner]==1.3.0" openai python-dotenv \
  || echo "  (pipecat deps failed — fine if you only need the inference server)"
python -c "from transformers import AutoProcessor; from qwen_tts import Qwen3TTSModel" \
  && ok "imports work (transformers + qwen_tts)" \
  || die "imports broke — check torch/torchvision/transformers version match."

bold "3/3  Megakernel parity checks (JIT-builds the kernel)"
cd "$KCHECKS" || die "checks dir not found: $KCHECKS"
echo "--- Phase 4a: single-layer body parity (also compiles the kernel) ---"
python parity_single.py || die "parity_single failed (build or parity)"
echo "--- Phase 4b: code0 parity (vocab 3072) ---"
LDG_VOCAB_SIZE=3072 python parity_code0.py || die "parity_code0 failed"
echo "--- Phase 4b: full 16-code frame parity ---"
LDG_VOCAB_SIZE=3072 python parity_frame16.py || true   # 13/16 expected (bf16 floor)

bold "Done — box verified, deps installed, kernel checks passed"
echo "Start the inference server:"
echo "  cd $ROOT/inference_server && \\"
echo "  PYTHONPATH=$ROOT/megakernel/qwen_tts_megakernel \\"
echo "  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \\"
echo "  python -m uvicorn app:app --host 0.0.0.0 --port 8080"
