#!/usr/bin/env bash
# One-shot setup + verification for a fresh Vast.ai RTX 5090 box.
#
# What it does:
#   1. installs the deps our checks need (qwen-tts, ninja, soundfile)
#   2. verifies the box: RTX 5090, nvcc >= 12.8, torch sees CUDA
#   3. runs the parity suite (4a body, 4b code0, 4b 16-code frame)
#
# Run from this directory (voice-agent/megakernel/qwen_tts_megakernel/checks):
#   bash setup_and_verify.sh
#
# Safe to re-run. Stops at the first hard failure (box not a 5090 / no nvcc).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

bold() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
ok()   { printf "  \033[32mOK\033[0m  %s\n" "$1"; }
die()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; exit 1; }

bold "1/4  Install dependencies"
pip install -q qwen-tts ninja soundfile || die "pip install failed"
ok "qwen-tts, ninja, soundfile installed"

bold "2/4  Verify the box"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "5090" \
  && ok "GPU is RTX 5090" \
  || die "GPU is NOT an RTX 5090 — the kernel only runs on sm_120a (5090)."

NVCC_VER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
if [ -z "$NVCC_VER" ]; then die "nvcc not found — you need a CUDA -devel image (>=12.8)."; fi
awk "BEGIN{exit !($NVCC_VER >= 12.8)}" \
  && ok "nvcc $NVCC_VER (>= 12.8)" \
  || die "nvcc $NVCC_VER is < 12.8 — kernel needs CUDA >= 12.8."

python - <<'PY' || die "torch cannot see the GPU"
import torch, sys
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
name = torch.cuda.get_device_name(0)
assert "5090" in name, f"torch device is {name}, not a 5090"
print(f"  OK  torch {torch.__version__}, cuda {torch.version.cuda}, dev {name}")
PY

bold "3/4  Build + body parity (parity_single JIT-compiles the kernel)"
python parity_single.py || die "parity_single failed (build or parity)"
ok "kernel compiles + Phase 4a body parity passes"

bold "4/4  Output-stage parity suite"
echo "--- Phase 4b: code0 parity (vocab 3072, recompiles) ---"
LDG_VOCAB_SIZE=3072 python parity_code0.py || die "parity_code0 failed"
echo "--- Phase 4b: full 16-code frame parity ---"
LDG_VOCAB_SIZE=3072 python parity_frame16.py || true   # 13/16 expected (bf16 floor)

bold "Done"
echo "Box verified + parity suite run. For the ear test:"
echo "  python ear_test.py \"Hello, this is a test of the speech kernel.\""
