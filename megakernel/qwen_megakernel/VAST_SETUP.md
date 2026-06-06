# Vast.ai Setup Runbook — Step 0 (rent box, verify, GO/NO-GO)

**Goal of this doc:** rent an RTX 5090 on Vast.ai, prove the unchanged megakernel
compiles and runs, and clear the **GO/NO-GO gate** (confirm the TTS talker's
architecture matches the kernel's hardcoded dims) — all **before** writing any
port code. If the gate fails, the port is a different, larger job and we replan.

You run every command on the rented box and paste the output back. I cannot
access the GPU.

---

## Why this order

The megakernel is hardcoded for the RTX 5090 (`-arch=sm_120a`, Blackwell) and
needs CUDA ≥ 12.8. None of it runs on the Mac. So Step 0 is purely: get a correct
box, confirm the tools, compile the kernel **unchanged**, and read the model
config. We change nothing until all of that passes.

---

## 1. Rent the instance

On https://vast.ai:

- **GPU:** RTX 5090 (exactly — the kernel won't run on other GPUs).
- **Image:** a PyTorch CUDA **`-devel`** image with **CUDA 12.8 or newer**.
  A devel image is required because it ships `nvcc` (the CUDA compiler); runtime
  images do not, and the kernel is JIT-compiled on first run.
  Good choice: `pytorch/pytorch:2.x.x-cuda12.8-cudnn9-devel` (or any
  CUDA ≥12.8 devel image with PyTorch).
- **Disk:** at least ~40 GB (model weights + CUDA build + pip caches).
- **Ports:** if you later run the Pipecat/inference servers here, expose the
  ports you need (8000, 7860). Not required for Step 0.

Start the instance and open its terminal (Vast web terminal or SSH).

---

## 2. First commands — verify the box

Run these and paste ALL output back:

```bash
# 2a. Confirm it's actually a 5090
nvidia-smi

# 2b. Confirm nvcc exists and is >= 12.8  (THIS is why we need a -devel image)
nvcc --version

# 2c. Confirm PyTorch sees CUDA and which CUDA it was built against
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
```

**Pass criteria:**
- `nvidia-smi` shows an **RTX 5090**.
- `nvcc --version` shows **release 12.8** or higher.
- The torch line prints `avail True` and device name contains `5090`.

If `nvcc` is missing → you rented a runtime image, not devel. Stop and re-rent a
`-devel` image. Do not try to work around it.

---

## 3. Get the megakernel onto the box and compile it UNCHANGED

We prove the build works before changing anything.

```bash
# 3a. Clone the reference kernel (the upstream repo this project vendored)
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel

# 3b. Install its deps. uv is fastest; pip also fine.
pip install uv && uv pip install --system -r requirements.txt
#   (or: pip install -r requirements.txt)

# 3c. Run the bench. First run JIT-compiles the .cu (takes a few minutes).
python -m qwen_megakernel.bench
```

**Pass criteria (paste the whole output):**
- It compiles with no nvcc errors.
- The **correctness check** prints HF tokens and MK (megakernel) tokens that
  **match** (same decoded text).
- The benchmark prints roughly **~1000 tok/s** for the megakernel (vs ~120 for
  PyTorch HF). The blog/readme reports 1036 tok/s / 8.4x.

If it compiles and the correctness check matches → **the build pipeline on your
box works.** This is the foundation; everything else is edits on top of it.

Common first-run issues:
- `nvcc fatal: Unsupported gpu architecture 'compute_120a'` → CUDA too old
  (need ≥12.8) or wrong GPU. Re-check Step 2.
- OOM during model load → pick an instance with more VRAM or free GPU memory.
- `ninja: command not found` → `pip install ninja` (it's in requirements but
  confirm).

---

## 4. GO/NO-GO gate — confirm the talker matches the kernel's dims

This is the single most important check. The kernel hardcodes these
(`qwen_megakernel/model.py`):

```
NUM_LAYERS = 28   NUM_KV_HEADS = 8   HEAD_DIM = 128
HIDDEN_SIZE = 1024   INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16 (Q_SIZE = 2048)   RMS eps = 1e-6
```

We must confirm the **Qwen3-TTS talker backbone** has the *same* numbers. Run:

```bash
python - <<'PY'
# Load the TTS model and print the TALKER backbone config.
# (Downloads ~the 0.6B TTS model on first run.)
import torch
from transformers import AutoModel, AutoConfig

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print("=== top-level config ===")
print(cfg)

# The talker backbone config is usually nested. Print likely sub-configs.
for attr in ["talker_config", "talker", "text_config", "thinker_config"]:
    sub = getattr(cfg, attr, None)
    if sub is not None:
        print(f"\n=== {attr} ===")
        print(sub)
PY
```

**Compare against the kernel's hardcoded dims. Paste the output.** Specifically
confirm for the **talker**:

| Field | Kernel expects | Talker (confirm) |
|-------|----------------|------------------|
| num_hidden_layers | 28 | ? |
| hidden_size | 1024 | ? |
| intermediate_size | 3072 | ? |
| num_attention_heads | 16 | ? |
| num_key_value_heads | 8 | ? |
| head_dim | 128 | ? |
| rms_norm_eps | 1e-6 | ? |
| rope_theta | (kernel uses 10000; talker likely 1000000) | ? |
| rope_scaling | (kernel: none; talker likely mrope) | ? |

**GO** if layers/hidden/intermediate/heads/head_dim/eps all match (rope_theta and
rope_scaling are *expected* to differ — those are the planned changes).

**NO-GO** if the core dims differ (e.g. it's actually a 1.7B talker, or hidden ≠
1024). Then stop: the kernel body is no longer reusable as-is and the port is a
much bigger rewrite. Paste the config and we replan before spending GPU hours.

---

## 5. Also dump the talker's weight key names (needed for the port)

The port has to point the kernel's weight loader at the talker's tensors. Capture
their exact names now so we don't burn GPU time guessing later:

```bash
python - <<'PY'
import torch
from transformers import AutoModel
m = AutoModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                              torch_dtype=torch.bfloat16, trust_remote_code=True)
keys = list(m.state_dict().keys())
print("total tensors:", len(keys))
print("\n=== talker layer-0 tensors (the 11 per-layer weights) ===")
for k in keys:
    if "talker" in k and (".0." in k or ".layers.0." in k):
        print(k, tuple(m.state_dict()[k].shape))
print("\n=== codec_head / output projection candidates ===")
for k in keys:
    if any(s in k.lower() for s in ["codec_head", "lm_head", "code_predictor", "embed"]):
        print(k, tuple(m.state_dict()[k].shape))
PY
```

Paste this. It tells us (a) the exact talker key prefix to load, (b) that
`codec_head` is separate from `embed_tokens` (untied — see KERNEL_CHANGES.md #2),
and (c) the `code_predictor` shapes for the PyTorch-side 15-code loop.

---

## What "done with Step 0" means

You can paste me:
1. `nvidia-smi` → RTX 5090 ✓
2. `nvcc --version` → ≥ 12.8 ✓
3. Unchanged megakernel bench → compiles + correctness match + ~1000 tok/s ✓
4. TTS talker config → core dims match the table (GO) ✓
5. Talker weight key names + codec_head/code_predictor shapes ✓

When those are in, we have a verified box and a confirmed GO. **Then** I write the
port changes (θ, talker weights, vocab → Phase 4a hidden-state parity), and you
test each increment on this same box.

---

## Cost note

Rent, do Step 0, and if you're not continuing immediately, **stop/destroy the
instance** so you're not billed idle. The model + kernel build are quick to
re-create from this runbook next session. Keep the instance only while actively
testing.
