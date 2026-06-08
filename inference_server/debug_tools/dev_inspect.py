"""
dev_inspect.py — dump the talker source to FILES so we can read them fully
(not scroll the terminal). We need to see:
  - the TOP of talker.forward (how self.model backbone is called, what inputs)
  - where/how code_predictor runs (the ~173ms/step part)
  - talker.model.forward (the 42ms backbone we want to replace with the kernel)

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python dev_inspect.py

Then just `cat` / open the three files it writes (paths printed at the end), or
paste them back here.
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
talker = eng.model.talker
tm = talker.model
cp = talker.code_predictor

OUT = os.path.dirname(__file__)


def dump(name, obj_or_func):
    path = os.path.join(OUT, name)
    try:
        src = inspect.getsource(obj_or_func)
    except Exception as e:
        src = f"(could not get source: {e!r})"
    with open(path, "w") as f:
        f.write(src)
    print(f"[wrote] {path}  ({len(src.splitlines())} lines)", flush=True)


print("talker class           :", type(talker).__name__, flush=True)
print("talker.model class     :", type(tm).__name__, flush=True)
print("code_predictor class   :", type(cp).__name__, flush=True)

# 1. the outer talker.forward (we saw the bottom; need the top + code_predictor)
dump("_src_talker_forward.py", type(talker).forward)
# 2. the backbone forward (42ms) we want to replace
dump("_src_talkermodel_forward.py", type(tm).forward)
# 3. the code_predictor — find its generate/forward (the 173ms/step part)
for meth in ("forward", "generate", "sample", "decode"):
    if hasattr(type(cp), meth):
        dump(f"_src_codepredictor_{meth}.py", getattr(type(cp), meth))

# also: does talker have a separate method that runs the code_predictor loop?
print("\ntalker methods that mention 'predict' or 'code':", flush=True)
for n in dir(talker):
    if "predict" in n.lower() or "code" in n.lower():
        print("   ", n, flush=True)

print("\n[done] Open the _src_*.py files (or paste them back).", flush=True)
