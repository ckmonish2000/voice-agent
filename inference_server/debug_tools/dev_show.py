"""
dev_show.py — write the talker source + where code_predictor runs to a FILE
(inference_server/_talker_dump.txt) so it can be copied/pasted easily.

We need to know whether the ~173ms/step code_predictor runs inside talker.forward
or in a separate loop — that decides our optimization.

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python dev_show.py

Then copy the file it writes:  cat _talker_dump.txt
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
talker = eng.model.talker
cp = talker.code_predictor

OUT_PATH = os.path.join(os.path.dirname(__file__), "_talker_dump.txt")
lines = []


def w(s=""):
    lines.append(s)


def show(title, func, head=None):
    w("\n" + "=" * 70)
    w(title)
    w("=" * 70)
    try:
        src = inspect.getsource(func).splitlines()
    except Exception as e:
        w(f"(no source: {e!r})")
        return
    if head:
        src = src[:head]
    for i, line in enumerate(src):
        w(f"{i:3d}| {line}")


# 1. top of talker.forward (the cut-off part)
show("talker.forward — FIRST 50 lines", type(talker).forward, head=50)

# 2. where code_predictor / codec_ids appear in the talker class
w("\n" + "=" * 70)
w("Lines in talker class mentioning code_predictor / codec_ids / predict:")
w("=" * 70)
talker_src = inspect.getsource(type(talker))
for i, line in enumerate(talker_src.splitlines()):
    if ("code_predictor" in line or "codec_ids" in line
            or "predict" in line.lower()):
        w(f"{i:4d}| {line}")

# 3. which talker method calls code_predictor, and how many times
w("\n" + "=" * 70)
w("Talker methods that call self.code_predictor(:")
w("=" * 70)
for name in dir(type(talker)):
    attr = getattr(type(talker), name, None)
    if callable(attr):
        try:
            s = inspect.getsource(attr)
            if "self.code_predictor(" in s:
                w(f"  {name}  (call sites: {s.count('self.code_predictor(')})")
        except Exception:
            pass

# 4. the code_predictor's own forward (the 173ms/step part)
show("code_predictor.forward", type(cp).forward, head=80)

text = "\n".join(lines)
with open(OUT_PATH, "w") as f:
    f.write(text)

print(text)
print(f"\n[wrote] {OUT_PATH}")
print("Copy it with:  cat _talker_dump.txt")
