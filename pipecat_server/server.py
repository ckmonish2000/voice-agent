"""
Process entrypoint for the Pipecat voice agent, port 7860.

Loads secrets from the repo-root .env, then hands control to the Pipecat runner,
which serves the SmallWebRTC signaling endpoint and a prebuilt test UI at /client.
The runner discovers the `bot` coroutine imported below.

Run (the inference server must already be up on :8000):
  .venv/bin/python pipecat_server/server.py -t webrtc
Then open http://localhost:7860/client in a browser.
"""

import os
import sys

from dotenv import load_dotenv

# Load the repo-root .env regardless of cwd (pipecat_server/../.env).
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_ENV)

# Fail fast with a clear message if keys are missing.
for _k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(_k):
        sys.exit(f"[server] missing {_k}. Copy .env.example to "
                 f".env in the repo root and fill it in.")

sys.path.insert(0, os.path.dirname(__file__))
from voice_agent import bot  # noqa: F401,E402  (runner discovers `bot`)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
