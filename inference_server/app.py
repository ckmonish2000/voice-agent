"""
FastAPI WebSocket inference server: text in -> streaming PCM out.

This is the "inference server" of the brief (Step 2). It loads the streaming
engine once and exposes one realtime endpoint. The talker decode behind it is
the megakernel seam; swapping the kernel in later does not change this file.

Protocol (WS /tts)
------------------
  client -> server : {"text": "hello world"}            (JSON text message)
  server -> client : <binary>  24 kHz mono int16 PCM    (one per codec hop)
                     ... repeated ...
  server -> client : {"event": "done", "metrics": {...}} (JSON text message)

Run:
  .venv/bin/python -m uvicorn inference_server.app:app --port 8000
  (or: .venv/bin/python inference_server/app.py)
"""

import os
import sys
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.insert(0, os.path.dirname(__file__))
from engine import StreamingTTSEngine, StreamConfig, SAMPLE_RATE  # noqa: E402
from metrics import summarize  # noqa: E402

app = FastAPI(title="Qwen3-TTS streaming inference server")

# Single shared engine; one in-flight request at a time (one model on MPS).
_engine: StreamingTTSEngine | None = None


@app.on_event("startup")
def _load():
    global _engine
    print("[server] loading streaming engine ...")
    _engine = StreamingTTSEngine()
    print("[server] warming up ...")
    _engine.warmup()
    print("[server] ready.")


@app.get("/health")
def health():
    return {"status": "ok" if _engine is not None else "loading",
            "sample_rate": SAMPLE_RATE}


@app.websocket("/tts")
async def tts(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)
            text = req.get("text", "").strip()
            if not text:
                await ws.send_text(json.dumps({"event": "error",
                                               "message": "empty text"}))
                continue

            cfg = StreamConfig(
                max_new_tokens=int(req.get("max_new_tokens", 512)),
                window=int(req.get("window", StreamConfig.window)),
                hop=int(req.get("hop", StreamConfig.hop)),
                do_sample=bool(req.get("do_sample", True)),
            )

            t_request = time.perf_counter()
            first_chunk_t = None
            total_bytes = 0

            # decode_stream is a blocking generator (model runs on MPS). Run it
            # in a worker thread and forward chunks as they arrive so the event
            # loop stays responsive.
            import asyncio
            loop = asyncio.get_event_loop()
            chunk_q: asyncio.Queue = asyncio.Queue()
            _DONE = object()

            def _produce():
                try:
                    for pcm in _engine.decode_stream(text, cfg):
                        loop.call_soon_threadsafe(chunk_q.put_nowait, pcm)
                finally:
                    loop.call_soon_threadsafe(chunk_q.put_nowait, _DONE)

            await loop.run_in_executor(None, lambda: None)  # ensure loop ready
            producer = loop.run_in_executor(None, _produce)

            while True:
                pcm = await chunk_q.get()
                if pcm is _DONE:
                    break
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                total_bytes += len(pcm)
                await ws.send_bytes(pcm)

            await producer  # surface any exception from the worker

            m = _engine.last_metrics
            report = summarize(
                metrics=m,
                request_t=t_request,
                first_chunk_t=first_chunk_t,
                total_pcm_bytes=total_bytes,
                sample_rate=SAMPLE_RATE,
            )
            await ws.send_text(json.dumps({"event": "done", "metrics": report}))
            print(f"[server] '{text[:40]}' -> {report}")
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
