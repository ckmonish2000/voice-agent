"""
bench_client.py — client-side benchmark for the running inference server.

Hits the /tts WebSocket, measures the numbers the brief asks for from the CLIENT
side (so they include network/tunnel), and prints a clean report + the server's
own metrics for comparison:

  - TTFC      : request sent -> first PCM byte received (client-observed)
  - RTF       : wall time to receive all audio / audio duration
  - tok/s     : audio codes per second of decode (from server metrics)
  - frame-by-frame proof: inter-chunk arrival gaps (NOT one big buffered frame)
  - per-utterance + averaged over N runs

Run (on the box, or on your Mac through an SSH -L 8000 tunnel):
  python bench_client.py                       # default URI + sentences
  python bench_client.py ws://localhost:8000/tts "Custom sentence."
  python bench_client.py --save out.wav "Hello there."   # also save audio
"""

import asyncio
import json
import sys
import time
import wave

import websockets

SAMPLE_RATE = 24000
DEFAULT_URI = "ws://localhost:8000/tts"
DEFAULT_SENTENCES = [
    "Hello, this is the megakernel speaking.",
    "The quick brown fox jumps over the lazy dog.",
    "Real time speech synthesis on a single GPU.",
]


async def one_run(uri, text, save_path=None):
    chunks = []                  # (arrival_perf, nbytes)
    pcm = bytearray()
    t_req = time.perf_counter()
    first_chunk_t = None
    server_metrics = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": text}))
        while True:
            msg = await ws.recv()
            now = time.perf_counter()
            if isinstance(msg, bytes):
                if first_chunk_t is None:
                    first_chunk_t = now
                chunks.append((now, len(msg)))
                pcm.extend(msg)
            else:
                evt = json.loads(msg)
                if evt.get("event") == "done":
                    server_metrics = evt.get("metrics")
                    break
                if evt.get("event") == "error":
                    print(f"  server error: {evt.get('message')}")
                    return None
    t_end = time.perf_counter()

    audio_s = (len(pcm) // 2) / SAMPLE_RATE
    ttfc_ms = (first_chunk_t - t_req) * 1000 if first_chunk_t else None
    total_s = t_end - t_req
    rtf = total_s / audio_s if audio_s else float("inf")

    # frame-by-frame proof: arrival gaps between chunks
    gaps = [chunks[i + 1][0] - chunks[i][0] for i in range(len(chunks) - 1)]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0

    if save_path:
        with wave.open(save_path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(pcm))

    return {
        "text": text,
        "ttfc_ms": ttfc_ms,
        "rtf": rtf,
        "audio_s": audio_s,
        "total_s": total_s,
        "chunks": len(chunks),
        "gap_mean_s": gap_mean,
        "server": server_metrics,
    }


def fmt(r):
    s = r["server"] or {}
    return (
        f"  text         : {r['text'][:50]!r}\n"
        f"  TTFC         : {r['ttfc_ms']:.1f} ms  (client-observed; target <60ms)\n"
        f"  RTF          : {r['rtf']:.3f}      (wall/audio; <1 = faster than realtime, target <0.15)\n"
        f"  audio length : {r['audio_s']:.2f} s   in {r['total_s']:.2f} s wall\n"
        f"  chunks       : {r['chunks']}  (mean gap {r['gap_mean_s']*1000:.0f} ms "
        f"-> {'STREAMING' if r['chunks'] > 1 else 'SINGLE FRAME (buffered!)'})\n"
        f"  server tok/s : {s.get('tokens_per_sec', '?')}   server RTF: {s.get('rtf','?')}"
    )


async def main():
    args = [a for a in sys.argv[1:]]
    save = None
    if "--save" in args:
        i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
    uri = DEFAULT_URI
    if args and args[0].startswith("ws"):
        uri = args.pop(0)
    sentences = args if args else DEFAULT_SENTENCES

    print(f"[bench] server: {uri}\n")
    results = []
    for i, text in enumerate(sentences):
        r = await one_run(uri, text, save_path=(save if i == 0 else None))
        if r:
            print(fmt(r)); print()
            results.append(r)

    if len(results) > 1:
        n = len(results)
        print("=" * 56)
        print(f"  AVERAGE over {n} runs")
        print(f"  TTFC : {sum(x['ttfc_ms'] for x in results)/n:.1f} ms")
        print(f"  RTF  : {sum(x['rtf'] for x in results)/n:.3f}")
        print("=" * 56)
    if save:
        print(f"\n[bench] saved first utterance -> {save}")


if __name__ == "__main__":
    asyncio.run(main())
