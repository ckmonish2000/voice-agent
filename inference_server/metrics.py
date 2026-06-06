"""
Streaming TTS metrics (Step 4 validation).

Computes the numbers the brief asks for, from a StreamMetrics object plus the
server-side request timing:

  tokens/sec  - audio code tokens produced per second of talker decode
                (frames * 16 / decode_seconds)
  TTFC        - Time To First Chunk: request -> first PCM chunk sent
                (target < 50 ms; on MPS this is a baseline, not the target)
  RTF         - Real Time Factor: decode_seconds / audio_seconds
                (target < 0.1; < 1.0 means faster than realtime)
  total_latency_s - request -> last chunk
  audio_seconds   - duration of produced audio
  frame_by_frame  - inter-chunk timing proof (not buffered-then-sent)

Targets are RTX-5090-megakernel goals; on Apple Silicon we report the honest
measured baseline and the gap, which the kernel is meant to close.
"""

from typing import Optional


# Brief's targets (for the megakernel on a 5090, not for MPS).
TARGET_TTFC_MS = 50.0
TARGET_RTF = 0.1


def summarize(
    metrics,
    request_t: float,
    first_chunk_t: Optional[float],
    total_pcm_bytes: int,
    sample_rate: int,
) -> dict:
    num_frames = metrics.num_frames
    num_codes = num_frames * 16
    audio_samples = total_pcm_bytes // 2  # int16
    audio_seconds = audio_samples / sample_rate if sample_rate else 0.0

    # Talker decode time: first frame produced -> last frame produced.
    if metrics.first_frame_t is not None and metrics.last_frame_t is not None:
        decode_seconds = max(metrics.last_frame_t - metrics.first_frame_t, 1e-9)
    else:
        decode_seconds = max((metrics.last_frame_t or request_t) - request_t, 1e-9)

    tokens_per_sec = num_codes / decode_seconds if decode_seconds else 0.0
    rtf = decode_seconds / audio_seconds if audio_seconds else float("inf")

    ttfc_ms = ((first_chunk_t - request_t) * 1000.0
               if first_chunk_t is not None else None)
    total_latency_s = ((metrics.last_frame_t or request_t) - request_t)

    # frame-by-frame proof: spread of inter-frame gaps
    ts = metrics.frame_arrival_ts
    if len(ts) >= 2:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gap_stats = {
            "count": len(ts),
            "mean_s": round(sum(gaps) / len(gaps), 4),
            "min_s": round(min(gaps), 4),
            "max_s": round(max(gaps), 4),
        }
    else:
        gap_stats = {"count": len(ts)}

    return {
        "frames": num_frames,
        "codes": num_codes,
        "audio_seconds": round(audio_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "ttfc_ms": round(ttfc_ms, 1) if ttfc_ms is not None else None,
        "ttfc_target_ms": TARGET_TTFC_MS,
        "ttfc_meets_target": (ttfc_ms is not None and ttfc_ms < TARGET_TTFC_MS),
        "rtf": round(rtf, 3),
        "rtf_target": TARGET_RTF,
        "rtf_meets_target": rtf < TARGET_RTF,
        "total_latency_s": round(total_latency_s, 3),
        "frame_by_frame": gap_stats,
    }


def format_report(report: dict) -> str:
    """Human-readable one-screen report for the harness / README."""
    lines = [
        "─" * 56,
        "  Streaming TTS metrics (Apple Silicon / MPS baseline)",
        "─" * 56,
        f"  frames produced      : {report['frames']}  ({report['codes']} codes)",
        f"  audio duration       : {report['audio_seconds']} s",
        f"  talker decode time   : {report['decode_seconds']} s",
        f"  tokens/sec           : {report['tokens_per_sec']}",
        f"  TTFC                 : {report['ttfc_ms']} ms"
        f"   (target <{report['ttfc_target_ms']} ms"
        f" → {'MET' if report['ttfc_meets_target'] else 'MISS, megakernel goal'})",
        f"  RTF                  : {report['rtf']}"
        f"   (target <{report['rtf_target']}"
        f" → {'MET' if report['rtf_meets_target'] else 'MISS, megakernel goal'})",
        f"  total latency        : {report['total_latency_s']} s",
        f"  frame-by-frame       : {report['frame_by_frame']}",
        "─" * 56,
    ]
    return "\n".join(lines)
