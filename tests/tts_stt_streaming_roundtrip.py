#!/usr/bin/env python3
"""
Streaming TTS -> STT roundtrip quality + latency test.

Calls /tts/stream, measures the time between the POST arriving at the server
and the first AUDIO byte being received (i.e. after the 44-byte WAV header),
reassembles the stream into a WAV, and compares Whisper's transcription back
to the source text.

Two metrics matter:

- time_to_first_audio (TTFA): wall time from request sent to first non-header
  byte of PCM received. This is the thing we optimised for.
- similarity: difflib ratio between normalized input and Whisper output on the
  full reassembled audio. Must stay >= 0.70 (same bar as the non-streaming
  harness).

Usage:
    python3 tts_stt_streaming_roundtrip.py
    python3 tts_stt_streaming_roundtrip.py --voice Emily.wav --languages nl,en
    python3 tts_stt_streaming_roundtrip.py --ttfa-budget-ms 1000

Fails if any case misses either the similarity threshold or the TTFA budget.
"""

import argparse
import difflib
import io
import json
import re
import struct
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# (lang_code, label, sample_text, length_tag)
# length_tag is informational: "short" ≲ 3s audio, "medium" ≈ 6s, "long" ≈ 12s+.
# The rolling-vocoder refactor is validated against "long" cases because the
# old two-chunk design buffer-underruns there (chunk 2 only arrives at EOS).
TEST_CASES = [
    ("en", "English",  "Hello world. This is a streaming test of the text to speech system.", "short"),
    ("nl", "Dutch",    "Goedemorgen. Dit is een streaming test van het spraaksynthese systeem.", "short"),
    ("de", "German",   "Guten Tag. Dies ist ein Streaming-Test des Sprachsynthesesystems.", "short"),
    ("fr", "French",   "Bonjour. Ceci est un test en streaming du système de synthèse vocale.", "short"),
    ("es", "Spanish",  "Hola. Esta es una prueba en streaming del sistema de síntesis de voz.", "short"),
    ("it", "Italian",  "Ciao. Questo è un test in streaming del sistema di sintesi vocale.", "short"),
    ("en", "English-long",
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "Meanwhile, the morning fog slowly lifts to reveal a clear blue sky, "
        "and distant birds begin their usual chorus of short whistles and calls. "
        "By the time the hikers reach the summit the weather has turned warm, "
        "and they pause to share water and take in the wide green valley below.",
        "long"),
    ("nl", "Dutch-long",
        "De snelle bruine vos springt over de luie hond bij de oever van de rivier. "
        "Ondertussen trekt de ochtendmist langzaam op en onthult een heldere blauwe hemel, "
        "en in de verte beginnen vogels aan hun gebruikelijke refrein van korte fluittonen. "
        "Tegen de tijd dat de wandelaars de top bereiken is het weer warm geworden, "
        "en ze pauzeren om water te delen en uit te kijken over de groene vallei beneden.",
        "long"),
]


WAV_HEADER_SIZE = 44  # matches the server's _streaming_wav_header()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


@dataclass
class Result:
    lang: str
    label: str
    original: str
    transcription: str
    ttfa_ms: float          # time to first AUDIO byte (past the 44-byte WAV header)
    total_ms: float
    audio_bytes: int        # raw PCM bytes (header excluded)
    similarity: float
    # Streaming continuity diagnostics.
    chunk_count: int = 0    # distinct non-empty PCM packets received
    max_inter_chunk_ms: float = 0.0   # longest gap between PCM packets (underrun proxy)
    longest_stall_samples: int = 0    # gap expressed as samples of 24 kHz audio
    error: Optional[str] = None


def stream_tts(host: str, port: int, text: str, voice: str, language: str,
               timeout: int, sample_rate: int = 24000,
               n_candidates: int = 1,
               ) -> tuple[bytes, float, float, int, float]:
    """Stream from /tts/stream and return continuity diagnostics.

    Returns (pcm_bytes, ttfa_ms, total_ms, chunk_count, max_gap_ms).

    pcm_bytes is raw PCM (header stripped). ttfa_ms is time from request-sent
    to the first byte of PCM (past the 44-byte WAV header).  chunk_count is
    the number of distinct socket reads that carried PCM, and max_gap_ms is
    the longest wall-time gap between successive PCM reads — a proxy for
    client buffer-underrun risk on long utterances.

    We read at a larger chunk_size than byte-level here (bytes-per-read is
    not a stable latency signal on recent urllib3; aggregated reads preserve
    TTFA accuracy because we time the first byte separately via the header
    boundary, which always lands in the very first inbound read).
    """
    url = f"http://{host}:{port}/tts/stream"
    payload = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "language": language,
        "output_format": "wav",
        "temperature": 0.0,
    }
    if n_candidates and n_candidates > 1:
        payload["n_candidates"] = int(n_candidates)

    header_buf = bytearray()
    pcm_buf = bytearray()
    ttfa_ms: Optional[float] = None
    chunk_count = 0
    max_gap_ms = 0.0
    last_chunk_t: Optional[float] = None
    t0 = time.perf_counter()

    with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            now = time.perf_counter()
            data = chunk
            if len(header_buf) < WAV_HEADER_SIZE:
                take = min(WAV_HEADER_SIZE - len(header_buf), len(data))
                header_buf.extend(data[:take])
                data = data[take:]
                if not data:
                    continue
            if ttfa_ms is None:
                ttfa_ms = (now - t0) * 1000.0
            if last_chunk_t is not None:
                gap_ms = (now - last_chunk_t) * 1000.0
                if gap_ms > max_gap_ms:
                    max_gap_ms = gap_ms
            last_chunk_t = now
            pcm_buf.extend(data)
            chunk_count += 1

    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttfa_ms is None:
        ttfa_ms = total_ms  # no audio at all — degenerate case
    return bytes(pcm_buf), float(ttfa_ms), float(total_ms), chunk_count, float(max_gap_ms)


def wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1,
                    bits: int = 16) -> bytes:
    """Real WAV file (with correct sizes) wrapping the received PCM, so Whisper
    can decode. The streaming header uses sentinel sizes which some clients
    reject; re-build with exact sizes here for the test harness.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def run_stt(host: str, port: int, audio_bytes: bytes, language: str,
            timeout: int) -> tuple[str, float]:
    url = f"http://{host}:{port}/asr"
    files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
    params = {"task": "transcribe", "language": language, "output": "json"}
    t0 = time.perf_counter()
    r = requests.post(url, files=files, params=params, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    body = r.text.strip()
    try:
        data = json.loads(body)
        text = data.get("text", body)
    except json.JSONDecodeError:
        text = body
    return text.strip(), elapsed_ms


def run_one(case, args) -> Result:
    # Back-compat: old 3-tuple cases still work; new cases carry a length tag.
    if len(case) == 4:
        lang, label, text, _length = case
    else:
        lang, label, text = case
    try:
        pcm, ttfa_ms, total_ms, chunk_count, max_gap_ms = stream_tts(
            args.tts_host, args.tts_port, text, args.voice, lang, args.tts_timeout,
            n_candidates=args.n_candidates,
        )
        wav_bytes = wrap_pcm_as_wav(pcm, sample_rate=24000)
        if args.save_audio:
            out = Path(args.save_audio) / f"{lang}_{label}_stream.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(wav_bytes)
        transcription, _ = run_stt(
            args.stt_host, args.stt_port, wav_bytes, lang, args.stt_timeout
        )
        sim = similarity(text, transcription)
        # samples of 24 kHz audio that fit into the longest inter-chunk gap;
        # values greater than chunk_count's average signal potential stall.
        longest_stall_samples = int(max_gap_ms * 24)
        return Result(lang, label, text, transcription, ttfa_ms, total_ms,
                      len(pcm), sim,
                      chunk_count=chunk_count,
                      max_inter_chunk_ms=max_gap_ms,
                      longest_stall_samples=longest_stall_samples)
    except Exception as e:
        return Result(lang, label, text, "", 0.0, 0.0, 0, 0.0, error=str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tts-host", default="localhost")
    p.add_argument("--tts-port", type=int, default=8004)
    p.add_argument("--stt-host", default="localhost")
    p.add_argument("--stt-port", type=int, default=9001)
    p.add_argument("--voice", default="Emily.wav")
    p.add_argument("--languages", default="", help="Comma-separated lang codes")
    p.add_argument("--length", default="",
                   help="Comma-separated length tags to run (short,medium,long). "
                        "Default runs all lengths.")
    p.add_argument("--tts-timeout", type=int, default=180)
    p.add_argument("--stt-timeout", type=int, default=60)
    p.add_argument("--save-audio", default="")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--ttfa-budget-ms", type=float, default=1000.0,
                   help="Max acceptable time-to-first-audio (excl. WAV header)")
    p.add_argument("--n-candidates", type=int, default=1,
                   help="Request best-of-N streaming (n>1 activates the "
                        "batched chunk-1 + commit + continue path).")
    args = p.parse_args()

    cases = TEST_CASES
    if args.languages:
        wanted = set(args.languages.split(","))
        cases = [c for c in cases if c[0] in wanted]
    if args.length:
        wanted_lengths = set(args.length.split(","))
        # Only 4-tuple cases have a length tag; older 3-tuples default to "short".
        cases = [c for c in cases
                 if (c[3] if len(c) == 4 else "short") in wanted_lengths]

    print(f"TTS={args.tts_host}:{args.tts_port}/tts/stream  STT={args.stt_host}:{args.stt_port}")
    print(f"Voice={args.voice}  TTFA budget={args.ttfa_budget_ms:.0f}ms  Sim threshold={args.threshold}\n")

    results = []
    for case in cases:
        print(f"[{case[0]}] {case[1]}: {case[2]}")
        r = run_one(case, args)
        results.append(r)
        if r.error:
            print(f"   ERROR: {r.error}\n")
            continue
        ttfa_ok = r.ttfa_ms <= args.ttfa_budget_ms
        sim_ok = r.similarity >= args.threshold
        stall_audio_s = r.longest_stall_samples / 24000.0
        status = "PASS" if ttfa_ok and sim_ok else "FAIL"
        audio_s = r.audio_bytes / (2 * 24000.0)
        print(f"   TTFA: {r.ttfa_ms:6.0f}ms  Total: {r.total_ms:6.0f}ms  "
              f"Audio: {audio_s:5.2f}s  Sim: {r.similarity:.2%}  "
              f"chunks={r.chunk_count}  max_gap={r.max_inter_chunk_ms:5.0f}ms "
              f"(~{stall_audio_s:.2f}s audio)  [{status}]")
        if not ttfa_ok:
            print(f"   ! TTFA over budget ({r.ttfa_ms:.0f} > {args.ttfa_budget_ms:.0f} ms)")
        if not sim_ok:
            print(f"   ! Similarity below threshold ({r.similarity:.2%} < {args.threshold:.0%})")
        print(f"   Heard: {r.transcription}\n")

    print("=" * 70)
    ok = [r for r in results if not r.error
          and r.similarity >= args.threshold
          and r.ttfa_ms <= args.ttfa_budget_ms]
    print(f"Summary: {len(ok)}/{len(results)} passed (sim >= {args.threshold:.0%}, "
          f"TTFA <= {args.ttfa_budget_ms:.0f}ms)")

    if results:
        ok_r = [r for r in results if not r.error]
        if ok_r:
            mean_ttfa = sum(r.ttfa_ms for r in ok_r) / len(ok_r)
            mean_total = sum(r.total_ms for r in ok_r) / len(ok_r)
            print(f"Mean TTFA: {mean_ttfa:.0f}ms   Mean total: {mean_total:.0f}ms")

    sys.exit(0 if len(ok) == len(results) else 1)


if __name__ == "__main__":
    main()
