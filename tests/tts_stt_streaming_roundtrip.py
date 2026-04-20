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


# (lang_code, label, sample_text)
TEST_CASES = [
    ("en", "English",  "Hello world. This is a streaming test of the text to speech system."),
    ("nl", "Dutch",    "Goedemorgen. Dit is een streaming test van het spraaksynthese systeem."),
    ("de", "German",   "Guten Tag. Dies ist ein Streaming-Test des Sprachsynthesesystems."),
    ("fr", "French",   "Bonjour. Ceci est un test en streaming du système de synthèse vocale."),
    ("es", "Spanish",  "Hola. Esta es una prueba en streaming del sistema de síntesis de voz."),
    ("it", "Italian",  "Ciao. Questo è un test in streaming del sistema di sintesi vocale."),
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
    error: Optional[str] = None


def stream_tts(host: str, port: int, text: str, voice: str, language: str,
               timeout: int) -> tuple[bytes, float, float]:
    """Stream from /tts/stream and return (pcm_bytes, ttfa_ms, total_ms).

    pcm_bytes is raw PCM (header stripped). ttfa_ms is time from request-sent
    to the first byte of PCM (i.e. skipping the initial 44-byte WAV header).
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

    header_buf = bytearray()
    pcm_buf = bytearray()
    ttfa_ms: Optional[float] = None
    t0 = time.perf_counter()

    with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1):  # byte-level for accurate TTFA
            if chunk is None:
                continue
            if len(header_buf) < WAV_HEADER_SIZE:
                take = min(WAV_HEADER_SIZE - len(header_buf), len(chunk))
                header_buf.extend(chunk[:take])
                remainder = chunk[take:]
                if len(header_buf) == WAV_HEADER_SIZE and remainder:
                    if ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t0) * 1000.0
                    pcm_buf.extend(remainder)
                continue
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t0) * 1000.0
            pcm_buf.extend(chunk)

    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttfa_ms is None:
        ttfa_ms = total_ms  # no audio at all — degenerate case
    return bytes(pcm_buf), float(ttfa_ms), float(total_ms)


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
    lang, label, text = case
    try:
        pcm, ttfa_ms, total_ms = stream_tts(
            args.tts_host, args.tts_port, text, args.voice, lang, args.tts_timeout
        )
        wav_bytes = wrap_pcm_as_wav(pcm, sample_rate=24000)
        if args.save_audio:
            out = Path(args.save_audio) / f"{lang}_stream.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(wav_bytes)
        transcription, _ = run_stt(
            args.stt_host, args.stt_port, wav_bytes, lang, args.stt_timeout
        )
        sim = similarity(text, transcription)
        return Result(lang, label, text, transcription, ttfa_ms, total_ms,
                      len(pcm), sim)
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
    p.add_argument("--tts-timeout", type=int, default=180)
    p.add_argument("--stt-timeout", type=int, default=60)
    p.add_argument("--save-audio", default="")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--ttfa-budget-ms", type=float, default=1000.0,
                   help="Max acceptable time-to-first-audio (excl. WAV header)")
    args = p.parse_args()

    cases = TEST_CASES
    if args.languages:
        wanted = set(args.languages.split(","))
        cases = [c for c in cases if c[0] in wanted]

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
        status = "PASS" if ttfa_ok and sim_ok else "FAIL"
        print(f"   TTFA: {r.ttfa_ms:6.0f}ms  Total: {r.total_ms:6.0f}ms  "
              f"PCM: {r.audio_bytes/1024:6.1f}KB  Sim: {r.similarity:.2%}  [{status}]")
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
