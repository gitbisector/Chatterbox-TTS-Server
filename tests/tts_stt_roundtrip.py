#!/usr/bin/env python3
"""
TTS → STT roundtrip quality test.

Generates audio from text via the TTS server, transcribes it via Whisper,
and compares the transcription back to the original text. Useful for verifying
that an engine variant (PyTorch vs ONNX, fp32 vs fp16) produces understandable speech.

Usage:
    python3 tts_stt_roundtrip.py
    python3 tts_stt_roundtrip.py --tts-port 8005 --stt-port 9001
    python3 tts_stt_roundtrip.py --voice Roel.wav --languages nl,en
"""

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# (lang_code, label, sample_text)
TEST_CASES = [
    ("en", "English",  "Hello world. This is a test of the text to speech system."),
    ("nl", "Dutch",    "Goedemorgen. Dit is een test van het spraaksynthese systeem."),
    ("de", "German",   "Guten Tag. Dies ist ein Test des Sprachsynthesesystems."),
    ("fr", "French",   "Bonjour. Ceci est un test du système de synthèse vocale."),
    ("es", "Spanish",  "Hola. Esta es una prueba del sistema de síntesis de voz."),
    ("it", "Italian",  "Ciao. Questo è un test del sistema di sintesi vocale."),
]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for fair comparison."""
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
    tts_seconds: float
    stt_seconds: float
    audio_bytes: int
    similarity: float
    error: Optional[str] = None


def run_tts(host: str, port: int, text: str, voice: str, language: str,
            timeout: int) -> tuple[bytes, float]:
    url = f"http://{host}:{port}/tts"
    payload = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "language": language,
        "output_format": "wav",
        "split_text": False,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.content, elapsed


def run_stt(host: str, port: int, audio_bytes: bytes, language: str,
            timeout: int) -> tuple[str, float]:
    url = f"http://{host}:{port}/asr"
    files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
    params = {"task": "transcribe", "language": language, "output": "json"}
    t0 = time.perf_counter()
    r = requests.post(url, files=files, params=params, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.text.strip()
    try:
        data = json.loads(body)
        text = data.get("text", body)
    except json.JSONDecodeError:
        text = body
    return text.strip(), elapsed


def run_one(case, args) -> Result:
    lang, label, text = case
    try:
        audio, tts_t = run_tts(args.tts_host, args.tts_port, text, args.voice, lang, args.tts_timeout)
        if args.save_audio:
            out = Path(args.save_audio) / f"{lang}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(audio)
        transcription, stt_t = run_stt(args.stt_host, args.stt_port, audio, lang, args.stt_timeout)
        sim = similarity(text, transcription)
        return Result(lang, label, text, transcription, tts_t, stt_t, len(audio), sim)
    except Exception as e:
        return Result(lang, label, text, "", 0.0, 0.0, 0, 0.0, error=str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tts-host", default="localhost")
    p.add_argument("--tts-port", type=int, default=8005, help="TTS port (8004=PyTorch, 8005=ONNX)")
    p.add_argument("--stt-host", default="localhost")
    p.add_argument("--stt-port", type=int, default=9001, help="Whisper STT port")
    p.add_argument("--voice", default="Emily.wav")
    p.add_argument("--languages", default="", help="Comma-separated lang codes (default: all)")
    p.add_argument("--tts-timeout", type=int, default=180)
    p.add_argument("--stt-timeout", type=int, default=60)
    p.add_argument("--save-audio", default="", help="Directory to save generated WAVs")
    p.add_argument("--threshold", type=float, default=0.70,
                   help="Similarity threshold below which a case fails (default 0.70)")
    args = p.parse_args()

    cases = TEST_CASES
    if args.languages:
        wanted = set(args.languages.split(","))
        cases = [c for c in cases if c[0] in wanted]

    print(f"TTS={args.tts_host}:{args.tts_port}  STT={args.stt_host}:{args.stt_port}")
    print(f"Voice={args.voice}  Threshold={args.threshold}\n")

    results = []
    for case in cases:
        print(f"[{case[0]}] {case[1]}: {case[2]}")
        r = run_one(case, args)
        results.append(r)
        if r.error:
            print(f"   ERROR: {r.error}\n")
            continue
        status = "PASS" if r.similarity >= args.threshold else "FAIL"
        print(f"   TTS: {r.tts_seconds:5.2f}s  STT: {r.stt_seconds:5.2f}s  "
              f"Audio: {r.audio_bytes/1024:6.1f}KB  Sim: {r.similarity:.2%}  [{status}]")
        print(f"   Heard: {r.transcription}\n")

    print("=" * 70)
    ok = [r for r in results if not r.error and r.similarity >= args.threshold]
    bad = [r for r in results if r.error or r.similarity < args.threshold]
    print(f"Summary: {len(ok)}/{len(results)} passed (sim ≥ {args.threshold:.0%})")

    if results:
        avg_tts = sum(r.tts_seconds for r in results if not r.error) / max(1, len(results) - len(bad))
        print(f"Mean TTS time: {avg_tts:.2f}s")

    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
