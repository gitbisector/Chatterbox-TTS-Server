"""End-to-end Chatterbox TTS benchmark: batch/stream, n=1/n=3, EN/NL.

Runs 3 trials per (lang, mode), computes wall / duration / RTF. Streaming mode
also records TTFA (time to first PCM byte after the WAV header). Similarity
against the source text is measured via the local Whisper container (STT at
localhost:9001 by default) for the two batch modes; streaming reuses the
concatenated PCM.

Put this somewhere persistent — previous bench scripts lived in /tmp and got
wiped between sessions.
"""
import argparse
import difflib
import io
import json
import re
import statistics
import sys
import time
import wave
from dataclasses import dataclass

import requests


EN_TEXT = (
    "This is a medium length English sample used for benchmarking the "
    "Chatterbox text to speech engine. The sentence is intentionally long "
    "enough to cover several hundred decode steps."
)
NL_TEXT = (
    "Dit is een Nederlands voorbeeld van gemiddelde lengte voor een "
    "prestatietest van de Chatterbox spraaksynthese. De zin is bewust lang "
    "genoeg om meerdere honderden decoderingsstappen te omvatten."
)

CASES = [("en", EN_TEXT), ("nl", NL_TEXT)]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def wav_duration(wav_bytes: bytes) -> float:
    """Duration of a PCM WAV. Streaming responses emit a header with
    data size = 0xFFFFFFFF (unknown), which makes ``wave`` report bogus
    nframes — fall back to computing from byte count in that case."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        nframes = w.getnframes()
        framerate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
    declared = nframes * channels * sampwidth
    pcm_bytes = len(wav_bytes) - 44  # PCM WAV header is 44 bytes
    if declared > pcm_bytes * 2 or declared <= 0:
        nframes = pcm_bytes // (channels * sampwidth)
    return nframes / framerate


def stt(host: str, port: int, wav_bytes: bytes, lang: str) -> str:
    url = f"http://{host}:{port}/asr"
    files = {"audio_file": ("audio.wav", wav_bytes, "audio/wav")}
    params = {"task": "transcribe", "language": lang, "output": "json"}
    r = requests.post(url, files=files, params=params, timeout=120)
    r.raise_for_status()
    body = r.text.strip()
    try:
        data = json.loads(body)
        return data.get("text", body).strip()
    except json.JSONDecodeError:
        return body


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def run_batch(tts_url: str, text: str, lang: str, voice: str,
              n_candidates: int, timeout: int) -> tuple[bytes, float]:
    payload = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "language": lang,
        "output_format": "wav",
        "split_text": False,
    }
    if n_candidates and n_candidates > 1:
        payload["n_candidates"] = n_candidates
    t0 = time.perf_counter()
    r = requests.post(f"{tts_url}/tts", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    return r.content, wall


def run_stream(tts_url: str, text: str, lang: str, voice: str,
               timeout: int) -> tuple[bytes, float, float]:
    """Returns (full_wav_bytes, total_wall, ttfa_seconds)."""
    payload = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "language": lang,
        "output_format": "wav",
        "split_text": False,
    }
    t0 = time.perf_counter()
    ttfa = None
    collected = bytearray()
    header_bytes_needed = 44  # standard PCM WAV header
    header_seen = 0
    with requests.post(f"{tts_url}/tts/stream", json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None):
            if not chunk:
                continue
            # TTFA = first byte of PCM after the 44-byte WAV header.
            if ttfa is None:
                remaining = header_bytes_needed - header_seen
                if len(chunk) <= remaining:
                    header_seen += len(chunk)
                    collected.extend(chunk)
                    continue
                collected.extend(chunk[:remaining])
                header_seen += remaining
                ttfa = time.perf_counter() - t0
                collected.extend(chunk[remaining:])
            else:
                collected.extend(chunk)
    wall = time.perf_counter() - t0
    return bytes(collected), wall, ttfa if ttfa is not None else wall


@dataclass
class Trial:
    lang: str
    mode: str  # "batch_n1", "batch_n3", "stream"
    wall: float
    dur: float
    rtf: float
    ttfa: float | None
    sim: float


def fmt(vals, n=3):
    return f"{statistics.median(vals):.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tts", default="http://localhost:8004")
    p.add_argument("--stt-host", default="localhost")
    p.add_argument("--stt-port", type=int, default=9001)
    p.add_argument("--voice", default="Emily.wav")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--skip-stt", action="store_true")
    p.add_argument("--label", default="INT4")
    args = p.parse_args()

    all_trials: list[Trial] = []

    print(f"label={args.label}  tts={args.tts}  stt={args.stt_host}:{args.stt_port}  voice={args.voice}  trials={args.trials}")
    print()

    def sim_of(wav, lang, text):
        if args.skip_stt:
            return float("nan")
        try:
            transcript = stt(args.stt_host, args.stt_port, wav, lang)
            return similarity(text, transcript)
        except Exception as e:
            print(f"  STT failed: {e}", file=sys.stderr)
            return float("nan")

    # Warm up once per case so first-call overhead doesn't pollute trial 1.
    for lang, text in CASES:
        print(f"[warmup] {lang} batch_n1 ...")
        run_batch(args.tts, text, lang, args.voice, 1, args.timeout)

    for lang, text in CASES:
        for mode in ("batch_n1", "stream", "batch_n3"):
            print(f"[{args.label}] {lang} {mode}:")
            for i in range(args.trials):
                if mode == "batch_n1":
                    wav, wall = run_batch(args.tts, text, lang, args.voice, 1, args.timeout)
                    ttfa = None
                elif mode == "batch_n3":
                    wav, wall = run_batch(args.tts, text, lang, args.voice, 3, args.timeout)
                    ttfa = None
                else:
                    wav, wall, ttfa = run_stream(args.tts, text, lang, args.voice, args.timeout)
                dur = wav_duration(wav)
                rtf = wall / dur if dur > 0 else float("nan")
                sim = sim_of(wav, lang, text)
                ttfa_str = f" ttfa={ttfa*1000:6.1f}ms" if ttfa is not None else ""
                print(f"  trial {i+1}: wall={wall:5.2f}s dur={dur:5.2f}s rtf={rtf:.3f}{ttfa_str} sim={sim:.3f}")
                all_trials.append(Trial(lang, mode, wall, dur, rtf, ttfa, sim))

    print()
    print("=== MEDIANS ===")
    by_key: dict[tuple[str, str], list[Trial]] = {}
    for t in all_trials:
        by_key.setdefault((t.lang, t.mode), []).append(t)
    header = f"{'lang':>4} {'mode':>10} {'wall':>6} {'dur':>6} {'rtf':>6} {'ttfa_ms':>8} {'sim':>6}"
    print(header)
    for (lang, mode), trs in by_key.items():
        med_wall = statistics.median(t.wall for t in trs)
        med_dur = statistics.median(t.dur for t in trs)
        med_rtf = statistics.median(t.rtf for t in trs)
        ttfa_vals = [t.ttfa for t in trs if t.ttfa is not None]
        med_ttfa = statistics.median(ttfa_vals) * 1000 if ttfa_vals else float("nan")
        sim_vals = [t.sim for t in trs if not (t.sim != t.sim)]  # drop NaN
        med_sim = statistics.median(sim_vals) if sim_vals else float("nan")
        print(f"{lang:>4} {mode:>10} {med_wall:6.2f} {med_dur:6.2f} {med_rtf:6.3f} {med_ttfa:8.1f} {med_sim:6.3f}")


if __name__ == "__main__":
    main()
