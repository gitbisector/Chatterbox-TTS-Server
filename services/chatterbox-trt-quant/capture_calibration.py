"""Capture LM input tensors from the running engine for FP8 calibration.

Runs a short sweep of synthesize() calls and saves the (inputs_embeds,
attention_mask, cfg_weight, past_key_values.*) arguments going INTO each
_run_lm call. modelopt.onnx.quantize consumes these to dial in per-layer
activation scales for FP8 weight-only quantization.

Writes an npz with one array per input name — each array stacked across
all captured steps.
"""
import os, sys, time
import numpy as np

# Guard so the capture hook only fires when CHATTERBOX_CAPTURE_CALIB=1 is set.
if os.environ.get("CHATTERBOX_CAPTURE_CALIB") != "1":
    print("CHATTERBOX_CAPTURE_CALIB not set; exiting without capture")
    sys.exit(0)

sys.path.insert(0, "/app")
os.environ["DISABLE_WATERMARK"] = "1"  # watermark has nothing to do with the LM
import engine_onnx as e

# ---- Monkey-patch _run_lm to save inputs ----
ORIG_RUN_LM = e._run_lm
MAX_STEPS_PER_CALL = int(os.environ.get("CAPTURE_MAX_STEPS", "20"))  # avoid exploding
OUT_PATH = os.environ.get("CAPTURE_OUT", "/tmp/calib_inputs.npz")

_captures = []  # list of dicts {name: ndarray}
_call_counter = {"n": 0}

def _wrapped_run_lm(inputs_embeds, attention_mask, cfg_weight, past_kv):
    _call_counter["n"] += 1
    # Tighter subsampling: keep only every 10th call. With 10 utterances
    # × ~50 steps each, that's ~50 samples — enough for FP8 calibration
    # (typical practice: 32-128 samples).
    if _call_counter["n"] % 10 == 0 and len(_captures) < 128:
        snap = {
            "inputs_embeds": inputs_embeds.astype(np.float16, copy=True),
            "attention_mask": attention_mask.astype(np.int64, copy=True),
            "cfg_weight": cfg_weight.astype(np.float16, copy=True),
        }
        # Grab all 30 layers × 2 KV inputs — modelopt needs the full set to run
        # the model forward during calibration.
        for i in range(30):
            snap[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"].copy()
            snap[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"].copy()
        _captures.append(snap)
    return ORIG_RUN_LM(inputs_embeds, attention_mask, cfg_weight, past_kv)

e._run_lm = _wrapped_run_lm

# ---- Load model + run a short diverse corpus ----
print("loading model...", flush=True)
assert e.load_model(), "load_model failed"

corpus = [
    ("en", "Hello world, this is a calibration sample."),
    ("en", "The quick brown fox jumps over the lazy dog."),
    ("en", "Please generate several seconds of speech for calibration."),
    ("nl", "Goedemorgen. Dit is een kalibratie voorbeeld."),
    ("nl", "Het sneeuwt vandaag en het is koud buiten."),
    ("nl", "De hoofdstad van Nederland is Amsterdam."),
    ("en", "This is another English sample with different prosody."),
    ("nl", "Nog een Nederlands voorbeeld voor de kalibratie."),
    ("en", "Short one."),
    ("nl", "Kort voorbeeld."),
]

t0 = time.perf_counter()
for i, (lang, text) in enumerate(corpus):
    print(f"[{i+1}/{len(corpus)}] {lang}: {text[:40]}", flush=True)
    wav, sr = e.synthesize(text=text, language=lang, temperature=0.0)
    if wav is None:
        print(f"  synthesize failed, skipping")
print(f"done in {time.perf_counter()-t0:.1f}s; captured {len(_captures)} LM calls", flush=True)

# ---- Save as a directory of per-sample npz (numpy-version-safe) ----
import shutil
OUT_DIR = OUT_PATH.rstrip("/")  # treat OUT_PATH as a directory stem
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR) if os.path.isdir(OUT_DIR) else os.remove(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"saving {len(_captures)} samples to {OUT_DIR}/...", flush=True)
for i, snap in enumerate(_captures):
    np.savez(os.path.join(OUT_DIR, f"sample_{i:04d}.npz"), **snap)
total = sum(os.path.getsize(os.path.join(OUT_DIR, f))
            for f in os.listdir(OUT_DIR))
print(f"saved {len(_captures)} npz, total {total/1e6:.1f} MB")
