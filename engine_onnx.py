# File: engine_onnx.py
# ONNX Runtime TTS inference engine for Chatterbox Multilingual using the v2 custom export.
#
# Uses:
#   - speech_encoder.onnx (community, unchanged — called once per inference)
#   - embed_tokens_v2.onnx (our re-export, scatter-free, batch=2 for CFG)
#   - language_model_v2.onnx (our re-export, CFG batching + layer-9/12/13 attention, fp16)
#   - conditional_decoder_n{4,6,10}.onnx (our re-export, parameterized CFM steps)
#
# With the v2 export we now have:
#   - Classifier-free guidance (cond + cfg_weight * (cond - uncond))
#   - AlignmentStreamAnalyzer running on exposed layer-9/12/13 attention
#   - fp16 LM weights
#   - Scatter-free graphs (CUDA-graph capturable)
#   - Configurable CFM step count (speed/quality knob)

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from unicodedata import category

import librosa
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from config import config_manager

# Local import — we package the alignment analyzer alongside the export scripts.
_SERVICES_DIR = Path(__file__).parent / "services" / "chatterbox-onnx-export"
if _SERVICES_DIR.exists() and str(_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICES_DIR))
from alignment_runtime import AlignmentStreamAnalyzer  # noqa: E402

logger = logging.getLogger(__name__)

# --- Constants ---
COMMUNITY_REPO_ID = "onnx-community/chatterbox-multilingual-ONNX"
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_HIDDEN_LAYERS = 30
NUM_KEY_VALUE_HEADS = 16
HEAD_DIM = 64

SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

# --- Module state ---
MODEL_LOADED: bool = False
model_device: Optional[str] = None

_speech_encoder_session: Optional[ort.InferenceSession] = None
_embed_tokens_session: Optional[ort.InferenceSession] = None
_language_model_session: Optional[ort.InferenceSession] = None
_cond_decoder_session: Optional[ort.InferenceSession] = None
_lm_output_names: list = []
_tokenizer = None
_model_dir: Optional[Path] = None
_cangjie_file: Optional[Path] = None
_default_voice_path: Optional[Path] = None
_cfm_n: int = 6
_watermarker = None  # perth.PerthImplicitWatermarker — shared; constructor loads a 37MB checkpoint


# --- Logits processor (numpy) ---

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def _sample(logits: np.ndarray, generate_tokens: np.ndarray, temperature: float,
            rep_processor: RepetitionPenaltyLogitsProcessor) -> np.ndarray:
    logits = rep_processor(generate_tokens, logits.astype(np.float32, copy=False))
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    if temperature > 0:
        lmax = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - lmax)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return np.array([[np.random.choice(probs.shape[-1], p=probs[0])]], dtype=np.int64)
    return np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)


# --- Language preprocessing (same as before) ---

_kakasi = None
_cangjie_converter = None


class ChineseCangjieConverter:
    def __init__(self, cangjie_file: str):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        with open(cangjie_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for entry in data:
            word, code = entry.split("\t")[:2]
            self.word2cj[word] = code
            if code not in self.cj2word:
                self.cj2word[code] = [word]
            else:
                self.cj2word[code].append(word)
        try:
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available — Chinese segmentation will be skipped")

    def _cangjie_encode(self, glyph: str):
        code = self.word2cj.get(glyph)
        if code is None:
            return None
        index = self.cj2word[code].index(glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)

    def __call__(self, text):
        output = []
        if self.segmenter is not None:
            full_text = " ".join(self.segmenter.cut(text))
        else:
            full_text = text
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = [f"[cj_{c}]" for c in cangjie]
                code.append("[cj_.]")
                output.append("".join(code))
            else:
                output.append(t)
        return "".join(output)


def _hiragana_normalize(text: str) -> str:
    global _kakasi
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        result = _kakasi.convert(text)
        out = []
        for r in result:
            inp = r['orig']
            hira = r["hira"]
            if any(19968 <= ord(c) <= 40959 for c in inp):
                if hira and hira[0] in ["は", "へ"]:
                    hira = " " + hira
                out.append(hira)
            elif all(12449 <= ord(c) <= 12538 for c in inp) if inp else False:
                out.append(r['orig'])
            else:
                out.append(inp)
        import unicodedata
        return unicodedata.normalize('NFKD', "".join(out))
    except ImportError:
        logger.warning("pykakasi not available — Japanese normalization skipped")
        return text


def _korean_normalize(text: str) -> str:
    def decompose_hangul(char):
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        return initial + medial + final
    return ''.join(decompose_hangul(c) for c in text).strip()


def _prepare_language(txt: str, language_id: str) -> str:
    global _cangjie_converter
    if language_id == 'zh':
        if _cangjie_converter is None and _cangjie_file is not None:
            _cangjie_converter = ChineseCangjieConverter(str(_cangjie_file))
        if _cangjie_converter is not None:
            txt = _cangjie_converter(txt)
    elif language_id == 'ja':
        txt = _hiragana_normalize(txt)
    elif language_id == 'ko':
        txt = _korean_normalize(txt)
    if language_id:
        txt = f"[{language_id.lower()}]{txt}"
    return txt


# --- Public API ---

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def get_model_info() -> dict:
    return {
        "loaded": MODEL_LOADED,
        "type": "multilingual-onnx-v2",
        "class_name": f"ONNX v2 (fp16, CFM-n{_cfm_n})",
        "device": model_device,
        "sample_rate": S3GEN_SR if MODEL_LOADED else None,
        "supports_paralinguistic_tags": False,
        "available_paralinguistic_tags": [],
        "turbo_available_in_package": False,
        "multilingual_available_in_package": True,
        "supports_multilingual": True,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


def _get_ort_providers() -> list:
    available = ort.get_available_providers()
    logger.info(f"ONNX Runtime available providers: {available}")
    providers: list = []
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
        }))
    providers.append("CPUExecutionProvider")
    return providers


def load_model() -> bool:
    global MODEL_LOADED, model_device, _cfm_n
    global _speech_encoder_session, _embed_tokens_session
    global _language_model_session, _cond_decoder_session, _lm_output_names
    global _tokenizer, _model_dir, _cangjie_file, _default_voice_path

    if MODEL_LOADED:
        logger.info("ONNX v2 TTS model already loaded.")
        return True

    try:
        # CFM step count: 4, 6, or 10 (default 6 — good balance)
        _cfm_n = config_manager.get_int("tts_engine.onnx_cfm_steps", 6)
        if _cfm_n not in (4, 6, 10):
            logger.warning(f"Unsupported CFM step count {_cfm_n}, falling back to 6")
            _cfm_n = 6

        # Paths — v2 models live in /app/onnx-models (bind-mounted or baked in).
        v2_dir = Path(os.environ.get("CHATTERBOX_ONNX_DIR", "/app/onnx-models"))
        if not v2_dir.exists():
            raise RuntimeError(f"ONNX v2 model directory not found: {v2_dir}. "
                               f"Mount your onnx-models folder there or set CHATTERBOX_ONNX_DIR.")

        logger.info(f"Loading ONNX v2 models from {v2_dir} (CFM steps={_cfm_n})")

        # Download community assets (speech_encoder, tokenizer, cangjie, default voice)
        hf_token = os.getenv("HF_TOKEN")
        cache_root = Path(config_manager.get_string("paths.model_cache", "./model_cache")) / "chatterbox-multilingual-onnx"
        for fname, subfolder in [
            ("speech_encoder.onnx", "onnx"),
            ("speech_encoder.onnx_data", "onnx"),
            ("tokenizer.json", None),
            ("tokenizer_config.json", None),
            ("Cangjie5_TC.json", None),
            ("default_voice.wav", None),
        ]:
            hf_hub_download(
                repo_id=COMMUNITY_REPO_ID,
                filename=fname,
                subfolder=subfolder,
                local_dir=str(cache_root),
                token=hf_token,
            )
        _model_dir = cache_root
        _cangjie_file = cache_root / "Cangjie5_TC.json"
        _default_voice_path = cache_root / "default_voice.wav"

        providers = _get_ort_providers()
        model_device = "cuda" if any("CUDA" in (p if isinstance(p, str) else p[0]) for p in providers) else "cpu"

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info("Loading speech_encoder (community)...")
        _speech_encoder_session = ort.InferenceSession(
            str(cache_root / "onnx" / "speech_encoder.onnx"), sess_options, providers=providers
        )

        logger.info("Loading embed_tokens_v2...")
        _embed_tokens_session = ort.InferenceSession(
            str(v2_dir / "embed_tokens_v2.onnx"), sess_options, providers=providers
        )

        logger.info("Loading language_model_v2 (CFG + attention, fp16)...")
        _language_model_session = ort.InferenceSession(
            str(v2_dir / "language_model_v2.onnx"), sess_options, providers=providers
        )
        _lm_output_names = [o.name for o in _language_model_session.get_outputs()]
        logger.info(f"LM has {len(_lm_output_names)} outputs: logits, attn_layers, + {len(_lm_output_names)-2} present KV entries")

        logger.info(f"Loading conditional_decoder_n{_cfm_n}...")
        _cond_decoder_session = ort.InferenceSession(
            str(v2_dir / f"conditional_decoder_n{_cfm_n}.onnx"), sess_options, providers=providers
        )

        logger.info("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(str(cache_root))

        # Load watermarker once. PerthImplicitWatermarker.__init__ loads a 37MB
        # checkpoint via torch.load — creating it per-request leaked ~30 MiB/call.
        # We'll also pad audio to a fixed length before each apply_watermark call
        # (below), because Perth's CPU torch.stft caches FFT plans per input shape,
        # and variable-length audio causes unbounded plan growth (~35 MiB/call).
        global _watermarker
        try:
            import perth
            _watermarker = perth.PerthImplicitWatermarker(device="cpu")
            logger.info("Perth watermarker loaded on cpu (once, shared)")
        except ImportError:
            logger.info("Perth not available — watermarking disabled")
            _watermarker = None

        MODEL_LOADED = True
        logger.info(f"ONNX v2 TTS model loaded on {model_device}, CFM-n{_cfm_n}, sample rate {S3GEN_SR} Hz")

        # Pre-warm the ORT CUDA memory arena with max-size inputs so it allocates
        # its final footprint here rather than growing per-request. ORT's arena
        # never shrinks, so after this point the memory footprint is stable.
        _warmup_arena()

        return True

    except Exception as e:
        logger.error(f"Failed to load ONNX v2 model: {e}", exc_info=True)
        MODEL_LOADED = False
        return False


def _warmup_arena():
    """Run each ORT session once with near-max-size inputs.

    ORT's CUDA arena grows on first encounter of every new shape and never
    shrinks. Doing this at load time means the process footprint plateaus
    immediately instead of creeping up over the first few dozen requests.

    Sizes chosen to cover the common case: up to ~500 text tokens, the
    default 800-token decode budget, and a ~240-token speech prompt (typical
    for 10-second reference audio).
    """
    try:
        logger.info("Pre-warming ORT CUDA memory arena with max-size inputs...")
        t0 = time.perf_counter()

        # Size budgets — pick "large enough for real traffic" values.
        MAX_TEXT = 500        # text tokens (any single language uses well under this)
        MAX_COND = 40         # speaker conditioning embedding length (speech_encoder output is ~33)
        MAX_DECODE = 800      # decode steps (matches default max_new_tokens)
        MAX_PROMPT = 300      # reference speech prompt tokens (varies with ref-audio length)
        MAX_GEN = 800         # generated speech tokens (rare to exceed this)

        max_prefill = MAX_COND + MAX_TEXT + 1  # + BOS
        max_total = max_prefill + MAX_DECODE

        # 1. speech_encoder with ~10s of audio (the hot shape)
        audio = np.zeros((1, S3GEN_SR * 10), dtype=np.float32)
        _speech_encoder_session.run(None, {"audio_values": audio})

        # 2. embed_tokens at max prefill shape
        ids = np.zeros((2, MAX_TEXT + 1), dtype=np.int64)
        pos = np.arange(MAX_TEXT + 1, dtype=np.int64)[None, :].repeat(2, axis=0)
        _embed_tokens_session.run(None, {
            "input_ids": ids,
            "position_ids": pos,
            "exaggeration": np.array([0.5], dtype=np.float32),
        })

        # 3. LM prefill at max shape → forces largest KV allocation
        prefill_embeds = np.zeros((2, max_prefill, 1024), dtype=np.float16)
        attn = np.ones((2, max_prefill), dtype=np.int64)
        empty_kv = {
            f"past_key_values.{l}.{kv}": np.zeros((2, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM), dtype=np.float16)
            for l in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
        }
        _run_lm(prefill_embeds, attn, np.array(0.5, dtype=np.float16), empty_kv)

        # 4. LM decode step at max past-length (worst case for KV cache size)
        step_embeds = np.zeros((2, 1, 1024), dtype=np.float16)
        past_len = max_total - 1
        attn_max = np.ones((2, max_total), dtype=np.int64)
        big_kv = {
            f"past_key_values.{l}.{kv}": np.zeros(
                (2, NUM_KEY_VALUE_HEADS, past_len, HEAD_DIM), dtype=np.float16
            )
            for l in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
        }
        _run_lm(step_embeds, attn_max, np.array(0.5, dtype=np.float16), big_kv)

        # 5. vocoder at max speech-token length
        max_speech = np.zeros((1, MAX_PROMPT + MAX_GEN), dtype=np.int64)
        spk_emb = np.zeros((1, 192), dtype=np.float32)
        spk_feat = np.zeros((1, 500, 80), dtype=np.float32)
        _cond_decoder_session.run(None, {
            "speech_tokens": max_speech,
            "speaker_embeddings": spk_emb,
            "speaker_features": spk_feat,
        })

        # 6. A real synthesize() pass — this walks the KV cache through every
        # intermediate past_len (0, 1, 2, ... N_steps) that real traffic hits.
        # Direct max-shape runs above don't populate those intermediate extents.
        try:
            synthesize(
                text="This is a warmup pass that populates the memory arena with every intermediate key-value cache shape encountered during real decoding.",
                audio_prompt_path=str(_default_voice_path),
                temperature=0.4,
                exaggeration=0.5,
                cfg_weight=0.5,
                seed=0,
                language="en",
            )
        except Exception as e:
            logger.debug(f"Warmup synthesize skipped: {e}")

        logger.info(f"Arena warmup complete ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        # Warmup is a performance optimization — don't fail startup if it crashes.
        logger.warning(f"Arena warmup failed (continuing anyway): {e}")


def _run_lm(inputs_embeds: np.ndarray, attention_mask: np.ndarray,
            cfg_weight: np.ndarray, past_kv: dict) -> dict:
    """Run language_model_v2 and return dict keyed by output name."""
    feed = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cfg_weight": cfg_weight,
    }
    feed.update(past_kv)
    outs = _language_model_session.run(_lm_output_names, feed)
    return dict(zip(_lm_output_names, outs))


def _present_to_past(out_dict: dict) -> dict:
    """Map present.N.{key,value} → past_key_values.N.{key,value}."""
    return {
        f"past_key_values.{n}.{kv}": out_dict[f"present.{n}.{kv}"]
        for n in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
    }


def _attn_for_cand(attn_layers: np.ndarray, cand_idx: int = 0) -> np.ndarray:
    """Return (3, H, S, total_S) from a language-model attn_layers output.

    The Phase-1 export emitted shape (3, H, S, total_S) — a single candidate.
    The Phase-2b export emits (3, N, H, S, total_S) and we slice out the
    requested candidate. Handles both shapes so the engine keeps working
    across graph versions.
    """
    if attn_layers.ndim == 5:  # (3, N, H, S, total_S)
        return attn_layers[:, cand_idx]
    return attn_layers  # already (3, H, S, total_S)


# ---- Streaming helpers ---------------------------------------------------

# Audio samples emitted per speech token at S3GEN_SR (25 Hz tokens × 960 samples = 24 kHz).
SAMPLES_PER_SPEECH_TOKEN = S3GEN_SR // 25  # 960

# Measured on DGX Spark after arena warmup: LM ~11 ms/step, vocoder ~300 ms fixed +
# small per-token cost. These are conservative defaults; real deployments should
# set `streaming_first_chunk_budget_ms` in config and let `_compute_first_chunk_tokens`
# derive K1. See `project_spark_loader_profiling.md` and the streaming plan.
_LM_STEP_MS_DEFAULT = 12.0
_VOCODER_FIXED_MS = 300.0
_VOCODER_PER_TOKEN_MS = 1.0
_STREAM_OVERHEAD_MS = 120.0  # SE + ET + network/framing


def _compute_first_chunk_tokens(budget_ms: float,
                                lm_step_ms: float = _LM_STEP_MS_DEFAULT,
                                vocoder_fixed_ms: float = _VOCODER_FIXED_MS,
                                vocoder_per_token_ms: float = _VOCODER_PER_TOKEN_MS,
                                overhead_ms: float = _STREAM_OVERHEAD_MS,
                                min_tokens: int = 25,
                                max_tokens: int = 150) -> int:
    """Largest K1 whose first-chunk cost fits in `budget_ms`.

    Solves  K·lm_step + (vocoder_fixed + K·vocoder_per_token) + overhead ≤ budget.
    Floors at `min_tokens` so we never stream sub-1-second chunks (prosody goes
    bad below ~25 tokens ≈ 1 s audio) — if the budget is that tight, just
    degrade to one-shot by returning `min_tokens` and letting the caller decide.
    Ceils at `max_tokens` to keep first-audio reasonable on fast hardware.
    """
    denom = max(1e-6, lm_step_ms + vocoder_per_token_ms)
    headroom = budget_ms - vocoder_fixed_ms - overhead_ms
    k = int(headroom / denom) if headroom > 0 else 0
    return max(min_tokens, min(max_tokens, k))


def _emit_pcm(wav_f32: np.ndarray) -> bytes:
    """Float32 waveform in [-1, 1] → int16 little-endian PCM bytes."""
    clipped = np.clip(wav_f32, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _linear_crossfade(tail_a: np.ndarray, head_b: np.ndarray) -> np.ndarray:
    """Equal-length linear crossfade. tail_a fades out, head_b fades in."""
    n = min(len(tail_a), len(head_b))
    if n == 0:
        return np.concatenate([tail_a, head_b])
    fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return tail_a[:n] * fade + head_b[:n] * (1.0 - fade)


# ---- Denoiser (Phase 2a) -------------------------------------------------

_rnnoise_sr = 48000       # RNNoise is trained at 48 kHz
_rnnoise_available = None  # None=untested, True/False after first probe


def _rnnoise_can_load() -> bool:
    """One-time probe: is pyrnnoise importable?"""
    global _rnnoise_available
    if _rnnoise_available is not None:
        return _rnnoise_available
    try:
        import pyrnnoise  # noqa: F401
        _rnnoise_available = True
        logger.info("pyrnnoise available — denoise path enabled")
    except ImportError:
        _rnnoise_available = False
        logger.info("pyrnnoise not installed — denoise disabled")
    return _rnnoise_available


def new_rnnoise_denoiser():
    """Construct a fresh RNNoise denoiser instance. Call once per synthesis
    request to avoid cross-request state pollution (pyrnnoise.RNNoise keeps
    temporal state across denoise_chunk calls). Returns None if pyrnnoise
    isn't installed.
    """
    if not _rnnoise_can_load():
        return None
    import pyrnnoise
    return pyrnnoise.RNNoise(_rnnoise_sr)


def _simple_upsample_2x(wav: np.ndarray) -> np.ndarray:
    """Zero-stuff + 2-tap linear interpolation = 2× upsample with crude
    anti-aliasing. Fast (O(N), no FFT, no polyphase matrix) and memory-flat.
    Good enough for RNNoise's frequency range — RNNoise itself is trained for
    ~20 kHz bandwidth and doesn't care about perfect 24-48 kHz reconstruction.
    """
    n = wav.shape[0]
    out = np.empty(2 * n, dtype=np.float32)
    out[0::2] = wav
    out[1::2] = np.concatenate([(wav[:-1] + wav[1:]) * 0.5, wav[-1:]])
    return out


def _simple_downsample_2x(wav: np.ndarray) -> np.ndarray:
    """Anti-aliased 2× downsample: 2-tap moving average then decimate.
    Matches the reverse of _simple_upsample_2x well enough that a round-trip
    through RNNoise at 48 kHz preserves speech content at 24 kHz.
    """
    if wav.shape[0] < 2:
        return wav[::2].astype(np.float32)
    smoothed = np.empty_like(wav)
    smoothed[0] = wav[0]
    smoothed[1:] = (wav[:-1] + wav[1:]) * 0.5
    return smoothed[::2].astype(np.float32)


def _fft_resample(wav: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """FFT-based resample for non-2× ratios. O(N log N) memory + time."""
    from scipy.signal import resample
    new_len = int(round(len(wav) * out_sr / in_sr))
    return resample(wav, new_len).astype(np.float32)


def _denoise_rnnoise(wav: np.ndarray, denoiser, sr: int = S3GEN_SR) -> np.ndarray:
    """Run a float32 [-1, 1] waveform through the given RNNoise denoiser.

    Resamples to 48 kHz, converts to int16, runs RNNoise frame-by-frame, then
    converts back. The denoiser keeps state across calls so streaming chunks
    preserve continuity — pass the same `denoiser` instance for every chunk of
    one utterance.

    On ARM cores (Spark) this takes ~5–15 ms per second of audio.
    """
    if denoiser is None or wav.size == 0:
        return wav

    # Ensure 1-D float32 input (defensive — `resample_poly` blows up
    # memory allocating a (N, filter_len) matrix if handed the wrong shape).
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if wav.size == 0:
        return wav

    # Resample 24 kHz → 48 kHz. We use a simple linear-interpolation upsample
    # (for 2× factor, just alternate sample + midpoint average). FIR-based
    # resamplers blow up memory on long utterances with certain shapes, and
    # the 2× upsample here is just a sample-rate-match for RNNoise — we don't
    # need aggressive anti-aliasing because RNNoise operates in its own trained
    # frequency range.
    if sr != _rnnoise_sr:
        wav_48 = _simple_upsample_2x(wav) if _rnnoise_sr == 2 * sr else _fft_resample(wav, sr, _rnnoise_sr)
    else:
        wav_48 = wav

    # float32 [-1, 1] → int16 for pyrnnoise.
    wav_i16 = (np.clip(wav_48, -1.0, 1.0) * 32767.0).astype(np.int16)

    # denoise_chunk yields (vad_score, denoised_frame) tuples — each frame is
    # shape (1, 480) int16 at 48 kHz. Concatenate frames along the sample axis.
    # partial=True flushes the trailing short frame so sample counts add up.
    try:
        frames = [frame.ravel() for _vad, frame in denoiser.denoise_chunk(wav_i16, partial=True)]
    except Exception as e:
        logger.warning(f"RNNoise failed ({e}); returning raw audio")
        return wav

    if not frames:
        return wav

    out_48 = np.concatenate(frames).astype(np.float32) / 32767.0

    # Resample back to original sr. 48 → 24 kHz = take every other sample
    # after a simple 2-tap lowpass to suppress high-frequency aliases.
    if sr != _rnnoise_sr:
        out = _simple_downsample_2x(out_48) if _rnnoise_sr == 2 * sr else _fft_resample(out_48, _rnnoise_sr, sr)
    else:
        out = out_48

    # RNNoise pads to 10 ms frames; pad/trim so the caller's sample-index
    # bookkeeping (crossfade, etc.) still lines up.
    if len(out) < len(wav):
        out = np.pad(out, (0, len(wav) - len(out)))
    elif len(out) > len(wav):
        out = out[:len(wav)]
    return out


def _denoise_enabled() -> bool:
    """Read the denoise.enabled config knob (default True)."""
    return config_manager.get_bool("denoise.enabled", True)


# ---- Best-of-N helpers (Phase 2b) ---------------------------------------

def _whisper_score(wav_i16: bytes, reference_text: str, language: str,
                   timeout_s: float = 15.0) -> float:
    """POST a WAV to the local Whisper service and return difflib similarity
    of the transcription vs ``reference_text``. Returns 0.0 on any error.
    """
    import difflib
    import io
    import re
    import wave
    import urllib.request

    try:
        # Wrap PCM bytes in a WAV container.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(S3GEN_SR)
            w.writeframes(wav_i16)
        wav_bytes = buf.getvalue()

        host = config_manager.get_string("whisper.host", "localhost")
        port = config_manager.get_int("whisper.port", 9001)
        url = f"http://{host}:{port}/asr?task=transcribe&language={language}&output=txt"

        # multipart/form-data for the POST.
        boundary = "----candidateform"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio_file"; filename="c.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            heard = resp.read().decode("utf-8", errors="replace").strip()

        def norm(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
            return re.sub(r"\s+", " ", s).strip()

        return difflib.SequenceMatcher(None, norm(reference_text), norm(heard)).ratio()
    except Exception as e:
        logger.warning(f"Whisper validation failed: {e}")
        return 0.0


def _synthesize_batched_bestof_n(
    text: str,
    audio_prompt_path: Optional[str],
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    seed: int,
    language: str,
    n_candidates: int,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Run N candidates through the LM in one batched decode loop (batch=2N
    pairs (cond, uncond) per candidate), vocode + denoise each, Whisper-score
    them, and return the best waveform.

    Requires the Phase-2b re-exported ``language_model_v2.onnx`` with
    batch-dynamic CFG combine. N=1 still works but is pointless here —
    ``synthesize()`` should dispatch to the non-batched path for n=1.
    """
    if seed != 0:
        set_seed(seed)

    lang = language.lower() if language else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    # ---- Identical setup to synthesize() up to the prefill inputs. ----
    voice_path = audio_prompt_path or str(_default_voice_path)
    audio_values, _ = librosa.load(voice_path, sr=S3GEN_SR)
    audio_values = audio_values[np.newaxis, :].astype(np.float32)

    prepared_text = _prepare_language(text, lang)
    input_ids = _tokenizer(prepared_text, return_tensors="np")["input_ids"].astype(np.int64)
    position_ids = np.where(
        input_ids >= START_SPEECH_TOKEN,
        0,
        np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
    ).astype(np.int64)

    cond_emb_np, prompt_token, ref_x_vector, prompt_feat = _speech_encoder_session.run(
        None, {"audio_values": audio_values}
    )
    cond_emb_b2 = np.broadcast_to(
        cond_emb_np.astype(np.float16),
        (2, cond_emb_np.shape[1], cond_emb_np.shape[2]),
    ).copy()
    cond_len = cond_emb_b2.shape[1]

    bos_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
    input_ids_bos = np.concatenate([input_ids, bos_ids], axis=1)
    position_ids_bos = np.concatenate(
        [position_ids, np.array([[0]], dtype=np.int64)], axis=1
    )
    input_ids_b2 = np.concatenate([input_ids_bos, input_ids_bos], axis=0)
    position_ids_b2 = np.concatenate([position_ids_bos, position_ids_bos], axis=0)
    text_embeds = _embed_tokens_session.run(None, {
        "input_ids": input_ids_b2,
        "position_ids": position_ids_b2,
        "exaggeration": np.array([exaggeration], dtype=np.float32),
    })[0]
    text_len = input_ids.shape[1]

    prefill_embeds_b2 = np.concatenate([cond_emb_b2, text_embeds], axis=1)

    # ---- Tile to batch=2N (adjacent (cond, uncond) pairs per candidate). ----
    B = 2 * n_candidates
    prefill_embeds = np.tile(prefill_embeds_b2, (n_candidates, 1, 1))
    _, prefill_len, _ = prefill_embeds.shape
    attention_mask = np.ones((B, prefill_len), dtype=np.int64)
    past_kv = {
        f"past_key_values.{l}.{kv}": np.zeros(
            (B, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM), dtype=np.float16
        )
        for l in range(NUM_HIDDEN_LAYERS) for kv in ("key", "value")
    }
    cfg_scalar = np.array(cfg_weight, dtype=np.float16)

    max_new_tokens = config_manager.get_int("generation_defaults.max_tokens", 800)
    t_gen_start = time.perf_counter()
    rep_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)

    # Per-candidate state
    analyzers = [
        AlignmentStreamAnalyzer(
            text_tokens_slice=(cond_len, cond_len + text_len),
            eos_idx=STOP_SPEECH_TOKEN,
        ) for _ in range(n_candidates)
    ]
    gen_tokens = [
        np.array([[START_SPEECH_TOKEN]], dtype=np.int64) for _ in range(n_candidates)
    ]
    eos_flags = [False] * n_candidates
    last_token = [START_SPEECH_TOKEN] * n_candidates

    logger.info(
        f"Batched best-of-{n_candidates} gen start lang={lang} "
        f"cond_len={cond_len} text_len={text_len}"
    )

    # ---- Prefill + first sample per candidate ----
    out = _run_lm(prefill_embeds, attention_mask, cfg_scalar, past_kv)
    past_kv = _present_to_past(out)
    logits_full = out["logits"]           # (N, prefill_len, V) fp16
    attn_5d = out["attn_layers"]          # (3, N, H, prefill_len, prefill_len)

    for c in range(n_candidates):
        logits_c = logits_full[c:c+1, -1, :].astype(np.float32)
        logits_c = analyzers[c].step(logits_c, attn_5d[:, c], next_token=None)
        nt_c = _sample(logits_c, gen_tokens[c], temperature, rep_proc)
        gen_tokens[c] = np.concatenate([gen_tokens[c], nt_c], axis=-1)
        tok = int(nt_c[0, 0])
        last_token[c] = tok
        if tok == STOP_SPEECH_TOKEN:
            eos_flags[c] = True

    # ---- Decode loop at batch=2N ----
    for i in range(1, max_new_tokens):
        if all(eos_flags):
            break
        # Per-candidate last-sampled token, duplicated for (cond, uncond) rows.
        # Shape (2N, 1).
        nt_b2n = np.array(
            [[t] for t in last_token for _ in range(2)], dtype=np.int64
        )
        pos_ids_b2n = np.full((B, 1), i, dtype=np.int64)
        step_embeds = _embed_tokens_session.run(None, {
            "input_ids": nt_b2n,
            "position_ids": pos_ids_b2n,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        })[0]  # (2N, 1, 1024)

        attention_mask = np.concatenate(
            [attention_mask, np.ones((B, 1), dtype=np.int64)], axis=1
        )

        out = _run_lm(step_embeds, attention_mask, cfg_scalar, past_kv)
        past_kv = _present_to_past(out)
        logits_full = out["logits"]                 # (N, 1, V)
        attn_5d = out["attn_layers"]                # (3, N, H, 1, total_S)

        for c in range(n_candidates):
            if eos_flags[c]:
                continue
            logits_c = logits_full[c:c+1, -1, :].astype(np.float32)
            logits_c = analyzers[c].step(
                logits_c, attn_5d[:, c], next_token=last_token[c]
            )
            nt_c = _sample(logits_c, gen_tokens[c], temperature, rep_proc)
            gen_tokens[c] = np.concatenate([gen_tokens[c], nt_c], axis=-1)
            tok = int(nt_c[0, 0])
            last_token[c] = tok
            if tok == STOP_SPEECH_TOKEN:
                eos_flags[c] = True

    lm_elapsed = time.perf_counter() - t_gen_start
    lens = [gen_tokens[c].shape[1] - 1 for c in range(n_candidates)]  # minus BOS
    logger.info(
        f"Batched LM done in {lm_elapsed:.2f}s, per-cand tokens={lens}, "
        f"eos={eos_flags}"
    )

    # ---- Vocode each candidate (serial for now; vocoder batching is a
    #      follow-up optimisation), denoise, transcribe, score. ----
    candidates: list = []
    for c in range(n_candidates):
        speech_c = gen_tokens[c][:, 1:]  # strip BOS
        if speech_c.size > 0 and speech_c[0, -1] == STOP_SPEECH_TOKEN:
            speech_c = speech_c[:, :-1]
        speech_c = np.concatenate([prompt_token, speech_c], axis=1)
        wav_c = _cond_decoder_session.run(None, {
            "speech_tokens": speech_c,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        wav_c = np.squeeze(wav_c, axis=0)
        if _denoise_enabled():
            d = new_rnnoise_denoiser()
            if d is not None:
                wav_c = _denoise_rnnoise(wav_c, d, sr=S3GEN_SR)
        # PCM int16 bytes for Whisper
        pcm = _emit_pcm(wav_c)
        sim = _whisper_score(pcm, text, lang)
        logger.info(f"  candidate {c}: {len(wav_c)/S3GEN_SR:.2f}s, sim={sim:.2%}")
        candidates.append({"wav": wav_c, "sim": sim, "idx": c})

    if not candidates:
        logger.error("No candidates produced")
        return None, None

    # Pick highest similarity. If all below threshold, take the one with the
    # longest audio (petermg's fallback — likely contains most content).
    threshold = config_manager.get_float("candidate_validation_threshold", 0.70)
    best = max(candidates, key=lambda c: c["sim"])
    if best["sim"] < threshold:
        logger.warning(
            f"All candidates below threshold ({best['sim']:.2%} < {threshold:.2%}); "
            "falling back to longest audio"
        )
        best = max(candidates, key=lambda c: len(c["wav"]))
    logger.info(f"Best-of-{n_candidates}: chose idx={best['idx']} sim={best['sim']:.2%}")

    wav = best["wav"]

    # Apply watermark (skip when DISABLE_WATERMARK=1).
    if _watermarker is not None and not os.environ.get("DISABLE_WATERMARK"):
        with torch.no_grad():
            wav = _watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

    return torch.from_numpy(wav).unsqueeze(0).float(), S3GEN_SR


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    n_candidates: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if not MODEL_LOADED:
        logger.error("ONNX v2 TTS model is not loaded.")
        return None, None

    # Resolve candidate count: caller-supplied, or config default.
    if n_candidates is None:
        n_candidates = config_manager.get_int("n_candidates", 1)
    n_candidates = max(1, int(n_candidates))
    if n_candidates > 1:
        return _synthesize_batched_bestof_n(
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            language=language,
            n_candidates=n_candidates,
        )

    try:
        if seed != 0:
            set_seed(seed)

        lang = language.lower() if language else "en"
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language '{lang}', falling back to 'en'")
            lang = "en"

        # Load reference audio
        voice_path = audio_prompt_path or str(_default_voice_path)
        audio_values, _ = librosa.load(voice_path, sr=S3GEN_SR)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)

        # Tokenize
        prepared_text = _prepare_language(text, lang)
        input_ids = _tokenizer(prepared_text, return_tensors="np")["input_ids"].astype(np.int64)
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
        ).astype(np.int64)

        # Speech encoder (one-shot)
        # Community output names: audio_features (=cond_emb), audio_tokens (=prompt_token),
        # speaker_embeddings (=ref_x_vector), speaker_features (=prompt_feat)
        se_out = _speech_encoder_session.run(None, {"audio_values": audio_values})
        cond_emb_np, prompt_token, ref_x_vector, prompt_feat = se_out
        # cond_emb_np is float32; LM expects float16. Also need batch=2 for CFG.
        cond_emb_b2 = np.broadcast_to(
            cond_emb_np.astype(np.float16),
            (2, cond_emb_np.shape[1], cond_emb_np.shape[2]),
        ).copy()
        cond_len = cond_emb_b2.shape[1]

        # Embed text tokens (batch=2 for CFG). The embedder internally zeroes
        # text-row-1 (uncond) while keeping speech/exag rows untouched.
        # Append the BOS speech token so the LM's prefill is [text..., BOS], matching
        # PyTorch t3.inference() which concatenates BOS onto the text embeddings.
        bos_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        input_ids_with_bos = np.concatenate([input_ids, bos_ids], axis=1)  # (1, S+1)
        bos_pos = np.array([[0]], dtype=np.int64)
        position_ids_with_bos = np.concatenate([position_ids, bos_pos], axis=1)  # (1, S+1)

        input_ids_b2 = np.concatenate([input_ids_with_bos, input_ids_with_bos], axis=0)  # (2, S+1)
        position_ids_b2 = np.concatenate([position_ids_with_bos, position_ids_with_bos], axis=0)
        text_embeds = _embed_tokens_session.run(None, {
            "input_ids": input_ids_b2,
            "position_ids": position_ids_b2,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        })[0]  # (2, S+1, 1024) float16 — includes BOS at the end
        text_len = input_ids.shape[1]  # text-only length for the alignment slice

        # Concatenate [cond_emb | text_embeds_with_bos] for prefill
        prefill_embeds = np.concatenate([cond_emb_b2, text_embeds], axis=1)  # (2, C+S+1, 1024)
        batch_size, prefill_len, _ = prefill_embeds.shape

        attention_mask = np.ones((batch_size, prefill_len), dtype=np.int64)
        past_kv = {
            f"past_key_values.{l}.{kv}": np.zeros((batch_size, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM), dtype=np.float16)
            for l in range(NUM_HIDDEN_LAYERS)
            for kv in ("key", "value")
        }
        cfg_scalar = np.array(cfg_weight, dtype=np.float16)

        max_new_tokens = config_manager.get_int("generation_defaults.max_tokens", 800)
        t_gen_start = time.perf_counter()
        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        # Alignment analyzer: text is at positions [cond_len, cond_len + text_len)
        analyzer = AlignmentStreamAnalyzer(
            text_tokens_slice=(cond_len, cond_len + text_len),
            eos_idx=STOP_SPEECH_TOKEN,
        )

        logger.info(f"ONNX v2 gen start lang={lang} cond_len={cond_len} text_len={text_len}")

        # ===== Prefill =====
        out = _run_lm(prefill_embeds, attention_mask, cfg_scalar, past_kv)
        logits_full = out["logits"]  # (1, prefill_len, V) fp16
        attn_layers = _attn_for_cand(out["attn_layers"])  # (3, H, S, total_S) fp32
        past_kv = _present_to_past(out)

        # Take logits at the last position (the BOS-equivalent / start of speech)
        logits_step = logits_full[:, -1, :].astype(np.float32)
        logits_step = analyzer.step(logits_step, attn_layers, next_token=None)
        next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
        generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)

        steps_run = 1
        if int(next_token[0, 0]) != STOP_SPEECH_TOKEN:
            for i in range(1, max_new_tokens):
                # Build next-token inputs — batch=2, repeat the sampled token on both rows.
                nt_b2 = np.concatenate([next_token, next_token], axis=0)  # (2, 1)
                # Position ids follow the community convention: always i (+ base offset).
                pos_ids_b2 = np.full((2, 1), i, dtype=np.int64)
                step_embeds = _embed_tokens_session.run(None, {
                    "input_ids": nt_b2,
                    "position_ids": pos_ids_b2,
                    "exaggeration": np.array([exaggeration], dtype=np.float32),
                })[0]  # (2, 1, 1024) fp16

                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
                )

                out = _run_lm(step_embeds, attention_mask, cfg_scalar, past_kv)
                logits_step = out["logits"][:, -1, :].astype(np.float32)
                attn_layers = _attn_for_cand(out["attn_layers"])
                past_kv = _present_to_past(out)

                logits_step = analyzer.step(logits_step, attn_layers, next_token=int(next_token[0, 0]))
                next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
                generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)
                steps_run += 1

                if int(next_token[0, 0]) == STOP_SPEECH_TOKEN:
                    logger.info(f"EOS at step {i+1}")
                    break
            else:
                logger.warning(f"Hit max_new_tokens={max_new_tokens} without EOS")

        # Strip BOS, EOS (if present)
        speech_tokens = generate_tokens[:, 1:]
        if speech_tokens.size > 0 and speech_tokens[0, -1] == STOP_SPEECH_TOKEN:
            speech_tokens = speech_tokens[:, :-1]
        # Prepend reference prompt tokens so the vocoder has context
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

        lm_elapsed = time.perf_counter() - t_gen_start
        logger.info(
            f"Generated {speech_tokens.shape[1]} speech tokens in "
            f"{steps_run} steps ({lm_elapsed:.2f}s, {lm_elapsed/max(steps_run,1)*1000:.1f}ms/step), "
            f"running vocoder..."
        )

        wav = _cond_decoder_session.run(None, {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        wav = np.squeeze(wav, axis=0)

        # Phase 2a: RNNoise post-pass removes vocoder artefacts (low-freq
        # rumble, high-freq hiss). Single denoiser for the full utterance.
        if _denoise_enabled():
            denoiser = new_rnnoise_denoiser()
            if denoiser is not None:
                wav = _denoise_rnnoise(wav, denoiser, sr=S3GEN_SR)

        # Apply watermark using the shared watermarker (created once at load).
        # Set DISABLE_WATERMARK=1 to bypass (used for leak diagnosis).
        # Wrap in torch.no_grad() — Perth's apply_watermark doesn't set no_grad
        # internally, and autograd state accumulation appears to be the source
        # of the ~35 MiB/call leak.
        if _watermarker is not None and not os.environ.get("DISABLE_WATERMARK"):
            with torch.no_grad():
                wav = _watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
        return wav_tensor, S3GEN_SR

    except Exception as e:
        logger.error(f"ONNX v2 synthesis error: {e}", exc_info=True)
        return None, None


def synthesize_stream(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    first_chunk_budget_ms: Optional[float] = None,
    first_chunk_tokens_override: Optional[int] = None,
    crossfade_ms: float = 30.0,
):
    """Generator yielding (pcm_bytes, sample_rate) for the utterance.

    Fires the vocoder twice per utterance in the common case:
    - Chunk 1 once we've decoded K1 speech tokens (K1 derived from budget)
    - Chunk 2 after EOS, on [prompt | all generated]

    The two vocoder passes overlap in time-domain content; chunk 2's start
    re-generates chunk 1's audio up to the boundary. We hold back the last
    `crossfade_ms` of chunk 1's emitted audio and cross-fade into chunk 2's
    corresponding region to mask CFM noise differences at the boundary.

    If EOS fires before K1 (short utterance), yields one chunk identical to
    `synthesize()`'s output (no streaming boundary to worry about).
    """
    if not MODEL_LOADED:
        logger.error("ONNX v2 TTS model is not loaded.")
        return

    try:
        if seed != 0:
            set_seed(seed)

        lang = language.lower() if language else "en"
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language '{lang}', falling back to 'en'")
            lang = "en"

        # Resolve chunk sizing: explicit override wins, else derive from budget,
        # else fall back to config, else conservative default (800 ms target).
        if first_chunk_tokens_override is not None:
            k1 = max(1, int(first_chunk_tokens_override))
        else:
            budget = first_chunk_budget_ms
            if budget is None:
                budget = float(config_manager.get_int("streaming.first_chunk_budget_ms", 800))
            k1 = _compute_first_chunk_tokens(budget)
        logger.info(f"Streaming config: K1={k1} tokens (~{k1/25.0:.2f} s audio)")

        # Phase 2a: RNNoise post-pass. Use a FRESH denoiser per vocoder output
        # (not per emitted sub-chunk) so that sample boundaries inside one
        # vocoder output stay continuous, and the two vocoder passes (chunk 1
        # and chunk 2) each get their own clean state. Matters for the
        # crossfade: denoising each vocoder output independently keeps content
        # at matching sample positions filter-aligned between chunk1 and chunk2.
        denoise_on = _denoise_enabled()

        def _denoise_full_wav(w: np.ndarray) -> np.ndarray:
            if not denoise_on or w.size == 0:
                return w
            d = new_rnnoise_denoiser()
            if d is None:
                return w
            return _denoise_rnnoise(w, d, sr=S3GEN_SR)

        # ---- Identical setup to synthesize() up to the decode loop. ----
        voice_path = audio_prompt_path or str(_default_voice_path)
        audio_values, _ = librosa.load(voice_path, sr=S3GEN_SR)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)

        prepared_text = _prepare_language(text, lang)
        input_ids = _tokenizer(prepared_text, return_tensors="np")["input_ids"].astype(np.int64)
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
        ).astype(np.int64)

        se_out = _speech_encoder_session.run(None, {"audio_values": audio_values})
        cond_emb_np, prompt_token, ref_x_vector, prompt_feat = se_out
        cond_emb_b2 = np.broadcast_to(
            cond_emb_np.astype(np.float16),
            (2, cond_emb_np.shape[1], cond_emb_np.shape[2]),
        ).copy()
        cond_len = cond_emb_b2.shape[1]

        bos_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        input_ids_with_bos = np.concatenate([input_ids, bos_ids], axis=1)
        bos_pos = np.array([[0]], dtype=np.int64)
        position_ids_with_bos = np.concatenate([position_ids, bos_pos], axis=1)

        input_ids_b2 = np.concatenate([input_ids_with_bos, input_ids_with_bos], axis=0)
        position_ids_b2 = np.concatenate([position_ids_with_bos, position_ids_with_bos], axis=0)
        text_embeds = _embed_tokens_session.run(None, {
            "input_ids": input_ids_b2,
            "position_ids": position_ids_b2,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        })[0]
        text_len = input_ids.shape[1]

        prefill_embeds = np.concatenate([cond_emb_b2, text_embeds], axis=1)
        batch_size, prefill_len, _ = prefill_embeds.shape
        attention_mask = np.ones((batch_size, prefill_len), dtype=np.int64)
        past_kv = {
            f"past_key_values.{l}.{kv}": np.zeros((batch_size, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM), dtype=np.float16)
            for l in range(NUM_HIDDEN_LAYERS)
            for kv in ("key", "value")
        }
        cfg_scalar = np.array(cfg_weight, dtype=np.float16)

        max_new_tokens = config_manager.get_int("generation_defaults.max_tokens", 800)
        t_gen_start = time.perf_counter()
        t_first_emit = None
        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        analyzer = AlignmentStreamAnalyzer(
            text_tokens_slice=(cond_len, cond_len + text_len),
            eos_idx=STOP_SPEECH_TOKEN,
        )

        logger.info(
            f"ONNX v2 stream-gen start lang={lang} cond_len={cond_len} text_len={text_len}"
        )

        # Prefill + first sampled token (same as synthesize()).
        out = _run_lm(prefill_embeds, attention_mask, cfg_scalar, past_kv)
        past_kv = _present_to_past(out)
        logits_step = out["logits"][:, -1, :].astype(np.float32)
        logits_step = analyzer.step(logits_step, _attn_for_cand(out["attn_layers"]), next_token=None)
        next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
        generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)

        steps_run = 1
        hit_eos = int(next_token[0, 0]) == STOP_SPEECH_TOKEN
        chunk1_emitted = False     # did we emit a partial chunk before EOS?
        crossfade_samples = int(crossfade_ms * S3GEN_SR / 1000.0)
        held_back_tail: Optional[np.ndarray] = None  # fade-out region held back for crossfade
        chunk1_audio_len: Optional[int] = None  # chunk 1's emitted audio length in samples

        if not hit_eos:
            for i in range(1, max_new_tokens):
                nt_b2 = np.concatenate([next_token, next_token], axis=0)
                pos_ids_b2 = np.full((2, 1), i, dtype=np.int64)
                step_embeds = _embed_tokens_session.run(None, {
                    "input_ids": nt_b2,
                    "position_ids": pos_ids_b2,
                    "exaggeration": np.array([exaggeration], dtype=np.float32),
                })[0]
                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
                )
                out = _run_lm(step_embeds, attention_mask, cfg_scalar, past_kv)
                past_kv = _present_to_past(out)
                logits_step = out["logits"][:, -1, :].astype(np.float32)
                logits_step = analyzer.step(logits_step, _attn_for_cand(out["attn_layers"]), next_token=int(next_token[0, 0]))
                next_token = _sample(logits_step, generate_tokens, temperature, rep_proc)
                generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)
                steps_run += 1

                if int(next_token[0, 0]) == STOP_SPEECH_TOKEN:
                    logger.info(f"EOS at step {i+1}")
                    hit_eos = True
                    break

                # Fire chunk 1 vocoder once we've accumulated K1 generated tokens
                # (prefix excludes the seed START_SPEECH_TOKEN, hence -1).
                if (not chunk1_emitted) and (steps_run - 1 >= k1):
                    gen_tail = generate_tokens[:, 1:]  # strip START_SPEECH_TOKEN
                    chunk1_tokens = np.concatenate([prompt_token, gen_tail], axis=1)
                    wav_c1 = _cond_decoder_session.run(None, {
                        "speech_tokens": chunk1_tokens,
                        "speaker_embeddings": ref_x_vector,
                        "speaker_features": prompt_feat,
                    })[0]
                    # Decoder already trims prompt audio internally — wav_c1 is
                    # just the generated speech.
                    gen_audio_c1 = np.squeeze(wav_c1, axis=0)
                    # Denoise the full chunk-1 waveform BEFORE slicing into
                    # reliable / held_back / emit regions so the splits stay
                    # aligned (RNNoise is stateful across frames within one call).
                    gen_audio_c1 = _denoise_full_wav(gen_audio_c1)

                    # Discard chunk 1's last CHUNK1_TAIL_MARGIN_TOKENS worth of
                    # audio: the vocoder generated those without future context
                    # and they tend to contain edge artifacts (dropped consonants,
                    # flutter). Chunk 2 re-generates them with full context.
                    CHUNK1_TAIL_MARGIN_TOKENS = 3
                    margin_samples = CHUNK1_TAIL_MARGIN_TOKENS * SAMPLES_PER_SPEECH_TOKEN
                    reliable_end = max(0, len(gen_audio_c1) - margin_samples)
                    reliable = gen_audio_c1[:reliable_end]
                    chunk1_audio_len = reliable_end

                    # Emit everything except the last crossfade_samples of the
                    # reliable region, which we hold back and blend with chunk 2.
                    if len(reliable) > crossfade_samples:
                        emit_now = reliable[:-crossfade_samples]
                        held_back_tail = reliable[-crossfade_samples:]
                    else:
                        # Reliable audio shorter than crossfade window — emit
                        # all, no crossfade.
                        emit_now = reliable
                        held_back_tail = None
                    if t_first_emit is None:
                        t_first_emit = time.perf_counter()
                    yield _emit_pcm(emit_now), S3GEN_SR
                    chunk1_emitted = True
            else:
                logger.warning(f"Hit max_new_tokens={max_new_tokens} without EOS")

        # Strip BOS, EOS (if present) and prepend reference prompt tokens.
        speech_tokens = generate_tokens[:, 1:]
        if speech_tokens.size > 0 and speech_tokens[0, -1] == STOP_SPEECH_TOKEN:
            speech_tokens = speech_tokens[:, :-1]
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

        lm_elapsed = time.perf_counter() - t_gen_start
        logger.info(
            f"Stream LM done: {speech_tokens.shape[1]} tokens in {steps_run} steps "
            f"({lm_elapsed:.2f}s, {lm_elapsed/max(steps_run,1)*1000:.1f}ms/step)"
        )

        # Final vocoder pass on the full token sequence. Decoder already trims
        # prompt audio — gen_audio_full is just the generated speech.
        t_voc = time.perf_counter()
        wav_full = _cond_decoder_session.run(None, {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        gen_audio_full = np.squeeze(wav_full, axis=0)
        # Phase 2a: denoise the full chunk-2 waveform before crossfade math
        # (fresh denoiser state matches chunk-1's fresh state so corresponding
        # sample positions stay filter-aligned).
        gen_audio_full = _denoise_full_wav(gen_audio_full)
        voc_elapsed = time.perf_counter() - t_voc

        if chunk1_emitted and held_back_tail is not None and chunk1_audio_len is not None:
            # Crossfade at the boundary. Chunk 1 emitted audio up to
            # chunk1_audio_len - crossfade_samples; we held back the trailing
            # crossfade_samples. Blend those with the corresponding region of
            # chunk 2's output, then emit the rest of chunk 2.
            boundary = chunk1_audio_len  # sample offset inside gen_audio_full
            xfade_start = boundary - crossfade_samples
            xfade_end = boundary
            if xfade_start < 0 or xfade_end > len(gen_audio_full):
                # Shouldn't happen; fall back to plain emission of new tail only.
                logger.warning("Crossfade window out of range — emitting hard boundary")
                yield _emit_pcm(held_back_tail), S3GEN_SR
                yield _emit_pcm(gen_audio_full[boundary:]), S3GEN_SR
            else:
                head_b = gen_audio_full[xfade_start:xfade_end]
                blended = _linear_crossfade(held_back_tail, head_b)
                tail_rest = gen_audio_full[xfade_end:]
                yield _emit_pcm(blended), S3GEN_SR
                if len(tail_rest) > 0:
                    yield _emit_pcm(tail_rest), S3GEN_SR
        elif chunk1_emitted:
            # Streamed chunk 1 but had no held_back (chunk shorter than crossfade
            # window). Emit everything past chunk 1's end from chunk 2.
            if chunk1_audio_len is not None:
                tail_rest = gen_audio_full[chunk1_audio_len:]
                if len(tail_rest) > 0:
                    yield _emit_pcm(tail_rest), S3GEN_SR
        else:
            # Never streamed (utterance shorter than K1) — emit everything now.
            if t_first_emit is None:
                t_first_emit = time.perf_counter()
            yield _emit_pcm(gen_audio_full), S3GEN_SR

        total_elapsed = time.perf_counter() - t_gen_start
        ttf = (t_first_emit - t_gen_start) if t_first_emit else total_elapsed
        logger.info(
            f"Stream done: ttf={ttf*1000:.0f}ms, vocoder={voc_elapsed*1000:.0f}ms, "
            f"total={total_elapsed*1000:.0f}ms, chunks=2 (streamed={chunk1_emitted})"
        )

    except Exception as e:
        logger.error(f"ONNX v2 stream synthesis error: {e}", exc_info=True)
        return


def unload_model() -> bool:
    global MODEL_LOADED, model_device
    global _speech_encoder_session, _embed_tokens_session
    global _language_model_session, _cond_decoder_session, _lm_output_names
    global _tokenizer, _model_dir, _watermarker

    logger.info("Unloading ONNX v2 TTS model...")
    _speech_encoder_session = None
    _embed_tokens_session = None
    _language_model_session = None
    _cond_decoder_session = None
    _lm_output_names = []
    _tokenizer = None
    _watermarker = None
    MODEL_LOADED = False
    model_device = None
    gc.collect()
    return True


def reload_model() -> bool:
    unload_model()
    return load_model()
