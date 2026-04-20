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


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if not MODEL_LOADED:
        logger.error("ONNX v2 TTS model is not loaded.")
        return None, None

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
        attn_layers = out["attn_layers"]  # (3, 16, prefill_len, prefill_len) fp32
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
                attn_layers = out["attn_layers"]
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
