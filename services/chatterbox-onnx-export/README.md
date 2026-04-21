# Chatterbox Multilingual — Custom ONNX Export

Optimized ONNX export pipeline for Chatterbox Multilingual TTS, designed to beat
the PyTorch BF16 inference path on a DGX Spark (GB10, aarch64, CUDA 13, sm_121).

## Why we need this

The `onnx-community/chatterbox-multilingual-ONNX` community export has two
fundamental quality issues and one performance issue that prevent it from being
a viable replacement for the PyTorch path:

1. **No CFG (Classifier-Free Guidance)** — the community `embed_tokens` and
   `language_model` models run at batch=1, so we can't compute the
   `cond + cfg_weight * (cond - uncond)` combine that PyTorch uses for stability.
2. **No alignment-based EOS forcing** — the community export doesn't expose
   the layer-9/12/13 attention outputs that the PyTorch
   `AlignmentStreamAnalyzer` uses to detect when the model has finished
   speaking. Without this, the model rambles past the actual content into
   garbage (verified in our TTS→STT roundtrip test).
3. **ScatterND ops** — both `embed_tokens` and `conditional_decoder` use
   ScatterND, which prevents ORT from capturing the inference as a CUDA graph.
   No CUDA graph means each step pays per-kernel launch overhead.

## What this directory contains

| File | Output | Purpose |
|------|--------|---------|
| `export_t3_cfg.py` | `language_model_v2.onnx` | T3 LLaMA with CFG batching + layer-9/12/13 attention output + fp16 + scatter-free |
| `export_embed_tokens.py` | `embed_tokens_v2.onnx` | Scatter-free, batch=2 ready |
| `export_s3gen_decoder.py` | `conditional_decoder_n{4,6,10}.onnx` | Vocoder with parameterized CFM step count |
| `alignment_runtime.py` | (importable module) | Numpy port of `AlignmentStreamAnalyzer` |

The `speech_encoder.onnx` from the community export is reused as-is — it's only
called once per inference (encodes reference audio) and is not on the hot path.

## Inference graph layout (new)

```
text → embed_tokens_v2 → (B=2, S, 1024) ───┐
                                           ├→ language_model_v2 → logits (1, S, V), attn_layers (3, H, T_q, T_kv)
audio_ref → speech_encoder ────cond_emb───┘
                                           ↓
                              alignment_runtime + sampling
                                           ↓
                              speech_tokens
                                           ↓
                            conditional_decoder_n{4,6,10} → waveform
```

## Build all exports

```bash
docker run --rm --runtime=nvidia --gpus all \
  -v chatterbox-hf-cache:/app/hf_cache \
  -v $(pwd):/export \
  -v $(pwd)/../../onnx-models:/output \
  -e HF_HOME=/app/hf_cache \
  chatterbox-spark:latest \
  bash -c 'python3 /export/export_t3_cfg.py && python3 /export/export_embed_tokens.py && python3 /export/export_s3gen_decoder.py'
```

Output ONNX files land in `../../onnx-models/`.
