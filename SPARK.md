# DGX Spark fork — what's different from upstream

This fork of `devnen/Chatterbox-TTS-Server` targets the **NVIDIA DGX Spark
(GB10, aarch64, Blackwell SM 12.1, CUDA 13)**.  It keeps upstream's
PyTorch/BF16 path intact (`chatterbox-pytorch` container, port 8006) as a
fallback and adds a second inference path that runs everything through
ONNX Runtime with CFG, exposed attention, and a rolling streaming
vocoder.

If you're coming from the upstream project, the **runtime code and REST
API are backward compatible**.  The changes are additive: new engine,
new config knobs, new Dockerfile, new endpoints alongside the old.

Everything Spark-specific lives in:

- `Dockerfile.spark-onnx` — ARM64 CUDA 13 multi-stage build (ORT from
  source + runtime image).
- `engine_onnx.py` — ONNX inference engine.
- `services/chatterbox-onnx-export/` — custom ONNX graph exports (see
  that directory's README for the graph-level changes vs upstream).
- `services/chatterbox-trt-quant/` — TRT INT8 quantization research
  (documented in `FINDINGS.md`).
- `services/whisper-patch/` — Spark-friendly ASR patches for
  `whisper-asr-webservice` (used as the selection-STT in best-of-N).

## Ship list (most recent first)

### Rolling N-chunk streaming + n=3 streaming (2026-04-20)

Replaced the two-chunk streaming design with a **rolling vocoder** that
fires every K tokens, eliminating the long-utterance buffer underrun
(chunk 2 used to arrive 3+ s after chunk 1 on 12+ s outputs, causing
audible pauses).

At the same time, landed **streaming best-of-N** via a batched
`B=2N` LM for the first `K1` tokens, scored via an STT service
(Parakeet-TDT recommended), then continued at `B=2` for the winner.
Output: **TTFA 1.2-1.4 s at n=3** on GB10 (vs the 3-4 s you'd get from
batch-then-stream), similarity ≥99% EN / ≥88% NL.

New knobs in `config.yaml`:

| Key | Default | What |
|-----|--------:|------|
| `streaming.rolling_chunk_tokens` | `0` | Rolling vocode cadence; 0 = use K1 |
| `streaming.first_chunk_budget_ms` | `800` | Latency target for first audio chunk |
| `streaming.crossfade_ms` | `30` | Crossfade width at each rolling boundary |

Per-request override via the existing `/tts/stream` body (plus
`n_candidates` to enable best-of-N streaming).

### Audio-quality fixes (2026-04-20)

- `onnx_cfm_steps` default **4 → 6**.  CFM-4 emitted chaotic near-Nyquist
  noise in transition regions (~8 hard clicks per 5.4 s EN utterance
  measured by sample-to-sample `|Δ|≥1.0`).  CFM-6 drops that to zero at
  ~50 % more per-vocode wall time.
- `denoise.enabled` **true → false**.  The rolling refactor fires RNNoise
  many times per utterance (fresh state each call); its ~3-tap
  resampler introduces click transients at 10 Hz frame boundaries that
  negate the cleanup gains.  Dockerfile still installs `pyrnnoise==0.4.3`
  so a future "denoise only the new region" rework is trivial to
  enable.
- New `_strip_trailing_repeat` eliminates the "syllable echo" at EOS
  when `alignment_runtime` force-stops on 2× token repeat — the
  repeated pair used to render into the final audio before the analyzer
  decided to stop.

### Phase 2d: TRT INT8 LM backend (2026-04-14, currently **disabled**)

A full TensorRT INT8 LM backend (`engine_trt_lm.py`), built via
`services/chatterbox-trt-quant/` with a TRT-LLM 1.2.1 container on
Blackwell.  Engine wrapper is validated (matches ORT logits at cos=1.0
on fp16, 4.7-10.8 ms/step on INT8).  Deployment blocked on
quantization-quality regression documented in
`services/chatterbox-trt-quant/FINDINGS.md` — the engine builds and runs
but the INT8 calibration reduces output fidelity past acceptable.  Flip
via `tts_engine.lm_backend: trt` in `config.yaml` if you want to
experiment.

### Phase 2b-c: Best-of-N batched LM + weight-only INT4 (2026-04-13)

Batched B=2N GPT decode enables best-of-N quality selection without
running N serial inferences.  At the same time, the LM weights are
loaded via `MatMulNBitsQuantizer`-produced INT4 (`language_model_v2.int4.onnx`),
cutting decoder wall time on DRAM-bandwidth-limited Spark.  Controlled
by `tts_engine.lm_onnx_path` in `config.yaml`.

### Phase 2a: RNNoise post-pass + alignment tightening (2026-04-10)

Originally added to clean up CFM-4 artefacts; now largely superseded by
CFM-6 (see above) and disabled by default.

### ONNX engine (2026-04)

Initial `engine_onnx.py` + custom ONNX exports (`language_model_v2.onnx`,
`embed_tokens_v2.onnx`, `conditional_decoder_n{4,6,10}.onnx`, plus an
in-process port of `AlignmentStreamAnalyzer`).  Delivers CFG +
exposed-attention EOS detection + fp16 weights + scatter-free graphs
that were missing from the community export.  Required for any of the
later phases to be possible.

## Quick start on Spark

```bash
cd ~/project/Chatterbox-TTS-Server
docker build -f Dockerfile.spark-onnx -t chatterbox-onnx:latest .
```

Image is ~9 GB (CUDA 13 runtime + ORT built for SM 12.1 + Python deps).
First build is ~30-60 min because of the ORT-from-source step; set up
`ort-wheels/` to cache the wheel if you rebuild often (see the comments
in `Dockerfile.spark-onnx`).

Run via the cluster services manager:

```bash
cd ~/project/containers
./services.sh start
./services.sh status
```

Or standalone from this repo:

```bash
cd ~/project/containers  # compose file is there, not here
docker compose -f docker-compose.services.yaml up -d chatterbox
```

Chatterbox listens on `http://localhost:8004`.  The `/tts/stream`
endpoint supports both the upstream Chatterbox schema (`voice_mode`,
`predefined_voice_id`, `split_text`, `chunk_size`) and the added
streaming parameters (`n_candidates`, `first_chunk_tokens_override`).
The OpenAI-compatible `/v1/audio/speech` endpoint works unchanged.

## Backward compatibility

- The upstream PyTorch BF16 path still works (`CHATTERBOX_ENGINE=pytorch`).
- The upstream REST API is unchanged; added fields are all optional.
- The upstream config sections (`generation_defaults`, `audio_output`,
  `ui_state`, etc.) are untouched.  `tts_engine.onnx_cfm_steps` default
  changed from 4 → 6 — set it back to 4 if the older behaviour is
  required.
- The upstream Web UI (`ui/`) still works.  The new streaming knobs
  don't have UI controls yet.

## Related containers in the same stack

- **xtts-spark** (sibling project, port 8030) — a newer alternative
  TTS service using [XTTS v2](https://huggingface.co/coqui/XTTS-v2)
  under a minimal FastAPI wrapper.  Cleaner audio and faster TTFA on
  short utterances; CPML (non-commercial) license on the model
  weights.  Lives at `~/project/xtts-spark` and tracks under
  `gitbisector/xtts-spark` on GitHub.  `services.sh` manages both.
- **whisper** (port 9001) — ASR for TTS→STT roundtrip tests and
  historical best-of-N selection scoring.
- **parakeet-nemo** (port 9002) — current selection STT for batched
  best-of-N, faster and more accurate than Whisper for short clips.
- **vLLM** — separate, managed by `sparkrun`.

## Future work

Tracked in the agent's project notes, paraphrased here:

1. Windowed (rather than growing-prefix) rolling vocoder to drop the
   2-3 s `max_gap` on 20+ s streaming outputs.
2. Move the streaming best-of-N scoring off the STT round-trip into a
   direct mel-feature speaker-similarity score (cheaper).
3. Revisit the TRT INT8 path once modelopt improves Blackwell INT8
   calibration, or try TRT-native PTQ bypassing modelopt entirely.
4. Re-enable denoise via a "denoise only the new region of each
   rolling vocode" path (keeps O(length) cost, no inter-chunk boundary
   clicks).
