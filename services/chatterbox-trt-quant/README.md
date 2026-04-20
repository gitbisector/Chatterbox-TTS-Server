# Chatterbox LM quantization for Blackwell (TensorRT)

Toolchain + scripts for quantizing the Chatterbox `language_model_v2.onnx` to
narrow precision (FP8 / INT8 / NVFP4) and building TensorRT engines for
DGX Spark (NVIDIA GB10, sm_121). Phase 2d of the streaming-Chatterbox plan.

## Status (2026-04-20)

| Path | State | Notes |
|------|-------|-------|
| Calibration capture | ✅ working | `capture_calibration.py` hooks `engine_onnx._run_lm`, dumps per-step inputs as `.npz` |
| modelopt FP8 ONNX quantize | ✅ produces a graph | But it's FP8 activations + INT8 weights — see FINDINGS.md |
| TRT engine build (FP8 path) | ❌ blocked | No Blackwell tactics for the fp8-act × int8-weight matmul that modelopt 0.37 emits |
| modelopt INT8 ONNX quantize | ✅ working | 207 s, 489 MB weights (half of fp16) |
| TRT engine build (INT8 path) | ✅ **working** | 38 s, 509 MB engine, `&&&& PASSED` |
| TRT engine smoke test | ✅ **working** | `trt_smoke.py` runs decode at **5–7 ms/step** vs 10–12 ms fp16 ORT — **~2× speedup** |
| Runtime wrapper (`engine_trt_lm.py`) | ⏭ pending | ~1 day — see FINDINGS "Next concrete steps" |
| `engine_onnx` integration | ⏭ pending | ~half day — `tts_engine.lm_backend = ort\|trt` knob |
| Gate measurement | ⏭ pending | similarity ≤ 2% drop on EN+NL |

## Files

- `capture_calibration.py` — runs in the chatterbox container, monkey-patches
  `engine_onnx._run_lm` to save inputs per call, runs a 10-utterance corpus,
  emits one `sample_NNNN.npz` per captured forward pass.
- `fp8_quantize.py` — runs in the TRT-LLM container. Loads samples from disk,
  drives `modelopt.onnx.quantization.quantize` with calibration data and
  shape spec. Includes a monkey-patch around `np.array_split` to handle our
  scalar `cfg_weight` input (which trips modelopt's per-iteration splitter).
  *Currently produces a TRT-unbuildable mixed-precision graph on Blackwell —
  kept as the recipe for when modelopt grows a `weight_dtype='fp8'` knob.*
- `int8_quantize.py` — same shape as `fp8_quantize.py` but
  `quantize_mode='int8'`. Produces a clean INT8 ONNX that TRT will fuse.
- `build_trt.sh` / `build_trt_int8.sh` — runs `trtexec --stronglyTyped`
  against the quantized ONNX to build a `.engine`. Programmatically generates
  min/opt/max shape strings for the 60 KV-cache inputs.
- `trt_smoke.py` — loads the built engine, runs at a few representative
  shapes, prints per-step latency. Sanity check before wiring into the
  Python engine path.
- `FINDINGS.md` — detailed lessons learned, especially what works and what
  fails on Blackwell with modelopt 0.37 + TRT 10.14.

## How to reproduce

Both stages run inside Docker containers. Calibration capture uses the existing
`chatterbox-onnx:latest` image (has our custom-built ORT). Quantization +
engine build uses `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (NGC-pulled,
multi-arch — picks the linux/arm64 variant on Spark automatically; needs
`docker login nvcr.io` first with a free NGC API key).

```bash
# 1. Capture calibration data — chatterbox container must be running
docker exec -e CHATTERBOX_CAPTURE_CALIB=1 \
            -e CAPTURE_OUT=/tmp/calib_samples \
            -e DISABLE_WATERMARK=1 \
            chatterbox python3 /path/to/capture_calibration.py
# Pull the npz directory out:
mkdir -p /tmp/quant_work/calib_samples
docker cp chatterbox:/tmp/calib_samples/. /tmp/quant_work/calib_samples/

# 2. Stage the LM ONNX
cp ~/project/Chatterbox-TTS-Server/onnx-models/language_model_v2.onnx* /tmp/quant_work/

# 3. Quantize + build engine (in the TRT-LLM container)
docker run --rm --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /tmp/quant_work:/work \
  -v /path/to/this/dir:/scripts:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.1 bash -c '
    pip install --quiet onnxruntime
    python3 /scripts/fp8_quantize.py    # or int8 — toggle quantize_mode
    bash /scripts/build_trt.sh
  '
```

Outputs land in `/tmp/quant_work/`:
- `language_model_v2.fp8.onnx` (+ `.onnx_data`) — quantized ONNX
- `language_model_v2.fp8.engine` — TRT engine (target of the build)

## Calibration data

The npz set used in the 2026-04-20 run is in the chatterbox container at
`/tmp/calib_samples/` (105 files, ~4.2 GB). Drawn from a 10-utterance EN+NL
corpus (5 each, mix of short and long), greedy sampling, every 10th
`_run_lm` call. Each file has 63 numpy arrays — 3 model inputs
(`inputs_embeds`, `attention_mask`, `cfg_weight`) plus 30 layers × 2 KV-cache
tensors. Replace freely; the modelopt config picks one prefill sample
(past_len=0) for the actual calibration pass.
