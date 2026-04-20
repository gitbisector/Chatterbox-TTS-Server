# FINDINGS — Phase 2d (LM quantization for Blackwell)

Captured 2026-04-20. Saving the state-of-knowledge before pivoting to INT8 so
we can return to the FP8 dead-end with full context if a newer modelopt /
TRT-LLM unlocks it.

## Goal

Replace the fp16 ONNX language model (one of the four graphs in our streaming
Chatterbox stack) with an FP8-weight TensorRT engine on DGX Spark (GB10,
sm_121), so the bandwidth-bound LM decode loop runs ~2× faster. The end
state is a `engine_trt_lm.py` runtime wrapper that drops into
`engine_onnx.py` behind a `tts_engine.lm_backend = ort|trt` config knob.

## What we shipped

### Toolchain — fully working
- `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` is multi-arch (linux/amd64 +
  linux/arm64); pulls cleanly with NGC auth (`docker login nvcr.io` with a
  free API key).
- TensorRT 10.14.1.48, TensorRT-LLM 1.2.1, nvidia-modelopt 0.37.0 all
  pre-installed and detected GB10 (compute_cap 12.1).
- onnxruntime is the one missing dep; `pip install onnxruntime` inside the
  container takes <30 s.

### Calibration capture — fully working
- `capture_calibration.py` monkey-patches `engine_onnx._run_lm`, runs a
  10-utterance EN+NL corpus, dumps every 10th call as a per-sample `.npz`.
- Deliberately captures all 60 KV inputs (30 layers × {key,value}) so a
  forward calibration pass has the right shape on every input.
- Saves as separate npz files (not pickle) to be **numpy-version safe** —
  chatterbox container has numpy 2.4, TRT-LLM container has 1.26, and pickle
  doesn't round-trip across that boundary.
- 2026-04-20 capture: 105 samples, 4.2 GB. Lives at
  `/tmp/calib_samples/` inside the chatterbox container.

### modelopt FP8 ONNX quantize — produces a valid graph
- `modelopt.onnx.quantization.quantize(quantize_mode='fp8', ...)` runs to
  completion in ~210 s. Output is `language_model_v2.fp8.onnx` (0.8 MB graph)
  + `.onnx_data` (490 MB — exactly half of fp16's 977 MB, as expected for
  per-channel 8-bit weight quantization).
- 271 quantized nodes out of 4256 total.

### Two non-obvious modelopt gotchas (already coded around in `fp8_quantize.py`)

1. **Scalar inputs trip `np.array_split`.** modelopt's
   `CalibrationDataProvider.__init__` does
   `np.array_split(arr, n_itr, axis=0)` on every input regardless of rank.
   Our `cfg_weight` is a 0-D scalar so the call raises
   `IndexError: tuple index out of range`. **Fix:** monkey-patch
   `np.array_split` to return n copies for `ndim==0` inputs (preserves the
   scalar instead of trying to slice it).

2. **n_iterations = batch_size / inferred_batch.** modelopt computes the
   number of calibration iterations as `data.shape[0] / inferred_input_shape[0]`.
   For our graph, inferred batch is symbolic (1 by default), and our sample
   has `inputs_embeds.shape[0] = 2` (cond + uncond). Without intervention
   modelopt does 2 iterations of batch=1, but the graph's CFG combine math
   requires both rows present — single-row forward fails. **Fix:** pass
   `calibration_shapes` declaring batch=2 explicitly so n_iter = 1 and the
   iteration sees both rows. Build the shape spec programmatically; skip
   the entry for cfg_weight since modelopt rejects rank mismatches there
   ("expects shape of rank 0, but calibration shape of rank 1 was passed").

## The wall — modelopt 0.37 emits FP8-act × INT8-weight, Blackwell can't fuse

After all the modelopt quirks were sorted, `trtexec --stronglyTyped` (the
required flag for mixed-precision builds on Blackwell+) reaches the
autotuner and dies:

```
[E] Error[9]: Skipping tactic ... Autotuner: no tactics to implement operation:
  fc: bias(f16[3072]) | activation(f8[1024])
                      , weight(i8[1024,3072])
                      , alpha(f32), beta(f32)
  // /v_proj/MatMul + /k_proj/MatMul + /q_proj/MatMul fusion: cask
[E] Error[10]: Could not find any implementation for node
  {ForeignNode[/Unsqueeze_2 + /Unsqueeze_3.../Add_212]}
```

The signature `f8 × i8` is a Hopper/Ada-era pattern. Blackwell tensor cores
expect **same-precision operands** for fused matmul (FP8×FP8 or INT8×INT8
or FP4×FP4). modelopt 0.37's `quantize_mode='fp8'` for ONNX produces
**FP8 activation Q/DQ + INT8 weight Q/DQ** — no knob in the public API to
push weights to FP8 too.

Tried `op_types_to_quantize=["MatMul"]` (limit blast radius to MatMul ops
only) — same error. The mixed-precision pattern is baked into modelopt's
fp8 mode at the ONNX level, not into which ops get touched.

## Why this is a real wall, not a workaround

The fundamental schema mismatch: `quantize_mode='fp8'` on the ONNX path
means "use FP8 Q/DQ on activations, use INT8 Q/DQ on weights". This was a
deliberate Hopper-era design (FP8 had better dynamic range for activations
where calibration is hard, INT8 was fine for static weights). Blackwell's
strict-typed fuser rejects this pattern.

The fix has to come from one of:
1. modelopt growing a `weight_dtype='fp8'` knob in the ONNX path.
2. Us bypassing modelopt — walk the ONNX graph, hand-quantize weight
   initializers to FP8E4M3FN + per-channel scale + DequantizeLinear node.
   Custom but bounded (~2 days).
3. Going via modelopt's *pytorch* path (`modelopt.torch.quantization`),
   which does emit pure FP8 weights, but expects a stock-architecture HF
   model. Our `LlamaForCFG` has CFG-in-graph and alignment-attention
   outputs that aren't standard.
4. **TRT-native PTQ from a fp16 ONNX** — bypass modelopt entirely. Feed
   the original fp16 `language_model_v2.onnx` into `trtexec --fp8 --calib`
   (or the equivalent `IBuilderConfig` API) and let TRT do the quantize +
   engine build in one shot. TRT's IInt8Calibrator interface is well-trod
   and on Blackwell has been extended for FP8/NVFP4. Avoids the modelopt
   ONNX-path-mix entirely. **Worth trying after INT8 proves the runtime
   wrapper end-to-end.** Expect the calibrator interface to need ~half a
   day of adaptation since it's C-level (read by TRT during build, hands
   batches via numpy).

## INT8 result (2026-04-20, post-pivot)

- modelopt `quantize_mode='int8'` produces a clean INT8-act × INT8-weight ONNX
  in 207 s. Output: `language_model_v2.int8.onnx` (2 MB graph) +
  `.onnx_data` (489 MB — half of fp16 977 MB, as expected).
- `trtexec --stronglyTyped` builds the engine in **38.1 s**, no errors.
  Engine size: 509 MB. `&&&& PASSED` from trtexec.
- One warning: `Profile kMAX values are not self-consistent` for `attention_mask`.
  My `--maxShapes` declared `attention_mask:2x800` but `inputs_embeds:2x500x1024`
  + `past_key_values:2x16x600x64` would imply `attention_mask` should be
  `2x1100`. Build still succeeded; tighten the shape spec for production.
- `trt_smoke.py` runs the engine at decode shapes:
  | shape | mean ms/step | logits sanity |
  |-------|--------------|---------------|
  | B=2, S=1, past=200 | **5.13 ms** | min=-3.03 max=3.73 mean=0.16 |
  | B=2, S=1, past=600 | **7.17 ms** | min=-3.55 max=3.66 mean=0.06 |
- vs the fp16 ORT baseline (10–12 ms/step), this is roughly **2×** faster on
  decode — exactly what halving weight bandwidth predicts.

The `min/prefill-tiny` and `prefill-typical` rows in trt_smoke.py reported
0.01 ms/step with all-zero logits — that's a smoke-test bug (output buffer
shape resolution before address binding), not an engine issue. The decode
runs are real and the logit ranges look healthy.

## Next concrete steps (post-INT8 success)

1. **Engine runtime wrapper** (`engine_trt_lm.py`): a class that mirrors
   `ort.InferenceSession.run()` so `engine_onnx._run_lm` can dispatch to
   either ORT or TRT based on a config flag. Holds the deserialised engine,
   reuses cudaMalloc'd device buffers across calls (no per-call alloc), and
   does `enqueueV3` + sync per step. ~1 day.
2. **engine_onnx integration**: `tts_engine.lm_backend = ort|trt` knob in
   `config.yaml`, `engine_onnx.load_model` picks the wrapper, the rest of
   the synthesize path is unchanged. ~half day.
3. **Gate measurement**: `tts_stt_streaming_roundtrip.py` on EN+NL. Goal:
   similarity within 2% of Phase-2c baseline, decode ms/step ≤ 7 (vs 10-12
   today).
4. **Upload `.engine`** to the HF model repo as a sidecar artifact. Note
   that engines are GPU-architecture-specific; a Spark-built engine won't
   run on Hopper or Ada. Document this clearly on the model card.

## Pivot for now: INT8 weights + INT8 activations

modelopt's `quantize_mode='int8'` produces a pure-INT8 graph that TRT can
fuse on Blackwell. The bandwidth win is the same as FP8 (1 byte per weight
either way); the speed win is smaller (Blackwell's INT8 tactics are
slightly slower than FP8 tactics, but both are real). Quality drop on a
500M-param decoder LM is typically <1% perplexity for INT8 PTQ — likely
within our 2% STT-similarity gate.

Recipe is identical to FP8 except for the `quantize_mode` arg, which is
why `fp8_quantize.py` is a single-line change away from being
`int8_quantize.py`. The TRT engine build via `--stronglyTyped` is the
same shape-spec gymnastics.

If INT8 produces a working engine end-to-end (build + load + decode),
that proves out the runtime wrapper / integration path entirely —
swapping in FP8 later (when modelopt or our own converter unblocks it)
is then just a different `.engine` file behind the same Python wrapper.

## State preserved on disk

- This directory has the scripts (`capture_calibration.py`, `fp8_quantize.py`,
  `build_trt.sh`) and these notes.
- Calibration npz set still inside the chatterbox container at
  `/tmp/calib_samples/` (4.2 GB, 105 samples).
- The failed FP8 ONNX is at `/tmp/fp8_test/language_model_v2.fp8.onnx`
  (host) — keep it; it's the proof that the modelopt path itself works,
  the 0-byte engine next to it is the proof that TRT can't consume it.
- Full plan + execution context lives at
  `~/.claude/plans/fluttering-meandering-tarjan.md`.

## Next session pickup

1. If you want to retry FP8 — check whether a newer TRT-LLM rc (1.3.0rc12+)
   added a `weight_dtype` knob to modelopt's ONNX path, or whether NVIDIA
   shipped a Blackwell-native `f8×i8` matmul tactic. Re-pull the container,
   re-run `fp8_quantize.py`, retry build. Everything in this directory is
   set up to be re-run as-is.
2. If you want to extend INT8 — produce variants for different bit widths
   (`quantize_mode='int4'` works the same way). INT4 weights gives 4×
   bandwidth reduction; quality budget needs a closer look.
3. If you want the hand-rolled FP8 weight conversion — start from the
   modelopt fp8 ONNX, walk the DequantizeLinear nodes whose weight
   initializer is INT8, replace with FP8 per-channel scaled tensors,
   adjust the schema. ~2 days estimate.
