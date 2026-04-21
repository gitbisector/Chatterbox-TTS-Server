"""Drive modelopt.onnx.quantize(fp8) on language_model_v2.onnx.

Passes a single real prefill sample as calibration_data. Weight-only FP8
PTQ is weight-dominated so one representative sample is enough for a first
pass; we can expand if the quality gate misses.
"""
import os, sys, time, glob
import numpy as np

ONNX_IN = os.environ.get("LM_ONNX_IN", "/work/language_model_v2.onnx")
ONNX_OUT = os.environ.get("LM_ONNX_OUT", "/work/language_model_v2.fp8.onnx")
CALIB_DIR = os.environ.get("CALIB_DIR", "/work/calib_samples")

print(f"input : {ONNX_IN}")
print(f"calib : {CALIB_DIR}")
print(f"output: {ONNX_OUT}")

sample_files = sorted(glob.glob(os.path.join(CALIB_DIR, "sample_*.npz")))
print(f"found {len(sample_files)} captured samples")

# Pick a prefill sample (one with past_len=0). Those have past_key_values.*
# shape (B, H, 0, D) — easy to identify.
chosen = None
for f in sample_files:
    with np.load(f) as npz:
        if npz["past_key_values.0.key"].shape[2] == 0:
            chosen = {k: npz[k].copy() for k in npz.files}
            print(f"picked prefill from {os.path.basename(f)}, S={chosen['inputs_embeds'].shape[1]}")
            break
if chosen is None:
    print("no prefill samples found — falling back to first sample")
    with np.load(sample_files[0]) as npz:
        chosen = {k: npz[k].copy() for k in npz.files}

# Tell modelopt the batch size is 2 (our sample's axis 0) so n_itr = 1
# instead of splitting cond/uncond across iterations (the graph's CFG combine
# math requires batch=2N for N candidates). cfg_weight stays scalar — shape
# "0" matches the rank-0 input declaration. We monkey-patch np.array_split
# below so modelopt can handle scalars.
calib_shape_parts = []
for k, v in chosen.items():
    if v.ndim == 0:
        # scalar
        continue  # let modelopt infer shape from the model
    else:
        calib_shape_parts.append(f"{k}:" + "x".join(str(d) for d in v.shape))
calibration_shapes = ",".join(calib_shape_parts)
print("calibration_shapes:", calibration_shapes[:120], "...")

calibration_data = chosen
print("calibration dict keys:", len(calibration_data))
print("inputs_embeds shape:", calibration_data["inputs_embeds"].shape, "dtype:", calibration_data["inputs_embeds"].dtype)
print("attention_mask shape:", calibration_data["attention_mask"].shape)
print("cfg_weight shape:", calibration_data["cfg_weight"].shape, "dtype:", calibration_data["cfg_weight"].dtype)

print("calling modelopt.onnx.quantize(fp8)...", flush=True)

# Monkey-patch np.array_split to handle 0-D arrays (our cfg_weight is scalar):
# just return a list of n copies so modelopt's per-iteration split works.
import numpy as _np
_orig_array_split = _np.array_split
def _array_split_safe(arr, n, axis=0):
    a = _np.asarray(arr)
    if a.ndim == 0:
        return [a for _ in range(n)]
    return _orig_array_split(arr, n, axis=axis)
_np.array_split = _array_split_safe

import modelopt.onnx.quantization as mq

t0 = time.perf_counter()
try:
    mq.quantize(
        onnx_path=ONNX_IN,
        quantize_mode="fp8",
        calibration_data=calibration_data,
        calibration_method="max",
        calibration_eps=["trt"],
        calibration_shapes=calibration_shapes,
        # MatMul-only quantization. Avoids the INT8 spillover modelopt does
        # for non-MatMul ops (Conv/Gemm/etc); TRT engine build requires a
        # single quantization type per fused op group on Blackwell.
        op_types_to_quantize=["MatMul"],
        use_external_data_format=True,
        output_path=ONNX_OUT,
        log_level="INFO",
    )
    print(f"quantize done in {time.perf_counter()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc(limit=10)
    sys.exit(1)

for suffix in ["", ".data", "_data"]:
    p = ONNX_OUT + suffix
    if os.path.exists(p):
        print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB")
