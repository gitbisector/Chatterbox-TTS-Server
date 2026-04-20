"""Drive modelopt.onnx.quantize(int8) on language_model_v2.onnx.

Same shape as fp8_quantize.py but with quantize_mode='int8'. INT8 weights
+ INT8 activations is a supported pattern on Blackwell (kSTRONGLY_TYPED
allows it; same-precision matmul has tactics).
"""
import os, sys, time, glob
import numpy as np

ONNX_IN = os.environ.get("LM_ONNX_IN", "/work/language_model_v2.onnx")
ONNX_OUT = os.environ.get("LM_ONNX_OUT", "/work/language_model_v2.int8.onnx")
CALIB_DIR = os.environ.get("CALIB_DIR", "/work/calib_samples")

print(f"input : {ONNX_IN}")
print(f"calib : {CALIB_DIR}")
print(f"output: {ONNX_OUT}")

sample_files = sorted(glob.glob(os.path.join(CALIB_DIR, "sample_*.npz")))
print(f"found {len(sample_files)} captured samples")

chosen = None
for f in sample_files:
    with np.load(f) as npz:
        if npz["past_key_values.0.key"].shape[2] == 0:
            chosen = {k: npz[k].copy() for k in npz.files}
            print(f"picked prefill from {os.path.basename(f)}, S={chosen['inputs_embeds'].shape[1]}")
            break
if chosen is None:
    print("no prefill samples — using first sample")
    with np.load(sample_files[0]) as npz:
        chosen = {k: npz[k].copy() for k in npz.files}

# Same calibration_shapes trick as fp8_quantize.py — declare batch=2 so
# n_iter=1 and the iteration sees both cond+uncond rows together (CFG
# combine math requires it). Skip cfg_weight (rank-0 scalar).
calib_shape_parts = []
for k, v in chosen.items():
    if v.ndim == 0:
        continue
    calib_shape_parts.append(f"{k}:" + "x".join(str(d) for d in v.shape))
calibration_shapes = ",".join(calib_shape_parts)
print("calibration_shapes:", calibration_shapes[:120], "...")

print("calibration dict keys:", len(chosen))
print("inputs_embeds shape:", chosen["inputs_embeds"].shape, "dtype:", chosen["inputs_embeds"].dtype)
print("cfg_weight shape:", chosen["cfg_weight"].shape, "dtype:", chosen["cfg_weight"].dtype)

# Same np.array_split monkey-patch for the rank-0 cfg_weight.
import numpy as _np
_orig_split = _np.array_split
def _split_safe(arr, n, axis=0):
    a = _np.asarray(arr)
    if a.ndim == 0:
        return [a for _ in range(n)]
    return _orig_split(arr, n, axis=axis)
_np.array_split = _split_safe

print("calling modelopt.onnx.quantize(int8)...", flush=True)
import modelopt.onnx.quantization as mq

t0 = time.perf_counter()
try:
    mq.quantize(
        onnx_path=ONNX_IN,
        quantize_mode="int8",
        calibration_data=chosen,
        calibration_method="max",
        calibration_eps=["trt"],
        calibration_shapes=calibration_shapes,
        op_types_to_quantize=["MatMul"],
        use_external_data_format=True,
        output_path=ONNX_OUT,
        log_level="INFO",
    )
    print(f"int8 quantize done in {time.perf_counter()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc(limit=10); sys.exit(1)

for suffix in ["", ".data", "_data"]:
    p = ONNX_OUT + suffix
    if os.path.exists(p):
        print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB")
