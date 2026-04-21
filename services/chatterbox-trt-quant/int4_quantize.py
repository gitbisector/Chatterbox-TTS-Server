"""Weight-only INT4 quantization of language_model_v2.onnx via ORT MatMulNBitsQuantizer.

Runs inside the chatterbox container (not the TRT toolchain container) — it
only needs the onnxruntime.quantization toolkit, which is bundled with ORT 1.24.

Recipe: bits=4, block_size=32, asymmetric. Exclude /speech_head/MatMul so the
final logit projection stays fp16 — preserves the ±12 logit range the sampler
needs (identified during the INT8 investigation).
"""
import os
import time

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    DefaultWeightOnlyQuantConfig,
)

ONNX_IN = os.environ.get("LM_ONNX_IN", "/app/onnx-models/language_model_v2.onnx")
ONNX_OUT = os.environ.get(
    "LM_ONNX_OUT", "/app/onnx-models/language_model_v2.int4.onnx"
)
BITS = int(os.environ.get("LM_BITS", "4"))
BLOCK_SIZE = int(os.environ.get("LM_BLOCK_SIZE", "32"))
IS_SYMMETRIC = os.environ.get("LM_SYMMETRIC", "0") == "1"

EXCLUDE = [
    n.strip()
    for n in os.environ.get("LM_EXCLUDE", "/speech_head/MatMul").split(",")
    if n.strip()
]

print(f"input      : {ONNX_IN}")
print(f"output     : {ONNX_OUT}")
print(f"bits       : {BITS}")
print(f"block_size : {BLOCK_SIZE}")
print(f"symmetric  : {IS_SYMMETRIC}")
print(f"exclude    : {EXCLUDE}")

t0 = time.time()
print("loading ONNX (with external data)...", flush=True)
m = onnx.load(ONNX_IN, load_external_data=True)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

algo = DefaultWeightOnlyQuantConfig(
    block_size=BLOCK_SIZE,
    is_symmetric=IS_SYMMETRIC,
    bits=BITS,
)
q = MatMulNBitsQuantizer(
    m,
    bits=BITS,
    block_size=BLOCK_SIZE,
    is_symmetric=IS_SYMMETRIC,
    nodes_to_exclude=EXCLUDE,
    algo_config=algo,
)

print("running MatMulNBitsQuantizer.process()...", flush=True)
t1 = time.time()
q.process()
print(f"  done in {time.time()-t1:.1f}s", flush=True)

out_dir = os.path.dirname(ONNX_OUT) or "."
os.makedirs(out_dir, exist_ok=True)
ext = os.path.basename(ONNX_OUT) + "_data"

# MatMulNBitsQuantizer wraps the model in ONNXModel — unwrap to ModelProto.
model_proto = q.model.model if hasattr(q.model, "model") else q.model

# Drop any existing external-data references so save_as_external_data rewrites
# cleanly; otherwise onnx.save may leave initializers pointing at the old file.
from onnx.external_data_helper import convert_model_to_external_data
for init in model_proto.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        init.ClearField("external_data")
        init.data_location = onnx.TensorProto.DEFAULT

print(f"saving to {ONNX_OUT} (external data: {ext})...", flush=True)
t2 = time.time()
convert_model_to_external_data(
    model_proto,
    all_tensors_to_one_file=True,
    location=ext,
    size_threshold=1024,
)
onnx.save_model(model_proto, ONNX_OUT)
print(f"  saved in {time.time()-t2:.1f}s")

import os.path as _p

print()
print(f"input size  : {_p.getsize(ONNX_IN) + _p.getsize(ONNX_IN + '_data'):,} bytes")
print(f"output size : {_p.getsize(ONNX_OUT) + _p.getsize(_p.join(out_dir, ext)):,} bytes")
print(f"total elapsed: {time.time()-t0:.1f}s")
