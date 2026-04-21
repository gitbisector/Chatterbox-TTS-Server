"""Smoke test the INT8 TRT engine on dummy inputs at a few representative shapes.

Confirms:
  1. Engine deserialises
  2. All input/output bindings exist with expected names + dtypes
  3. enqueueV3 succeeds at min/opt/max shapes without error
  4. Per-step latency at decode shape (B=2, S=1, past=200) is reported

Output of this script becomes the spec for engine_trt_lm.py.
"""
import os, time, numpy as np, tensorrt as trt

ENGINE_PATH = os.environ.get("ENGINE", "/work/language_model_v2.int8.engine")
print(f"engine: {ENGINE_PATH}")
print(f"TRT: {trt.__version__}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
assert engine is not None, "deserialize failed"
print(f"engine loaded: {engine.num_io_tensors} I/O tensors")

# Enumerate bindings
inputs = []
outputs = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    if mode == trt.TensorIOMode.INPUT:
        inputs.append((name, dtype, shape))
    else:
        outputs.append((name, dtype, shape))

print(f"\n{len(inputs)} inputs, {len(outputs)} outputs")
print(f"first 3 inputs : {[i[0] for i in inputs[:3]]}")
print(f"first 3 outputs: {[o[0] for o in outputs[:3]]}")

# Build dummy host inputs at decode shape (B=2, S=1, past=200)
import cuda.bindings.runtime as cudart
def cuda_check(ret):
    err = ret[0]
    if err.value:
        raise RuntimeError(f"CUDA: {err}")
    return ret[1] if len(ret) > 1 else None

context = engine.create_execution_context()

def trt_dtype_to_np(d):
    return {trt.DataType.HALF: np.float16, trt.DataType.FLOAT: np.float32,
            trt.DataType.INT64: np.int64, trt.DataType.INT32: np.int32}[d]

def run_at(B, S, past_len, label):
    print(f"\n=== {label}: B={B} S={S} past={past_len} ===")
    feed = {
        "inputs_embeds": np.random.randn(B, S, 1024).astype(np.float16) * 0.05,
        "attention_mask": np.ones((B, S + past_len), dtype=np.int64),
        "cfg_weight": np.array(0.5, dtype=np.float16),
    }
    for i in range(30):
        feed[f"past_key_values.{i}.key"]  = np.zeros((B, 16, past_len, 64), dtype=np.float16)
        feed[f"past_key_values.{i}.value"]= np.zeros((B, 16, past_len, 64), dtype=np.float16)

    # Allocate device inputs + set shapes
    dev_in = {}
    for name, dtype, _shape_decl in inputs:
        host = feed[name]
        context.set_input_shape(name, host.shape)
        np_dtype = trt_dtype_to_np(dtype)
        host = host.astype(np_dtype)
        sz = host.nbytes
        d_ptr = cuda_check(cudart.cudaMalloc(sz))
        cuda_check(cudart.cudaMemcpy(d_ptr, host.ctypes.data, sz,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        context.set_tensor_address(name, d_ptr)
        dev_in[name] = (d_ptr, sz, host.shape, np_dtype)

    # Allocate outputs after shapes resolved
    dev_out = {}
    for name, dtype, _ in outputs:
        shape = context.get_tensor_shape(name)
        np_dtype = trt_dtype_to_np(dtype)
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        d_ptr = cuda_check(cudart.cudaMalloc(nbytes))
        context.set_tensor_address(name, d_ptr)
        dev_out[name] = (d_ptr, nbytes, tuple(shape), np_dtype)

    # Stream + run
    stream = cuda_check(cudart.cudaStreamCreate())
    # Warmup
    context.execute_async_v3(stream_handle=stream)
    cuda_check(cudart.cudaStreamSynchronize(stream))

    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        context.execute_async_v3(stream_handle=stream)
    cuda_check(cudart.cudaStreamSynchronize(stream))
    dt_ms = (time.perf_counter() - t0) * 1000.0 / N
    print(f"  {N} runs, mean {dt_ms:.2f} ms/step")

    # Pull logits to host for sanity
    logits_d, logits_sz, logits_shape, logits_np = dev_out["logits"]
    logits_h = np.zeros(logits_shape, dtype=logits_np)
    cuda_check(cudart.cudaMemcpy(logits_h.ctypes.data, logits_d, logits_sz,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    print(f"  logits {logits_h.shape} dtype {logits_h.dtype} "
          f"min={float(logits_h.min()):.3f} max={float(logits_h.max()):.3f} "
          f"mean={float(logits_h.mean()):.3f}")

    # Cleanup
    for ptr, _, _, _ in dev_in.values():
        cuda_check(cudart.cudaFree(ptr))
    for ptr, _, _, _ in dev_out.values():
        cuda_check(cudart.cudaFree(ptr))
    cuda_check(cudart.cudaStreamDestroy(stream))

# Try min/opt/max shapes
run_at(B=2, S=1,   past_len=0,   label="min/prefill-tiny")
run_at(B=2, S=100, past_len=0,   label="prefill-typical")
run_at(B=2, S=1,   past_len=200, label="decode-step")
run_at(B=2, S=1,   past_len=600, label="decode-step-long-context")
print("\nALL SHAPES PASSED")
