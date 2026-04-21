# File: engine_trt_lm.py
# TensorRT runtime wrapper for the INT8-quantized language_model_v2 engine.
#
# Exposes TRTLanguageModelSession with a .run(output_names, feed) surface that
# matches onnxruntime.InferenceSession for the three methods engine_onnx.py
# touches (get_inputs, get_outputs, run). This lets engine_onnx dispatch the
# LM forward pass to either ORT (fp16 ONNX) or TRT (INT8 .engine) behind a
# `tts_engine.lm_backend = ort|trt` config knob — no other code changes.
#
# The engine lives in a persistent IExecutionContext with one CUDA stream.
# Device buffers grow on demand: each run() checks whether the current
# allocation is large enough for the call's input/output bytes, and if not
# it cudaFree's + cudaMalloc's the bigger size and rebinds the tensor
# address. After a few requests the buffers hit their high-water mark and
# the hot path has no allocations — just set_input_shape + memcpy in +
# execute + memcpy out. Lazy growth (vs. a kMAX pre-allocation) is required
# here because the int8 engine's profile declares kMAX values that are not
# mutually self-consistent (see FINDINGS.md: attention_mask:2x800 vs
# inputs_embeds:2x500x1024 + past:2x16x600x64 → 1100), so probing every
# binding at its own kMAX to size outputs ahead of time would blow up the
# shape calculator.
#
# Measured against services/chatterbox-trt-quant/trt_smoke.py: 5–7 ms per
# decode step on DGX Spark (GB10) vs. 10–12 ms for fp16 ORT.
#
# Known residual cost: engine_onnx threads the KV cache through host memory
# (present.N.{k,v} output of step N is read out, then re-sent as
# past_key_values.N.{k,v} input of step N+1). That host-bounce is the next
# optimisation target after this wrapper stabilises — keeping KV on device
# requires also restructuring the caller, so it's deliberately out of scope
# here.

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# TRT + the CUDA Python bindings are only needed when the TRT backend is
# actually selected. Import lazily so `import engine_trt_lm` succeeds on
# hosts that don't have TensorRT installed (e.g. the ORT-only container).
try:
    import tensorrt as trt  # type: ignore
    import cuda.bindings.runtime as cudart  # type: ignore
    _TRT_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    trt = None  # type: ignore
    cudart = None  # type: ignore
    _TRT_IMPORT_ERROR = _e


def _require_trt() -> None:
    if _TRT_IMPORT_ERROR is not None:
        raise ImportError(
            "engine_trt_lm requires tensorrt and cuda-python "
            f"(import failed: {_TRT_IMPORT_ERROR!r})"
        )


def _trt_to_np_dtype(d) -> np.dtype:
    """Map a TRT DataType to the numpy dtype TRT uses for host buffers."""
    _require_trt()
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF:  np.float16,
        trt.DataType.INT8:  np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT64: np.int64,
        trt.DataType.BOOL:  np.bool_,
        trt.DataType.UINT8: np.uint8,
    }
    if d not in mapping:
        raise TypeError(f"Unsupported TRT dtype: {d}")
    return np.dtype(mapping[d])


def _np_to_ort_type_str(np_dtype: np.dtype) -> str:
    """Return the ORT-style type string ('tensor(float16)', etc.) for a numpy dtype.

    Only used so that get_inputs()/get_outputs() look like ORT NodeArg objects
    for any caller that introspects .type. engine_onnx.py itself only reads
    .name, but keep the surface honest for anyone tee-ing this into other
    diagnostics.
    """
    table = {
        "float32": "tensor(float)",
        "float16": "tensor(float16)",
        "int8":    "tensor(int8)",
        "int32":   "tensor(int32)",
        "int64":   "tensor(int64)",
        "bool":    "tensor(bool)",
        "uint8":   "tensor(uint8)",
    }
    return table.get(np_dtype.name, f"tensor({np_dtype.name})")


def _cuda_check(ret):
    """cuda.bindings.runtime returns (err, *values) tuples. Raise on error,
    return the bare value (or None) on success."""
    if not isinstance(ret, tuple):
        return ret
    err = ret[0]
    if getattr(err, "value", 0):
        raise RuntimeError(f"CUDA error: {err}")
    return ret[1] if len(ret) > 1 else None


@dataclass
class _Binding:
    """ORT NodeArg-compatible descriptor: only .name, .type, .shape are exposed."""
    name: str
    type: str
    shape: list  # str for symbolic dims (e.g. 'dim_0'), int for fixed dims


@dataclass
class _IOBuffer:
    name: str
    is_input: bool
    np_dtype: np.dtype
    engine_rank: int          # declared rank; caller shapes are coerced to match
    capacity_nbytes: int = 0  # currently-allocated device bytes
    d_ptr: int = 0            # current device pointer


class TRTLanguageModelSession:
    """Minimal onnxruntime.InferenceSession stand-in around one TRT engine.

    Thread-safety: TRT's IExecutionContext is not thread-safe, so every
    public method is serialised by an internal RLock. If callers ever need
    concurrent inference they should instantiate one session per worker
    thread (cheap: the heavy cost is engine deserialisation, not the
    context).
    """

    def __init__(self, engine_path: str, device_id: int = 0,
                 profile_index: int = 0) -> None:
        _require_trt()

        self._path = Path(engine_path)
        if not self._path.exists():
            raise FileNotFoundError(f"TRT engine not found: {self._path}")

        self._device_id = device_id
        self._profile_index = profile_index
        self._lock = threading.RLock()
        self._closed = False

        _cuda_check(cudart.cudaSetDevice(device_id))

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(self._path, "rb") as f:
            engine_blob = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_blob)
        if self._engine is None:
            raise RuntimeError(f"deserialize_cuda_engine failed: {self._path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("create_execution_context failed")

        self._stream = _cuda_check(cudart.cudaStreamCreate())

        self._buffers: Dict[str, _IOBuffer] = {}
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_descs: List[_Binding] = []
        self._output_descs: List[_Binding] = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            np_dtype = _trt_to_np_dtype(self._engine.get_tensor_dtype(name))
            is_input = (mode == trt.TensorIOMode.INPUT)

            declared = list(self._engine.get_tensor_shape(name))
            sym_shape: list = [
                int(d) if d >= 0 else f"dim_{j}" for j, d in enumerate(declared)
            ]

            self._buffers[name] = _IOBuffer(
                name=name, is_input=is_input, np_dtype=np_dtype,
                engine_rank=len(declared),
            )

            desc = _Binding(name, _np_to_ort_type_str(np_dtype), sym_shape)
            if is_input:
                self._input_names.append(name)
                self._input_descs.append(desc)
            else:
                self._output_names.append(name)
                self._output_descs.append(desc)

        logger.info(
            "TRT LM loaded: %s (%d inputs, %d outputs; buffers allocated lazily)",
            self._path.name, len(self._input_names), len(self._output_names),
        )

    # --- Internal helpers ---------------------------------------------

    def _coerce_shape_to_rank(self, arr: np.ndarray, rank: int, name: str) -> np.ndarray:
        """If the engine declares a higher rank than the caller's array, prepend
        size-1 dims. The common case is a rank-0 numpy scalar for a binding
        that the ONNX graph declared as shape (1,). Data bytes are identical.
        """
        if arr.ndim == rank:
            return arr
        if arr.ndim < rank:
            return arr.reshape((1,) * (rank - arr.ndim) + arr.shape)
        raise ValueError(
            f"input {name!r} has rank {arr.ndim} (shape {arr.shape}) "
            f"but engine expects rank {rank}"
        )

    def _ensure_capacity(self, buf: _IOBuffer, nbytes: int) -> None:
        """Grow a device buffer to at least ``nbytes``; no-op if already big
        enough. Rebinds the tensor address whenever the pointer changes.
        """
        if nbytes <= buf.capacity_nbytes and buf.d_ptr:
            return
        if buf.d_ptr:
            _cuda_check(cudart.cudaFree(buf.d_ptr))
            buf.d_ptr = 0
        grown = max(nbytes, buf.capacity_nbytes * 2, 1)
        buf.d_ptr = _cuda_check(cudart.cudaMalloc(grown))
        buf.capacity_nbytes = grown
        self._context.set_tensor_address(buf.name, buf.d_ptr)

    # --- ORT-compatible surface ----------------------------------------

    def get_inputs(self) -> List[_Binding]:
        return list(self._input_descs)

    def get_outputs(self) -> List[_Binding]:
        return list(self._output_descs)

    def run(self, output_names: Optional[Sequence[str]],
            feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Execute one forward pass.

        ``output_names`` may be None (return every output in binding order)
        or a sequence of output names. ``feed`` maps every input name to a
        numpy array whose shape fits inside the engine profile's min/max.
        Arrays are coerced to the binding's expected dtype if they differ.

        Returns a list of numpy arrays in the order requested — mirroring
        ``ort.InferenceSession.run``.
        """
        if self._closed:
            raise RuntimeError("TRTLanguageModelSession is closed")

        requested = (list(output_names) if output_names is not None
                     else list(self._output_names))
        unknown = [n for n in requested
                   if n not in self._buffers or self._buffers[n].is_input]
        if unknown:
            raise KeyError(f"unknown output(s) requested: {unknown}")

        missing = [n for n in self._input_names if n not in feed]
        if missing:
            raise KeyError(f"missing input(s) in feed: {missing}")

        with self._lock:
            # 1. Coerce, set input shapes, grow buffers, H2D-copy.
            for name in self._input_names:
                buf = self._buffers[name]
                # Use asarray, not ascontiguousarray — the latter promotes
                # rank-0 scalars to rank-1 (ndim>=1 guarantee), which would
                # mis-shape cfg_weight (declared rank 0 in the engine).
                arr = np.asarray(feed[name], dtype=buf.np_dtype)
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                arr = self._coerce_shape_to_rank(arr, buf.engine_rank, name)
                if not self._context.set_input_shape(name, arr.shape):
                    raise RuntimeError(
                        f"set_input_shape({name!r}, {arr.shape}) rejected — "
                        f"shape outside profile {self._profile_index}"
                    )
                self._ensure_capacity(buf, max(1, arr.nbytes))
                if arr.nbytes > 0:
                    _cuda_check(cudart.cudaMemcpyAsync(
                        buf.d_ptr, arr.ctypes.data, arr.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        self._stream,
                    ))

            # 2. Resolve output shapes and grow output buffers.
            out_shapes: Dict[str, Tuple[int, ...]] = {}
            for name in self._output_names:
                dims = self._context.get_tensor_shape(name)
                shape = tuple(int(d) for d in dims)
                out_shapes[name] = shape
                buf = self._buffers[name]
                nbytes = max(1, int(np.prod(shape) or 1)) * buf.np_dtype.itemsize
                self._ensure_capacity(buf, nbytes)

            # 3. Enqueue. execute_async_v3 returns False on binding errors.
            if not self._context.execute_async_v3(stream_handle=self._stream):
                raise RuntimeError("execute_async_v3 returned False")

            # 4. D2H-copy the requested outputs into fresh host arrays.
            host_out: Dict[str, np.ndarray] = {}
            for name in requested:
                buf = self._buffers[name]
                shape = out_shapes[name]
                arr = np.empty(shape, dtype=buf.np_dtype)
                if arr.nbytes > 0:
                    _cuda_check(cudart.cudaMemcpyAsync(
                        arr.ctypes.data, buf.d_ptr, arr.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self._stream,
                    ))
                host_out[name] = arr

            _cuda_check(cudart.cudaStreamSynchronize(self._stream))

        return [host_out[n] for n in requested]

    # --- Lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Release CUDA resources. Idempotent."""
        with self._lock:
            if self._closed:
                return
            for buf in self._buffers.values():
                if buf.d_ptr:
                    try:
                        _cuda_check(cudart.cudaFree(buf.d_ptr))
                    except Exception as e:
                        logger.warning("cudaFree(%s) failed: %s", buf.name, e)
                    buf.d_ptr = 0
                    buf.capacity_nbytes = 0
            if self._stream is not None:
                try:
                    _cuda_check(cudart.cudaStreamDestroy(self._stream))
                except Exception as e:
                    logger.warning("cudaStreamDestroy failed: %s", e)
                self._stream = None
            self._context = None
            self._engine = None
            self._runtime = None
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
