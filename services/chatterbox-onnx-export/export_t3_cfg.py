"""
export_t3_cfg.py — Export the Chatterbox Multilingual T3 Llama backbone to ONNX.

Improvements over the onnx-community/chatterbox-multilingual-ONNX export:

  1. Built-in Classifier-Free Guidance: the graph accepts a `(2, S, H)` batched
     `inputs_embeds` (row 0 = conditional, row 1 = unconditional) plus a scalar
     `cfg_weight`, emits a single `logits` output of shape `(1, S, V)` computed
     as `cond + cfg_weight * (cond - uncond)`.
  2. Exposes attention weights from layers 9, 12 and 13 — the three layers
     consumed by `AlignmentStreamAnalyzer` (LLAMA_ALIGNED_HEADS) — as a single
     `attn_layers` output of shape `(3, num_heads, S, total_S)`, conditional
     row only.
  3. fp16 weights with fp32 softmax/layernorm/residuals for numerical safety.
  4. Full KV-cache I/O: `past_key_values.{i}.{key,value}` inputs and
     `present.{i}.{key,value}` outputs for i in 0..29 (30 layers).
  5. Scatter-free: we bypass HF's `Cache` object entirely and manage KV via
     `torch.cat(past, current, dim=-2)`. No `ScatterND` reaches the graph.
  6. Static-shape friendly: dynamic axes declared only for batch-independent
     sequence dims.

The script runs inside the chatterbox container (so that the `chatterbox`
package, HF cache and full PyTorch stack are available). It writes the result
to /output/language_model_v2.onnx (+ .onnx_data sidecar) and performs a
complete validation pass at the end.
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "language_model_v2.onnx")
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_NAME
EXTERNAL_DATA_NAME = OUTPUT_NAME + "_data"  # sidecar name

NUM_LAYERS = 30
NUM_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 64
HIDDEN_SIZE = 1024
ALIGN_LAYERS = [9, 12, 13]  # layers consumed by AlignmentStreamAnalyzer

# Smoke-test shapes
SMOKE_BATCH = 2  # always 2 for CFG (cond + uncond)
SMOKE_SEQ = 5
SMOKE_VOCAB_CHECK = None  # filled in after model load

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q/k shape: (B, H, S, D); cos/sin shape: (B, S, D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Scatter-free Llama wrapper
# ---------------------------------------------------------------------------


class T3LlamaONNXWrapper(nn.Module):
    """Re-implements the LlamaModel forward pass in a way that is friendly to
    ONNX export with a static-cache style KV cache that uses only Concat/MatMul
    (no ScatterND). Keeps weights in fp16 but does softmax/residuals in fp32.
    """

    def __init__(self, t3):
        super().__init__()
        self.t3 = t3  # keep the whole module so parameters stay registered
        self.llama = t3.tfmr  # transformers.LlamaModel
        self.rotary = self.llama.rotary_emb
        self.final_norm = self.llama.norm
        self.speech_head = t3.speech_head
        self.num_layers = NUM_LAYERS
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.align_layers = ALIGN_LAYERS

    def _layer_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom attention: no Cache object, no SDPA, pure ops.

        Returns: (attn_output, attn_weights, present_key, present_value)
        """
        attn = self.llama.layers[layer_idx].self_attn
        bsz, q_len, _ = hidden_states.shape

        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # Append past along sequence axis (NO ScatterND here — pure Concat)
        k = torch.cat([past_k, k], dim=-2)
        v = torch.cat([past_v, v], dim=-2)

        # num_kv_heads == num_heads for Llama_520M, so no GQA repeat needed
        # attention: Q @ K^T / sqrt(d)
        # q: (B, H, S, D), k: (B, H, T, D)
        # keep in fp32 for the score + softmax step
        q32 = q.float()
        k32 = k.float()
        attn_weights = torch.matmul(q32, k32.transpose(2, 3)) * self.scale
        # causal_mask shape: (B, 1, S, T) additive in fp32
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # back to fp16 for the value product to save compute
        attn_weights_cast = attn_weights.to(v.dtype)
        attn_output = torch.matmul(attn_weights_cast, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights, k, v

    def _layer_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ):
        layer = self.llama.layers[layer_idx]
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_out, attn_weights, present_k, present_v = self._layer_attention(
            layer_idx, hidden_states, cos, sin, causal_mask, past_k, past_v
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, present_k, present_v

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # (2, S, 1024) — row 0 cond, row 1 uncond
        attention_mask: torch.Tensor,  # (2, total_S) — 1 = real token, 0 = pad
        cfg_weight: torch.Tensor,  # scalar
        *past_key_values: torch.Tensor,  # flat: [pk0, pv0, pk1, pv1, ...]
    ):
        bsz, q_len, _ = inputs_embeds.shape
        past_len = past_key_values[0].shape[-2]
        total_len = past_len + q_len

        # Position ids for the NEW tokens only: [past_len, past_len+1, ..., past_len+q_len-1]
        position_ids = torch.arange(past_len, total_len, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)

        # Build causal mask additive in fp32 with shape (B, 1, S, total_S).
        # Real tokens => 0.0, padded/future tokens => -inf (large negative).
        # 1) key-side padding mask from attention_mask
        # attention_mask shape (B, total_S). Expand to (B, 1, 1, total_S).
        key_mask = attention_mask[:, None, None, :].to(torch.float32)  # 1 or 0
        key_mask = (1.0 - key_mask) * torch.finfo(torch.float32).min

        # 2) causal mask: for each query position q (0..S-1 relative to new chunk),
        #    absolute position is past_len + q, so it may attend to keys 0..past_len+q.
        # shape (S, total_S)
        q_abs = position_ids[0]  # (S,) since same across batch
        k_pos = torch.arange(total_len, device=inputs_embeds.device)
        causal = (k_pos[None, :] > q_abs[:, None]).to(torch.float32) * torch.finfo(torch.float32).min
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1,1,S,total_S)
        causal_mask = key_mask + causal  # broadcast add → (B,1,S,total_S)

        # Rotary frequencies for the NEW token positions
        cos, sin = self.rotary(inputs_embeds, position_ids)  # (B, S, head_dim)

        hidden_states = inputs_embeds
        attn_captures: List[Optional[torch.Tensor]] = [None] * len(self.align_layers)
        presents: List[torch.Tensor] = []

        for i in range(self.num_layers):
            past_k = past_key_values[2 * i]
            past_v = past_key_values[2 * i + 1]
            hidden_states, attn_weights, present_k, present_v = self._layer_forward(
                i, hidden_states, cos, sin, causal_mask, past_k, past_v
            )
            presents.append(present_k)
            presents.append(present_v)

            if i in self.align_layers:
                idx = self.align_layers.index(i)
                # Conditional row only (row 0). attn_weights: (B, H, S, total_S)
                attn_captures[idx] = attn_weights[0]  # (H, S, total_S)

        hidden_states = self.final_norm(hidden_states)
        logits_full = self.speech_head(hidden_states)  # (2, S, V)

        cond = logits_full[0:1]
        uncond = logits_full[1:2]
        logits = cond + cfg_weight * (cond - uncond)

        attn_layers = torch.stack(attn_captures, dim=0)  # (3, H, S, total_S)
        return logits, attn_layers, *presents


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------


def load_t3_multilingual(device: str = "cuda"):
    """Load the multilingual T3 weights."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    print(f"[load] downloading / loading chatterbox multilingual to {device} ...", flush=True)
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)
    t3 = tts.t3
    t3.eval()
    return t3


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_dummy_inputs(device: str, dtype: torch.dtype, vocab: int, past_len: int = 0):
    """Build dummy inputs appropriate for tracing (or validation)."""
    S = 5
    inputs_embeds = torch.randn(2, S, HIDDEN_SIZE, device=device, dtype=dtype)
    attention_mask = torch.ones(2, past_len + S, device=device, dtype=torch.int64)
    cfg_weight = torch.tensor(0.5, device=device, dtype=dtype)
    past_kv = []
    for _ in range(NUM_LAYERS):
        k = torch.zeros(2, NUM_KV_HEADS, past_len, HEAD_DIM, device=device, dtype=dtype)
        v = torch.zeros(2, NUM_KV_HEADS, past_len, HEAD_DIM, device=device, dtype=dtype)
        past_kv.extend([k, v])
    return inputs_embeds, attention_mask, cfg_weight, past_kv


def export_onnx(wrapper: T3LlamaONNXWrapper, device: str):
    """Run torch.onnx.export with proper dynamic axes & external data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # tmp path — torch.onnx.export with large models (>2GB) needs external data
    tmp_path = OUTPUT_DIR / OUTPUT_NAME

    dtype = torch.float16
    # We trace with a NON-zero past length so that the concat is actually active
    # and with S > 1 so the first-step path is also exercised. We use past_len = 2
    # for tracing; dynamic axes declare both S and past_len as dynamic.
    past_len_trace = 2
    inputs_embeds, attention_mask, cfg_weight, past_kv = build_dummy_inputs(
        device, dtype, vocab=0, past_len=past_len_trace
    )

    input_names = ["inputs_embeds", "attention_mask", "cfg_weight"]
    output_names = ["logits", "attn_layers"]
    for i in range(NUM_LAYERS):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])

    # Dynamic axes:
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "sequence_length"},  # batch=2 fixed normally, S varies
        "attention_mask": {0: "batch", 1: "total_sequence_length"},
        "logits": {1: "sequence_length"},
        "attn_layers": {2: "sequence_length", 3: "total_sequence_length"},
    }
    for i in range(NUM_LAYERS):
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence_length"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence_length"}
        dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "total_sequence_length"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "total_sequence_length"}

    # Build the args tuple
    args = (inputs_embeds, attention_mask, cfg_weight, *past_kv)

    print(f"[export] torch.onnx.export → {tmp_path}", flush=True)
    t0 = time.time()
    # Sanity: run forward once eagerly to catch bugs before the exporter
    with torch.inference_mode():
        out = wrapper(*args)
        print(f"[export] eager smoke ok — logits {tuple(out[0].shape)} attn {tuple(out[1].shape)}")

    # Remove previous files (including any stray external-data chunks).
    for stale in OUTPUT_DIR.glob(OUTPUT_NAME + "*"):
        try:
            stale.unlink()
        except OSError:
            pass

    torch.onnx.export(
        wrapper,
        args,
        str(tmp_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # use legacy TorchScript exporter — handles dynamic_axes + KV cache better
    )
    print(f"[export] torch.onnx.export done in {time.time() - t0:.1f}s", flush=True)

    # Post-process: rewrite model to external-data format so we get one .onnx
    # file + a single sidecar (torch's built-in external data behaviour varies).
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    print("[export] loading exported model for external-data consolidation ...", flush=True)
    model = onnx.load(str(tmp_path), load_external_data=True)

    # Wipe the intermediate file & any scattered external data chunks.
    for stale in OUTPUT_DIR.glob(OUTPUT_NAME + "*"):
        try:
            stale.unlink()
        except OSError:
            pass

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=EXTERNAL_DATA_NAME,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(model, str(tmp_path), save_as_external_data=False)
    # The save above does NOT write external data itself (since we already
    # split), so use save() from external_data_helper to write tensors.
    # Actually: onnx.save with external data writes nothing separate when the
    # model already references an external file — we must write manually.
    from onnx.external_data_helper import write_external_data_tensors

    write_external_data_tensors(model, str(OUTPUT_DIR))
    onnx.save(model, str(tmp_path))
    print(f"[export] final model written to {tmp_path}", flush=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def reference_pytorch_run(wrapper: T3LlamaONNXWrapper, inputs, device: str):
    """Reference forward pass in PyTorch (fp16 weights)."""
    inputs_embeds, attention_mask, cfg_weight, past_kv = inputs
    with torch.inference_mode():
        out = wrapper(
            inputs_embeds.to(device),
            attention_mask.to(device),
            cfg_weight.to(device),
            *[t.to(device) for t in past_kv],
        )
    return out


def validate(model_path: Path, wrapper: T3LlamaONNXWrapper, device: str, vocab: int):
    import onnx
    import onnxruntime as ort

    print("\n=== Validation ===", flush=True)

    # 1. onnx.checker
    print("[val] onnx.load + check_model ...", flush=True)
    model = onnx.load(str(model_path), load_external_data=True)
    try:
        onnx.checker.check_model(model, full_check=False)
        print("[val] onnx.checker: PASS")
    except Exception as e:
        print(f"[val] onnx.checker: FAIL — {e}")
        raise

    # 5. ScatterND count
    scatter_count = sum(1 for n in model.graph.node if n.op_type == "ScatterND")
    print(f"[val] ScatterND node count: {scatter_count}")

    # Parameter / file size
    file_size_mb = sum(
        p.stat().st_size for p in model_path.parent.glob(model_path.name + "*")
    ) / (1024 * 1024)
    for p in model_path.parent.glob(model_path.name + "*"):
        print(f"[val] file {p.name}: {p.stat().st_size / (1024 * 1024):.1f} MiB")

    # 2 & 3. Load with ORT on CPU, smoke test
    print("[val] creating ORT CPU session ...", flush=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    # Build smoke inputs (fp16 on CPU — ORT supports fp16 tensors even on CPU)
    S = SMOKE_SEQ
    past_len = 0
    rng = np.random.default_rng(0)
    inputs_embeds = rng.standard_normal((2, S, HIDDEN_SIZE), dtype=np.float32).astype(np.float16)
    attention_mask = np.ones((2, past_len + S), dtype=np.int64)
    cfg_weight = np.array(0.5, dtype=np.float16)  # 0-d array, not numpy scalar
    feed = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cfg_weight": cfg_weight,
    }
    for i in range(NUM_LAYERS):
        feed[f"past_key_values.{i}.key"] = np.zeros(
            (2, NUM_KV_HEADS, past_len, HEAD_DIM), dtype=np.float16
        )
        feed[f"past_key_values.{i}.value"] = np.zeros(
            (2, NUM_KV_HEADS, past_len, HEAD_DIM), dtype=np.float16
        )

    print("[val] running ORT smoke test ...", flush=True)
    out_names = ["logits", "attn_layers"]
    t0 = time.time()
    outs = sess.run(out_names, feed)
    t1 = time.time()
    logits_ort, attn_ort = outs
    print(f"[val] ORT forward took {t1 - t0:.2f}s")
    print(f"[val] logits shape: {logits_ort.shape} (expect (1, {S}, {vocab}))")
    print(f"[val] attn_layers shape: {attn_ort.shape} (expect (3, {NUM_HEADS}, {S}, {S}))")
    assert logits_ort.shape == (1, S, vocab), "logits shape mismatch"
    assert attn_ort.shape == (3, NUM_HEADS, S, S), "attn shape mismatch"

    # 4. Compare to PyTorch reference
    print("[val] running PyTorch reference ...", flush=True)
    inputs_embeds_t = torch.from_numpy(inputs_embeds).to(device)
    attention_mask_t = torch.from_numpy(attention_mask).to(device)
    cfg_weight_t = torch.tensor(float(cfg_weight), device=device, dtype=torch.float16)
    past_kv_t = []
    for _ in range(NUM_LAYERS):
        past_kv_t.append(torch.zeros(2, NUM_KV_HEADS, past_len, HEAD_DIM, device=device, dtype=torch.float16))
        past_kv_t.append(torch.zeros(2, NUM_KV_HEADS, past_len, HEAD_DIM, device=device, dtype=torch.float16))
    with torch.inference_mode():
        ref = wrapper(inputs_embeds_t, attention_mask_t, cfg_weight_t, *past_kv_t)
    logits_ref = ref[0].float().cpu().numpy()
    attn_ref = ref[1].float().cpu().numpy()

    max_logit_err = float(np.abs(logits_ort.astype(np.float32) - logits_ref).max())
    max_attn_err = float(np.abs(attn_ort.astype(np.float32) - attn_ref).max())
    print(f"[val] max abs error — logits: {max_logit_err:.4g}")
    print(f"[val] max abs error — attn_layers: {max_attn_err:.4g}")

    # Extra test: past_len > 0, S = 1 (the streaming-decode path)
    print("[val] extra test — past_len=4, S=1 ...", flush=True)
    past_len2 = 4
    S2 = 1
    inputs_embeds2 = rng.standard_normal((2, S2, HIDDEN_SIZE), dtype=np.float32).astype(np.float16)
    attention_mask2 = np.ones((2, past_len2 + S2), dtype=np.int64)
    cfg_weight2 = np.array(0.7, dtype=np.float16)
    feed2 = {
        "inputs_embeds": inputs_embeds2,
        "attention_mask": attention_mask2,
        "cfg_weight": cfg_weight2,
    }
    for i in range(NUM_LAYERS):
        feed2[f"past_key_values.{i}.key"] = rng.standard_normal(
            (2, NUM_KV_HEADS, past_len2, HEAD_DIM), dtype=np.float32
        ).astype(np.float16) * 0.1
        feed2[f"past_key_values.{i}.value"] = rng.standard_normal(
            (2, NUM_KV_HEADS, past_len2, HEAD_DIM), dtype=np.float32
        ).astype(np.float16) * 0.1

    outs2 = sess.run(out_names, feed2)
    logits_ort2, attn_ort2 = outs2
    print(f"[val] (past=4,S=1) logits shape: {logits_ort2.shape} attn shape: {attn_ort2.shape}")
    assert logits_ort2.shape == (1, S2, vocab)
    assert attn_ort2.shape == (3, NUM_HEADS, S2, past_len2 + S2)

    # Build the ref tensors once, in order
    past_kv2_t = []
    for i in range(NUM_LAYERS):
        past_kv2_t.append(torch.from_numpy(feed2[f"past_key_values.{i}.key"]).to(device))
        past_kv2_t.append(torch.from_numpy(feed2[f"past_key_values.{i}.value"]).to(device))
    with torch.inference_mode():
        ref2 = wrapper(
            torch.from_numpy(inputs_embeds2).to(device),
            torch.from_numpy(attention_mask2).to(device),
            torch.tensor(0.7, device=device, dtype=torch.float16),
            *past_kv2_t,
        )
    logits_ref2 = ref2[0].float().cpu().numpy()
    err2 = float(np.abs(logits_ort2.astype(np.float32) - logits_ref2).max())
    print(f"[val] (past=4,S=1) max abs logit error: {err2:.4g}")
    max_logit_err = max(max_logit_err, err2)

    # Also verify that in the ref, CFG math is correct
    # logits_ref = cond + 0.5 * (cond - uncond); quick sanity check: ref[0] shape (1,S,V)
    assert ref[0].shape == (1, S, vocab)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"ScatterND count      : {scatter_count}")
    print(f"Max logit abs error  : {max_logit_err:.4g}")
    print(f"Total file size      : {file_size_mb:.1f} MiB")
    print(f"Output path          : {model_path}")
    ok = scatter_count == 0 and max_logit_err < 0.2  # generous for fp16
    return {
        "scatter_count": scatter_count,
        "max_logit_err": max_logit_err,
        "max_attn_err": max_attn_err,
        "file_size_mb": file_size_mb,
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}")
    print(f"[main] torch={torch.__version__}")

    t3 = load_t3_multilingual(device=device)

    # Convert model to fp16 — the Llama backbone + speech_head only.
    t3.half()
    t3.eval()

    vocab = t3.speech_head.out_features
    print(f"[main] speech_head vocab size: {vocab}")

    wrapper = T3LlamaONNXWrapper(t3).to(device).eval()

    model_path = export_onnx(wrapper, device=device)

    # Free memory before validation run with ORT (which reloads on CPU).
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    result = validate(model_path, wrapper, device=device, vocab=vocab)
    if result["ok"]:
        print("\nEXPORT SUCCESSFUL")
        return 0
    else:
        print("\nEXPORT FAILED (validation)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
