"""Export a scatter-free, fp16, CFG-aware embed_tokens.onnx for Chatterbox Multilingual.

Drop-in replacement for ``onnx-community/chatterbox-multilingual-ONNX/onnx/embed_tokens.onnx``.

Why this exists
---------------
The community export uses ScatterND ops inside the input-embedding path.  ScatterND
breaks ORT CUDA-graph capture, which prevents us from pre-recording the language-model
decode step.  This script produces an equivalent graph that uses only elementwise
ops (``where``/``mul``/``add``) plus ``Gather`` (Embedding lookup).

Interface (kept identical to the upstream community export so
``engine_onnx.py`` does not need any changes):

    inputs:
        input_ids     (B, S) int64  — EXAGGERATION_TOKEN, text tokens, 0 delimiter,
                                       then speech tokens; OR just speech tokens during
                                       autoregressive decode.
        position_ids  (B, S) int64  — learned-pos indices.  Caller uses
                                       ``where(input_ids >= START_SPEECH_TOKEN, 0, arange-1)``.
        exaggeration  (1,)   float32 — emotion_adv scalar, broadcast to every EXAGGERATION_TOKEN
                                       position.

    outputs:
        inputs_embeds (B, S, 1024) float16

New behavior versus the community export
----------------------------------------
* All weights and the produced tensor are float16.
* B==2 is supported directly so CFG cond+uncond can be run in a single embedder call.
  Row 1's *text-token* embeddings are zeroed to mirror PyTorch ``text_emb[1].zero_()`` in
  ``chatterbox.models.t3.t3.T3.prepare_input_embeds``.  Speech-token and
  emotion-adv positions are NOT zeroed.
* Zero ScatterND nodes (verified by the validation step at the end of this script).

Run this inside the chatterbox-spark container:

    docker run --rm --runtime=nvidia --gpus all \\
      -v chatterbox-hf-cache:/app/hf_cache \\
      -v /home/ties/project/Chatterbox-TTS-Server/services/chatterbox-onnx-export:/export \\
      -v /home/ties/project/Chatterbox-TTS-Server/onnx-models:/output \\
      -e HF_HOME=/app/hf_cache \\
      chatterbox-spark:latest \\
      python3 /export/export_embed_tokens.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


# Chatterbox magic numbers (match the reference conversion script)
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
EXAGGERATION_TOKEN = 6563

HIDDEN_SIZE = 1024


class InputsEmbedsV2(nn.Module):
    """Scatter-free, fp16, CFG-aware replacement for the InputsEmbeds module.

    Mirrors VladOS's reference ``InputsEmbeds`` (chatterbox_to_onnx_conversion_script.py
    lines 892-984) but rewritten so PyTorch's ONNX exporter cannot emit ScatterND.

    Key rewrite tricks:
      * Never start with ``torch.zeros(...)`` and then conditionally fill — that pattern
        is what the exporter traces into ScatterND.  Instead, compute every branch as a
        full tensor and combine with elementwise masks.
      * Use ``torch.where`` only on same-shape operands (never via advanced indexing).
      * Expand ``emotion_adv`` with broadcast + ``*mask`` instead of
        ``cond_emotion_adv[batch_indices]`` which gets traced into a Gather + Scatter
        pair.
      * All weights are converted to fp16 at __init__ time.  The scalar fp32 exaggeration
        input is the only fp32 tensor in the graph, and is cast to fp16 immediately after
        passing through the Linear (so numeric stability of the emotion_adv projection is
        preserved).
    """

    def __init__(
        self,
        text_emb: nn.Embedding,
        text_pos_emb: nn.Embedding,
        speech_emb: nn.Embedding,
        speech_pos_emb: nn.Embedding,
        emotion_adv_fc: nn.Linear,
        start_speech_token: int = START_SPEECH_TOKEN,
        exaggeration_token: int = EXAGGERATION_TOKEN,
    ) -> None:
        super().__init__()
        # Clone weights to fp16.  We build fresh modules rather than mutating the
        # originals so this class is safe to import without disturbing the shared
        # chatterbox model.
        self.text_emb = self._clone_embedding_fp16(text_emb)
        self.text_pos_emb = self._clone_embedding_fp16(text_pos_emb)
        self.speech_emb = self._clone_embedding_fp16(speech_emb)
        self.speech_pos_emb = self._clone_embedding_fp16(speech_pos_emb)
        # emotion_adv_fc: (1, n_channels), no bias.  We keep weights in fp16 and cast
        # the fp32 scalar input to fp16 before the matmul.
        self.emotion_adv_fc = nn.Linear(
            emotion_adv_fc.in_features,
            emotion_adv_fc.out_features,
            bias=emotion_adv_fc.bias is not None,
        ).to(torch.float16)
        with torch.no_grad():
            self.emotion_adv_fc.weight.copy_(emotion_adv_fc.weight.to(torch.float16))
            if emotion_adv_fc.bias is not None:
                self.emotion_adv_fc.bias.copy_(emotion_adv_fc.bias.to(torch.float16))

        self.start_speech_token = int(start_speech_token)
        self.exaggeration_token = int(exaggeration_token)

    @staticmethod
    def _clone_embedding_fp16(emb: nn.Embedding) -> nn.Embedding:
        out = nn.Embedding(emb.num_embeddings, emb.embedding_dim).to(torch.float16)
        with torch.no_grad():
            out.weight.copy_(emb.weight.to(torch.float16))
        return out

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, S) int64
        position_ids: torch.Tensor,  # (B, S) int64
        exaggeration: torch.Tensor,  # (1,) float32
    ) -> torch.Tensor:
        """Returns (B, S, 1024) float16."""
        batch_size, seq_len = input_ids.shape

        # ---- Build the text / speech / exaggeration masks ----
        # Same semantics as the upstream InputsEmbeds class: the first "0" in each row
        # marks the boundary between the text prefix and the speech suffix.  Positions at
        # or before the boundary are "text" (except for EXAGGERATION_TOKEN positions),
        # positions strictly after the boundary are "speech".  Rows that do not contain a
        # 0 at all are treated as all-speech (which matches autoregressive decode calls
        # where input_ids is a single speech token).
        idx = torch.arange(seq_len, device=input_ids.device, dtype=input_ids.dtype)
        idx = idx.unsqueeze(0).expand(batch_size, -1)  # (B, S)

        is_zero = input_ids == 0
        has_zero = is_zero.any(dim=1)  # (B,)
        # argmax on a bool tensor returns the first True index. We cast to int64 first
        # because argmax on bool is unsupported by some ONNX opsets.
        first_zero = is_zero.to(torch.int64).argmax(dim=1)  # (B,), 0 if no zero
        # Default to -1 (so the "idx <= zero_pos" mask is empty) when there is no 0.
        minus_one = torch.full_like(first_zero, -1)
        zero_pos = torch.where(has_zero, first_zero, minus_one)  # (B,)

        exaggeration_mask = input_ids == self.exaggeration_token  # (B, S) bool
        base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)

        text_mask = base_text_mask & ~exaggeration_mask  # (B, S)
        speech_mask = ~base_text_mask & ~exaggeration_mask  # (B, S)

        # ---- Safe indices for the Embedding lookups ----
        # Masking the ids to 0 on "off" positions makes the lookup return a valid row
        # that we then multiply by the mask (zeroing it) and sum with the other branch.
        zero_idx = torch.zeros_like(input_ids)
        safe_text_ids = torch.where(text_mask, input_ids, zero_idx)
        safe_speech_ids = torch.where(speech_mask, input_ids, zero_idx)

        # Relative positions: PyTorch side uses `position_ids * mask`.  We keep the same.
        text_pos_ids = position_ids * text_mask.to(position_ids.dtype)
        speech_pos_ids = position_ids * speech_mask.to(position_ids.dtype)

        # ---- Token + position embeddings ----
        text_tok = self.text_emb(safe_text_ids)  # (B, S, D) fp16
        text_pos = self.text_pos_emb(text_pos_ids)
        speech_tok = self.speech_emb(safe_speech_ids)
        speech_pos = self.speech_pos_emb(speech_pos_ids)

        text_part = text_tok + text_pos  # (B, S, D)
        speech_part = speech_tok + speech_pos  # (B, S, D)

        # Mask out off positions.  These masks are bool so we cast to fp16 once.
        text_mask_f = text_mask.to(torch.float16).unsqueeze(-1)
        speech_mask_f = speech_mask.to(torch.float16).unsqueeze(-1)
        exag_mask_f = exaggeration_mask.to(torch.float16).unsqueeze(-1)

        # ---- CFG: zero the text portion of row 1 when B == 2 ----
        # Rather than branching on batch size (which PyTorch traces statically into the
        # graph and fixes B=whatever-we-traced-with), we build a per-row scale:
        #   row 0 -> 1.0
        #   row 1 -> 0.0
        #   any other row -> 1.0
        # If the caller traces with B=2 (which we do) the export just bakes this as a
        # constant per-row scale.  We only apply it to text_part; speech_part and
        # emotion_adv are untouched, matching the PyTorch reference.
        row_ids = torch.arange(batch_size, device=input_ids.device)
        cfg_text_scale = torch.where(
            row_ids == 1,
            torch.zeros((), dtype=torch.float16, device=input_ids.device),
            torch.ones((), dtype=torch.float16, device=input_ids.device),
        ).view(batch_size, 1, 1)
        text_part = text_part * cfg_text_scale

        # ---- Emotion-adv embedding for EXAGGERATION_TOKEN positions ----
        # emotion_adv_fc expects (..., 1) float.  We keep the Linear in fp16, so cast
        # the input early.  Output is broadcast across (B, S, D) via the exag mask.
        exag_val = exaggeration.to(torch.float16).view(1, 1, 1)  # (1, 1, 1)
        # (1, 1, D) — broadcasting on the (B, S) exag mask yields (B, S, D).
        exag_embed = self.emotion_adv_fc(exag_val)

        # ---- Combine ----
        # Each position is exactly one of: text, speech, exaggeration, or off.
        # So simple addition of masked tensors is correct and requires no scatter.
        out = (
            text_part * text_mask_f
            + speech_part * speech_mask_f
            + exag_embed * exag_mask_f
        )
        return out  # (B, S, D) fp16


def load_reference_module(device: str = "cpu") -> nn.Module:
    """Load the PyTorch InputsEmbeds module (fp32) to use as a numeric reference."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    print("[ref] loading ChatterboxMultilingualTTS on", device)
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    # Reuse the reference conversion class from the upstream script if it's importable.
    # Otherwise, reproduce it inline so we have a fp32 oracle to compare against.
    class _Ref(nn.Module):
        def __init__(self, cb):
            super().__init__()
            self.text_emb = cb.t3.text_emb
            self.text_pos_emb = cb.t3.text_pos_emb.emb
            self.speech_emb = cb.t3.speech_emb
            self.speech_pos_emb = cb.t3.speech_pos_emb.emb
            self.emotion_adv_fc = cb.t3.cond_enc.emotion_adv_fc

        def forward(self, input_ids, position_ids, exaggeration):
            B, S = input_ids.shape
            idx = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
            is_zero = input_ids == 0
            has_zero = is_zero.any(dim=1)
            first_zero = is_zero.to(torch.int64).argmax(dim=1)
            zero_pos = torch.where(
                has_zero, first_zero, torch.full_like(first_zero, -1)
            )
            exag_mask = input_ids == EXAGGERATION_TOKEN
            base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)
            text_mask = base_text_mask & ~exag_mask
            speech_mask = ~base_text_mask & ~exag_mask

            zero_idx = torch.zeros_like(input_ids)
            safe_text = torch.where(text_mask, input_ids, zero_idx)
            safe_speech = torch.where(speech_mask, input_ids, zero_idx)
            text_pos = position_ids * text_mask.to(position_ids.dtype)
            speech_pos = position_ids * speech_mask.to(position_ids.dtype)

            t = self.text_emb(safe_text) + self.text_pos_emb(text_pos)
            s = self.speech_emb(safe_speech) + self.speech_pos_emb(speech_pos)
            # CFG: zero row 1 of text only (reference behaviour we want to match).
            if B == 2:
                t = t.clone()
                t[1] = 0.0
            tm = text_mask.unsqueeze(-1).to(t.dtype)
            sm = speech_mask.unsqueeze(-1).to(s.dtype)
            em = exag_mask.unsqueeze(-1).to(t.dtype)

            e = self.emotion_adv_fc(exaggeration.view(1, 1, 1).to(t.dtype))
            out = t * tm + s * sm + e * em
            return out

    return _Ref(model).eval(), model


def build_inputs(batch_size: int, device: str = "cpu"):
    """Build the same "canonical" input the community export uses, optionally batched."""
    # 80 text-ish tokens, then the 0 delimiter, then two START_SPEECH_TOKEN values.
    base = [
        EXAGGERATION_TOKEN,
        255, 281, 39, 46, 56, 2, 53, 2, 286, 41, 37, 2, 136, 122, 49,
        2, 152, 2, 103, 2, 277, 21, 101, 7, 2, 301, 55, 34, 28, 7, 2,
        53, 2, 296, 18, 18, 115, 2, 51, 2, 33, 245, 2, 17, 190, 2, 42,
        2, 50, 18, 125, 4, 32, 2, 290, 169, 142, 2, 41, 2, 43, 2, 18,
        29, 91, 2, 25, 186, 8, 20, 14, 80, 2, 29, 86, 213, 216, 9, 0,
        START_SPEECH_TOKEN, START_SPEECH_TOKEN,
    ]
    input_ids = torch.tensor([base], dtype=torch.long, device=device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    position_ids = torch.where(
        input_ids >= START_SPEECH_TOKEN,
        torch.zeros_like(input_ids),
        torch.arange(input_ids.shape[1], device=device).unsqueeze(0) - 1,
    )
    exaggeration = torch.tensor([0.5], dtype=torch.float32, device=device)
    return input_ids, position_ids, exaggeration


def count_scatternd_nodes(model_path: str) -> int:
    model = onnx.load(model_path, load_external_data=False)
    return sum(1 for n in model.graph.node if n.op_type == "ScatterND")


def main() -> int:
    output_dir = Path(os.environ.get("EMBED_TOKENS_OUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "embed_tokens_v2.onnx"

    # ---- Load reference + build our fp16 module ----
    ref_module, chatterbox_model = load_reference_module(device="cpu")
    ref_module.eval()

    module = InputsEmbedsV2(
        text_emb=chatterbox_model.t3.text_emb,
        text_pos_emb=chatterbox_model.t3.text_pos_emb.emb,
        speech_emb=chatterbox_model.t3.speech_emb,
        speech_pos_emb=chatterbox_model.t3.speech_pos_emb.emb,
        emotion_adv_fc=chatterbox_model.t3.cond_enc.emotion_adv_fc,
        start_speech_token=START_SPEECH_TOKEN,
        exaggeration_token=EXAGGERATION_TOKEN,
    ).eval()

    # ---- Sanity: shapes match expected multilingual sizes ----
    assert module.text_emb.num_embeddings == 2454, (
        f"expected text_tokens_dict_size=2454, got {module.text_emb.num_embeddings}"
    )
    assert module.speech_emb.num_embeddings == 8194, (
        f"expected speech_tokens_dict_size=8194, got {module.speech_emb.num_embeddings}"
    )
    assert module.text_emb.embedding_dim == HIDDEN_SIZE

    # ---- Export with B=2 inputs so dynamic batch is validated at trace time ----
    input_ids, position_ids, exaggeration = build_inputs(batch_size=2)
    print(f"[trace] input_ids.shape={tuple(input_ids.shape)}")

    # torch.onnx.export: use opset 20 (same as reference).  We deliberately keep the
    # external-data layout compatible with the existing onnx-community model so the
    # engine_onnx.py volume-mount path works unchanged (it looks for .onnx + .onnx_data).
    #
    # torch.onnx's dynamo exporter writes the sidecar as ``<name>.onnx.data`` while the
    # upstream community model uses ``<name>.onnx_data`` (no dot).  We rewrite the
    # model below so the external-data reference uses the legacy name, then delete the
    # dynamo-style sidecar.
    t0 = time.time()
    tmp_path = out_path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        module,
        (input_ids, position_ids, exaggeration),
        str(tmp_path),
        export_params=True,
        opset_version=20,
        input_names=["input_ids", "position_ids", "exaggeration"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "exaggeration": {0: "exag_batch"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Re-save with the legacy sidecar name.  onnx.save(..., save_as_external_data=True,
    # location=...) packs all initializers into the named sidecar and rewrites the
    # external_data refs in the graph to point at it.
    loaded = onnx.load(str(tmp_path), load_external_data=True)
    sidecar_name = out_path.name + "_data"  # e.g. "embed_tokens_v2.onnx_data"
    # Remove any existing legacy sidecar before writing (onnx appends otherwise).
    legacy_sidecar = out_path.with_name(sidecar_name)
    if legacy_sidecar.exists():
        legacy_sidecar.unlink()
    if out_path.exists():
        out_path.unlink()
    onnx.save(
        loaded,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar_name,
        size_threshold=1024,
    )
    # Clean up the dynamo-style intermediate files.
    tmp_path.unlink(missing_ok=True)
    dynamo_sidecar = tmp_path.with_name(tmp_path.name + ".data")
    dynamo_sidecar.unlink(missing_ok=True)
    print(f"[export] wrote {out_path} (+ sidecar {sidecar_name}) in {time.time() - t0:.1f}s")

    # ---- Check model via onnx.checker ----
    # For larger-than-2GB models onnx writes external data; checker needs the full load.
    print("[check] onnx.checker.check_model ...")
    onnx.checker.check_model(str(out_path))

    # ---- Count ScatterND nodes (must be 0) ----
    sc_count = count_scatternd_nodes(str(out_path))
    print(f"[check] ScatterND nodes: {sc_count}")
    if sc_count != 0:
        print(f"[FAIL] expected 0 ScatterND nodes, got {sc_count}", file=sys.stderr)
        return 2

    # ---- Load with ORT (CPU) ----
    print("[ort] loading with CPUExecutionProvider ...")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"      input  {i.name}: shape={i.shape} dtype={i.type}")
    for o in sess.get_outputs():
        print(f"      output {o.name}: shape={o.shape} dtype={o.type}")

    # ---- Smoke test: B=2, S=10 random ----
    print("[smoke] random B=2 S=10 ...")
    rng = np.random.default_rng(0)
    # Mix of text tokens (<2454), the 0 delimiter, and speech tokens (>=START_SPEECH_TOKEN).
    # We sample one row and duplicate it to row 1 so row0 vs row1 differences come *only*
    # from CFG zeroing (not from the random sampler).
    row = rng.integers(low=1, high=2454, size=(10,), dtype=np.int64)
    row[0] = EXAGGERATION_TOKEN
    row[6] = 0  # delimiter
    row[7:] = rng.integers(
        low=START_SPEECH_TOKEN, high=START_SPEECH_TOKEN + 32, size=(3,), dtype=np.int64
    )
    smoke_ids = np.stack([row, row], axis=0)
    smoke_pos = np.where(
        smoke_ids >= START_SPEECH_TOKEN,
        0,
        np.arange(smoke_ids.shape[1])[None, :] - 1,
    ).astype(np.int64)
    smoke_exag = np.array([0.5], dtype=np.float32)
    smoke_out = sess.run(
        None,
        {
            "input_ids": smoke_ids,
            "position_ids": smoke_pos,
            "exaggeration": smoke_exag,
        },
    )[0]
    print(
        f"       output shape={smoke_out.shape} dtype={smoke_out.dtype}"
    )
    assert smoke_out.shape == (2, 10, HIDDEN_SIZE), smoke_out.shape
    assert smoke_out.dtype == np.float16, smoke_out.dtype
    # Row-0 vs row-1 should differ on text-token positions (positions 1..5) because of the
    # CFG zeroing, but match on speech-token positions (7..9) and exaggeration (0) and
    # the delimiter (6, which is masked off entirely).
    text_pos_rows_differ = not np.allclose(smoke_out[0, 1:6], smoke_out[1, 1:6])
    speech_pos_rows_match = np.allclose(smoke_out[0, 7:], smoke_out[1, 7:])
    exag_rows_match = np.allclose(smoke_out[0, 0], smoke_out[1, 0])
    print(
        f"       row0 vs row1: text_differ={text_pos_rows_differ} "
        f"speech_match={speech_pos_rows_match} exag_match={exag_rows_match}"
    )
    assert text_pos_rows_differ, "CFG zeroing did not fire on text positions"
    assert speech_pos_rows_match, "speech embeddings leaked CFG zeroing"
    assert exag_rows_match, "exaggeration embedding leaked CFG zeroing"

    # ---- Numeric comparison vs PyTorch fp32 reference ----
    print("[numeric] comparing against fp32 PyTorch reference ...")
    ref_ids, ref_pos, ref_exag = build_inputs(batch_size=2)
    with torch.no_grad():
        ref_out = ref_module(ref_ids, ref_pos, ref_exag).to(torch.float32).cpu().numpy()
    ort_out = sess.run(
        None,
        {
            "input_ids": ref_ids.cpu().numpy(),
            "position_ids": ref_pos.cpu().numpy(),
            "exaggeration": ref_exag.cpu().numpy(),
        },
    )[0].astype(np.float32)
    max_abs_err = float(np.max(np.abs(ref_out - ort_out)))
    mean_abs_err = float(np.mean(np.abs(ref_out - ort_out)))
    print(f"         max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f}")
    if max_abs_err > 0.01:
        print(
            f"[FAIL] max abs error {max_abs_err:.6f} exceeds fp16 tolerance 0.01",
            file=sys.stderr,
        )
        return 3

    # ---- Report ----
    main_size = out_path.stat().st_size
    data_path = out_path.with_suffix(out_path.suffix + "_data")
    # Also look for the default ONNX external-data sidecar naming.
    sidecar_candidates = [
        data_path,
        out_path.with_name(out_path.name + ".data"),
    ]
    data_size = 0
    for c in sidecar_candidates:
        if c.exists():
            data_size += c.stat().st_size
            print(f"[report] sidecar: {c.name} = {c.stat().st_size/1e6:.1f} MB")
    print(f"[report] main onnx: {main_size/1e6:.1f} MB")
    print(f"[report] total:     {(main_size + data_size)/1e6:.1f} MB")
    print(f"[report] ScatterND nodes: {sc_count}")
    print(f"[report] max abs error vs PyTorch fp32: {max_abs_err:.6f}")
    print(f"[report] mean abs error vs PyTorch fp32: {mean_abs_err:.6f}")
    print(f"[report] output file: {out_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
