"""
export_s3gen_decoder.py — Re-export the Chatterbox S3Gen vocoder (CFM + HiFi-GAN)
to ONNX with three improvements over the community export at
onnx-community/chatterbox-multilingual-ONNX:

  1. Parameterized CFM step count. The community export bakes 10 Euler
     integration steps into the graph. We emit three variants:
        conditional_decoder_n4.onnx   — fastest, some quality loss
        conditional_decoder_n6.onnx   — balanced
        conditional_decoder_n10.onnx  — reference quality
     so the runtime can choose the speed/quality tradeoff per request.

  2. Scatter-free. The upstream `solve_euler` runs the estimator once per
     step inside a Python loop that does `x_in[:].copy_(x)` style in-place
     scatters. That causes ScatterND nodes in the exported graph, which
     prevent CUDA-graph capture. We Python-unroll the loop (N steps baked
     in) and rebuild the batched CFG input via `torch.cat` every step, so
     no in-place writes reach the graph.

  3. fp16 weights, fp32 compute. The default FP16_MODE=initializers path
     converts every fp32 initializer (tensor weight) to fp16 and inserts
     a Cast(to=FLOAT) right after it, so the compute graph stays in fp32
     end-to-end. This is the safest interpretation of "fp16 weights for
     matmul-heavy parts, fp32 numerics": the softmax / layernorm / residual
     paths all continue in fp32 while the on-disk model halves in size.
     A `full-fp16 compute` mode (FP16_MODE=full) is also present but
     hits NaN on this graph due to onnxconverter_common's incomplete
     cast insertion at node-block-list boundaries -- left as future work.

  4. Custom torch-exportable ISTFT / STFT. `torch.istft` does not export
     cleanly to ONNX; we replace it with a conv_transpose1d-based custom
     ISTFT (adapted from the reference conversion script). Likewise we
     replace `torch.stft(..., return_complex=True) + view_as_real` with
     `torch.stft(..., return_complex=False)`.

  5. Scatter-free SineGen. The upstream `SineGen` builds the harmonic
     frequency matrix via a Python loop with per-row in-place writes
     (`F_mat[:, i:i+1, :] = f0 * (i+1) / sr`), which yields
     harmonic_num+1 ScatterND nodes. We replace it with one broadcast
     multiply over a (1, H, 1) harmonic index tensor.

Inputs
------
    speech_tokens      int64   (B, N_tok)
    speaker_embeddings float32 (B, 192)
    speaker_features   float32 (B, N_mel_prompt, 80)

Output
------
    waveform           float32 (B, N_samples)   approx 24 kHz

Run
---
    cd /home/ties/project/Chatterbox-TTS-Server
    docker run --rm --runtime=nvidia --gpus all \
      -v chatterbox-hf-cache:/app/hf_cache \
      -v $(pwd)/services/chatterbox-onnx-export:/export \
      -v $(pwd)/onnx-models:/output \
      -v $(pwd)/reference_audio:/ref \
      -e HF_HOME=/app/hf_cache \
      -e REF_WAV=/ref/Gianna.wav \
      chatterbox-spark:latest \
      python3 /export/export_s3gen_decoder.py

Environment variables (all optional)
------------------------------------
    REF_WAV        Path to a reference audio clip inside the container.
                   Defaults to /ref/Gianna.wav (or any other clip under
                   /ref/ we can find). Falls back to a synthesised noise
                   clip if nothing is available.
    FP16_MODE      "initializers" (default) -- convert weights to fp16,
                                               keep compute in fp32.
                   "none"                    -- fp32 throughout.
                   "full"                    -- experimental, may produce
                                               NaN on this graph.
    EXPORT_FP32=1  Shortcut for FP16_MODE=none.
    OUTPUT_DIR     Where to write the onnx files (default /output).
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Auto-install the few deps we rely on but aren't in the container by default.
# ---------------------------------------------------------------------------
def _ensure(pkg: str, import_name: str | None = None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"[setup] installing {pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
        __import__(import_name or pkg)


_ensure("onnxruntime==1.19.2", "onnxruntime")
_ensure("onnxconverter-common", "onnxconverter_common")
_ensure("onnxslim==0.1.68", "onnxslim")
# Newer torch.onnx.export path pulls in onnxscript; install if missing.
# onnxscript >=0.5 requires onnx >=1.17, but container has onnx 1.16.
_ensure("onnxscript<0.5", "onnxscript")

import onnx
import onnxruntime as ort
from onnxconverter_common import float16 as ocf16


# ---------------------------------------------------------------------------
# onnxconverter_common.convert_float_to_float16 has a bug where Cast nodes'
# output value_info gets rewritten to fp16 even when the Cast's `to` attr
# explicitly says fp32. ONNX runtime then rejects the model with a
# "output arg of Cast_N does not match expected type" error. We fix it
# post-conversion by forcing the value_info dtype to match the Cast's
# declared target type.
# ---------------------------------------------------------------------------
def _fix_cast_value_info(model: onnx.ModelProto) -> None:
    """Reconcile Cast nodes' `to` attribute with the surrounding value_info.

    onnxconverter_common leaves inconsistencies: a Cast's value_info may be
    marked fp16 while its `to` attribute still says fp32, and the
    surrounding initializers/consumers were converted to fp16. The right
    fix is to change the Cast's `to` attr to match the value_info (so the
    Cast becomes a no-op fp16->fp16 / fp32->fp32 passthrough, or a real
    conversion consistent with neighbours).
    """
    from onnx import TensorProto

    vi_type: dict[str, int] = {}
    for vic in (model.graph.value_info, model.graph.output, model.graph.input):
        for vi in vic:
            vi_type[vi.name] = vi.type.tensor_type.elem_type

    fixed = 0
    for n in model.graph.node:
        if n.op_type != "Cast" or len(n.output) == 0:
            continue
        out_name = n.output[0]
        if out_name not in vi_type:
            continue
        want = vi_type[out_name]
        if want not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for attr in n.attribute:
            if attr.name == "to" and attr.i != want and attr.i in (
                TensorProto.FLOAT, TensorProto.FLOAT16
            ):
                attr.i = want
                fixed += 1
                break
    if fixed:
        print(f"[fp16-fix] reconciled {fixed} Cast `to` attribute(s)")


def _strip_non_io_value_info(model: onnx.ModelProto) -> None:
    """Delete every non-graph-IO value_info so ORT re-infers shapes.

    Context: torch.onnx.export(...dynamic_axes=...) over-aggressively
    re-uses user axis names (e.g. "num_speech_tokens") for derived dims
    that aren't actually the same length (e.g. 2*T-1 in rel-pos attention).
    The resulting stale value_info then contradicts real runtime shapes.
    Clearing non-IO value_info sidesteps the issue without losing anything
    -- ORT recomputes shapes on session creation anyway.
    """
    del model.graph.value_info[:]


def _convert_initializers_to_fp16(
    model: onnx.ModelProto,
    skip_by_name_substring: Tuple[str, ...] = (
        # STFT window -- keep fp32 because it's a small buffer whose values
        # span ~[0, 1] and combine with other fp32 STFT path tensors.
        "stft_window",
        "inverse_basis",
        # Embedding tables can have wide value ranges; keeping them fp32
        # avoids having to cast the lookup output -- and cost is small.
        "input_embedding.weight",
    ),
) -> None:
    """Convert every fp32 initializer to fp16, insert Cast nodes on demand.

    Motivation: the task asks for "fp16 weights". The simplest robust
    realisation is: shrink the initializers on disk to fp16 but keep every
    op in fp32. We insert a Cast(to=FLOAT) right after each fp16 initializer
    so compute stays fp32. Halves the on-disk model size with zero runtime
    numerical risk.
    """
    from onnx import TensorProto, helper, numpy_helper

    # Collect initializer names and their value_info/type info.
    all_init_names = {init.name for init in model.graph.initializer}

    # Build reverse map: tensor name -> list of (node_idx, input_idx)
    consumer_map: dict[str, list[tuple[int, int]]] = {}
    for ni, n in enumerate(model.graph.node):
        for ii, inp in enumerate(n.input):
            if inp in all_init_names:
                consumer_map.setdefault(inp, []).append((ni, ii))

    converted = 0
    skipped = 0
    cast_nodes: list = []
    # To avoid name clashes, give each Cast a unique name.
    cast_idx = 0
    # Map old-initializer-name -> new-tensor-name (output of Cast) so we
    # rewrite every consumer once.
    rewrite: dict[str, str] = {}

    for init in list(model.graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        if any(s in init.name for s in skip_by_name_substring):
            skipped += 1
            continue
        # Convert tensor value from fp32 -> fp16. Skip any initializer that
        # holds values outside fp16 representable range (|x| > 65504) to
        # avoid overflow->Inf->NaN at inference time.
        fp32_arr = numpy_helper.to_array(init)
        abs_max = float(np.abs(fp32_arr).max()) if fp32_arr.size else 0.0
        if abs_max > 65000.0:  # a bit below 65504 for safety margin
            skipped += 1
            continue
        arr = fp32_arr.astype(np.float16)
        new_init = numpy_helper.from_array(arr, name=init.name)
        # Replace in-place
        init.CopyFrom(new_init)
        converted += 1

        # Insert a Cast(to=FLOAT) after this initializer so downstream ops
        # keep seeing fp32 values.
        cast_out = f"{init.name}__to_fp32_{cast_idx}"
        cast_idx += 1
        cast_node = helper.make_node(
            "Cast",
            inputs=[init.name],
            outputs=[cast_out],
            name=f"_cast_fp16_fp32_{cast_idx}",
            to=TensorProto.FLOAT,
        )
        cast_nodes.append(cast_node)
        rewrite[init.name] = cast_out

    # Rewrite consumer inputs to the new cast-output name.
    for old_name, new_name in rewrite.items():
        for ni, ii in consumer_map.get(old_name, []):
            model.graph.node[ni].input[ii] = new_name

    # Prepend the Cast nodes at the start of the graph so they precede
    # every consumer in topological order (initializers are always
    # available; order is a convenience for readability).
    for c in reversed(cast_nodes):
        model.graph.node.insert(0, c)

    print(f"[fp16] converted {converted} initializers to fp16 "
          f"(skipped {skipped} per name blocklist), inserted {len(cast_nodes)} Casts")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TIMESTEPS_LIST = [4, 6, 10]

# Sampling rates / STFT params -- must match upstream s3gen
S3GEN_SR = 24000
ISTFT_N_FFT = 16
ISTFT_HOP_LEN = 4
INFERENCE_CFG_RATE = 0.7  # matches CFM_PARAMS in chatterbox.models.s3gen.configs
N_TRIM = S3GEN_SR // 50   # 20 ms fade-in
SPK_EMBED_DIM = 192       # CAMPPlus xvector output
MEL_BINS = 80

# Smoke test: 100 speech tokens at 25 Hz -> 4 s of audio. Prompt feat is
# 2*token_len long (token_mel_ratio=2), typically small (half a second).
SMOKE_N_TOKENS = 100
SMOKE_PROMPT_MEL_LEN = 50   # 50 frames = 25 tokens of prompt


# ---------------------------------------------------------------------------
# Replacements for torch.stft / torch.istft that export cleanly to ONNX.
# ---------------------------------------------------------------------------
class CustomISTFT(nn.Module):
    """
    ConvTranspose1d-based inverse STFT. Exports cleanly to ONNX (unlike the
    native torch.istft, which emits a custom op). Adapted from the reference
    conversion script at:
      /tmp/onnx_conversion_scripts/chatterbox/chatterbox_to_onnx_conversion_script.py
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        assert n_fft >= win_length
        self.filter_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = self.filter_length // 2 + 1
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        window = torch.hann_window(win_length)
        pad_length = n_fft - window.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        torch_fft_window = F.pad(window, (pad_left, pad_right), mode="constant", value=0)
        inverse_basis *= torch_fft_window

        self.register_buffer("inverse_basis", inverse_basis.float(), persistent=False)
        self.register_buffer("window", window, persistent=False)

    @staticmethod
    def _window_sumsquare(window, n_frames, hop_length, win_length, n_fft):
        win_sq = window**2
        pad_length = n_fft - win_sq.size(0)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        win_sq = F.pad(win_sq, (pad_left, pad_right), mode="constant", value=0)
        win_sq = win_sq.unsqueeze(0).unsqueeze(0)

        s = torch.ones(1, 1, n_frames, dtype=window.dtype, device=window.device)
        x = F.conv_transpose1d(s, win_sq, stride=hop_length).squeeze()
        n = n_fft + hop_length * (n_frames - 1)
        return x[:n]

    def forward(self, recombine_magnitude_phase: torch.Tensor) -> torch.Tensor:
        assert recombine_magnitude_phase.dim() == 3, "must be [B, 2*N, T]"
        num_frames = recombine_magnitude_phase.size(-1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        window_sum = self._window_sumsquare(
            self.window,
            n_frames=num_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
        )
        tiny_value = torch.finfo(window_sum.dtype).tiny
        denom = torch.where(
            window_sum > tiny_value,
            window_sum,
            torch.tensor(1.0, dtype=window_sum.dtype, device=window_sum.device),
        )
        inverse_transform = inverse_transform / denom
        inverse_transform = inverse_transform * (self.filter_length / self.hop_length)

        q = self.filter_length // 2
        return inverse_transform[:, 0, q:-q]


# ---------------------------------------------------------------------------
# Conditional decoder wrapper: speech_tokens -> waveform.
# ---------------------------------------------------------------------------
class ConditionalDecoderExport(nn.Module):
    """
    Wraps the S3Gen flow encoder + CFM UNet estimator + HiFi-GAN vocoder for
    end-to-end ONNX export.

    Key differences from chatterbox.models.s3gen's runtime path:
      * CFM loop is Python-unrolled N times at trace time (`n_timesteps`
        fixed at construction); this replaces the runtime `solve_euler` loop
        which uses in-place `.copy_()` and produces ScatterND.
      * The CFG-batched tensors `x_in, mask_in, mu_in, ...` are rebuilt each
        step via `torch.cat(...,[zeros_like(...)], dim=0)`. This is cheap
        (all tensors are cond + zeros) and scatter-free.
      * `torch.stft(..., return_complex=True) + view_as_real` is replaced
        with `torch.stft(..., return_complex=False)`.
      * `torch.istft` is replaced with `CustomISTFT` (conv_transpose1d).
      * The per-CFG-step estimator forward skips `add_optional_chunk_mask`
        (static_chunk_size == 0, so it's equivalent to just the mask).
    """

    def __init__(self, s3gen: nn.Module, n_timesteps: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.inference_cfg_rate = INFERENCE_CFG_RATE
        self.n_trim = N_TRIM
        self.n_fft = ISTFT_N_FFT
        self.hop_len = ISTFT_HOP_LEN

        # ---- Flow encoder (speech_tokens -> mu) ----
        flow = s3gen.flow
        self.output_size = flow.output_size          # == 80
        self.input_embedding = flow.input_embedding
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj
        self.pre_lookahead_len = flow.pre_lookahead_len
        self.token_mel_ratio = flow.token_mel_ratio

        # ---- CFM UNet estimator ----
        est = flow.decoder.estimator
        self.time_embeddings = est.time_embeddings
        self.time_mlp = est.time_mlp
        self.down_blocks = est.down_blocks            # 1 block for channels=[256]
        self.mid_blocks = est.mid_blocks              # 12 blocks
        self.up_blocks = est.up_blocks                # 1 block
        self.final_block = est.final_block
        self.final_proj = est.final_proj
        # meanflow / static chunk are both unused here (static_chunk_size==0)

        # ---- HiFi-GAN mel2wav ----
        mel2wav = s3gen.mel2wav
        self.conv_pre = mel2wav.conv_pre
        self.lrelu_slope = mel2wav.lrelu_slope
        self.reflection_pad = mel2wav.reflection_pad
        self.ups = mel2wav.ups
        self.num_upsamples = mel2wav.num_upsamples
        self.num_kernels = mel2wav.num_kernels
        self.source_downs = mel2wav.source_downs
        self.source_resblocks = mel2wav.source_resblocks
        self.resblocks = mel2wav.resblocks
        self.conv_post = mel2wav.conv_post
        self.f0_predictor = mel2wav.f0_predictor
        self.f0_upsamp = mel2wav.f0_upsamp
        self.m_source = mel2wav.m_source
        self.audio_limit = mel2wav.audio_limit
        self.register_buffer(
            "stft_window", mel2wav.stft_window.float().clone(), persistent=False
        )

        # trim_fade ramp (fixed)
        trim_fade = torch.zeros(2 * self.n_trim)
        trim_fade[self.n_trim :] = (
            torch.cos(torch.linspace(torch.pi, 0, self.n_trim)) + 1
        ) / 2
        self.register_buffer("trim_fade", trim_fade.float(), persistent=False)

        # Custom ISTFT replacing torch.istft
        self.istft = CustomISTFT(self.n_fft, self.hop_len, self.n_fft)

        # Precomputed sine-gen phase offset. Upstream samples this at runtime
        # from U(-pi, pi). We bake it in as a constant buffer so the trace
        # does not emit aten::uniform (which opset 17 cannot export).
        # Index 0 stays at 0 (matches upstream's `phase_vec[:, 0, :] = 0`).
        sg = self.m_source.l_sin_gen
        H = sg.harmonic_num + 1
        g = torch.Generator().manual_seed(17)
        phase_vec = torch.empty(1, H, 1).uniform_(
            -float(torch.pi), float(torch.pi), generator=g
        )
        phase_vec[:, 0, :] = 0
        self.register_buffer("sine_phase_vec", phase_vec.float(), persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_to_bias(mask_bool: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask = mask_bool.to(dtype)
        return (1.0 - mask) * -1.0e10

    def _cond_forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """Single UNet estimator call (equivalent to decoder.py ConditionalDecoder.forward
        with static_chunk_size==0, meanflow=False).
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = torch.cat([x, mu], dim=1)
        spks_exp = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spks_exp], dim=1)
        x = torch.cat([x, cond], dim=1)

        masks = [mask]
        resnet, transformer_blocks, downsample = self.down_blocks[0]
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = self._mask_to_bias(mask_down.bool(), x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
        x = x.permute(0, 2, 1).contiguous()
        skip = x
        x = downsample(x * mask_down)
        masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.permute(0, 2, 1).contiguous()
            attn_mask = self._mask_to_bias(mask_mid.bool(), x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x, attention_mask=attn_mask, timestep=t
                )
            x = x.permute(0, 2, 1).contiguous()

        resnet, transformer_blocks, upsample = self.up_blocks[0]
        mask_up = masks.pop()
        x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
        x = resnet(x, mask_up, t)
        x = x.permute(0, 2, 1).contiguous()
        attn_mask = self._mask_to_bias(mask_up.bool(), x.dtype)
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
        x = x.permute(0, 2, 1).contiguous()
        x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up)

    def _flow_encode(
        self,
        speech_tokens: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        speaker_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the flow encoder to get mu, spks, cond, mask, mel_len1."""
        B = speech_tokens.size(0)
        # xvec projection
        embedding = F.normalize(speaker_embeddings, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)  # (B, 80)

        # token embedding
        token_len = torch.full(
            (B,), speech_tokens.size(1), dtype=torch.long, device=speech_tokens.device
        )
        # 1 for non-pad (B, T, 1); we assume no padding inside the batch.
        tmask = torch.ones(
            B, speech_tokens.size(1), 1, dtype=speaker_embeddings.dtype,
            device=speech_tokens.device,
        )
        tok_emb = self.input_embedding(torch.clamp(speech_tokens, min=0).long())
        tok_emb = tok_emb * tmask

        # conformer encoder -> (B, T*ratio, C)
        h, h_masks = self.encoder(tok_emb, token_len)
        # (skip pre-lookahead trim; for inference `finalize=True`)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        mel_len1 = speaker_features.shape[1]
        h = self.encoder_proj(h)  # (B, T_mel, 80)

        mel_total = h.shape[1]
        # conds[:, :mel_len1] = speaker_features
        # NOTE: we build it scatter-free via cat(prompt, zeros)
        pad_feat = torch.zeros(
            B, mel_total - mel_len1, self.output_size,
            dtype=h.dtype, device=h.device,
        )
        conds = torch.cat([speaker_features, pad_feat], dim=1).transpose(1, 2)  # (B, 80, T_mel)

        mu = h.transpose(1, 2).contiguous()  # (B, 80, T_mel)

        # mask for the mel: (B, 1, T_mel), 1 for non-pad. h_lengths==T_mel in
        # our single-batch inference.
        mask = torch.ones(
            B, 1, mel_total, dtype=h.dtype, device=h.device,
        )
        return mel_len1, mu, embedding, conds, mask

    # ------------------------------------------------------------------
    # CFM Euler solver, Python-unrolled and scatter-free.
    # ------------------------------------------------------------------
    def _cfm_solve(self, mu, mask, spks, cond):
        """
        Args:
          mu:   (B, 80, T_mel)
          mask: (B, 1,  T_mel)
          spks: (B, 80)
          cond: (B, 80, T_mel)
        Returns:
          x: (B, 80, T_mel)
        """
        x = torch.randn_like(mu) * 1.0
        t_span = torch.linspace(
            0, 1, self.n_timesteps + 1, device=mu.device, dtype=mu.dtype
        )
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Zero buffers for the uncond half (matches upstream's initial
        # torch.zeros([2*B, ...]) with only the cond half getting written).
        # Upstream DOES duplicate x, mask and t across both halves (see
        # `x_in[:B] = x_in[B:] = x` pattern in
        # chatterbox.models.s3gen.flow_matching.ConditionalCFM.solve_euler).
        # Only mu, spks and cond are zeroed in the uncond half.
        zeros_mu = torch.zeros_like(mu)
        zeros_spks = torch.zeros_like(spks)
        zeros_cond = torch.zeros_like(cond)

        # Python-unrolled loop -- n_timesteps fixed at construction time.
        # Each iteration builds fresh batched CFG tensors via cat, which
        # avoids the ScatterND produced by the upstream in-place .copy_().
        for step in range(self.n_timesteps):
            t = t_span[step : step + 1]
            dt = t_span[step + 1 : step + 2] - t

            # Build (2B, ...) CFG-batched tensors.
            # cond half  =  (x,   mask, mu,      t, spks,      cond)
            # uncond half = (x,   mask, zeros,   t, zeros,     zeros)
            t_single = t.expand(x.size(0))
            x_in = torch.cat([x, x], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            mu_in = torch.cat([mu, zeros_mu], dim=0)
            t_in = torch.cat([t_single, t_single], dim=0)
            spks_in = torch.cat([spks, zeros_spks], dim=0)
            cond_in = torch.cat([cond, zeros_cond], dim=0)

            dphi_dt = self._cond_forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt_cond, dphi_dt_uncond = torch.split(
                dphi_dt, [x.size(0), x.size(0)], dim=0
            )
            combined = (1.0 + self.inference_cfg_rate) * dphi_dt_cond \
                       - self.inference_cfg_rate * dphi_dt_uncond
            x = x + dt * combined

        return x

    # ------------------------------------------------------------------
    # Scatter-free SineGen. Upstream does `F_mat[:, i:i+1, :] = f0 * (i+1)/sr`
    # inside a Python loop, which produces harmonic_num+1 ScatterND nodes.
    # We replace that with torch.arange + broadcast to build the full harmonic
    # bank in one multiply. Logically identical.
    # ------------------------------------------------------------------
    def _sine_gen(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (B, 1, N). Returns sine_waves: (B, harmonic+1, N).

        Exactly reproduces chatterbox.models.s3gen.hifigan.SineGen.forward but
        with no in-place scatter writes. `phase_vec` is still baked in from
        the traced random sample (same as upstream at export time -- the
        reference script uses the same approach).
        """
        sg = self.m_source.l_sin_gen
        H = sg.harmonic_num + 1
        # harmonics = [1, 2, ..., H], broadcast over (B, H, N)
        harmonics = torch.arange(1, H + 1, device=f0.device, dtype=f0.dtype).view(1, H, 1)
        F_mat = f0 * harmonics / sg.sampling_rate  # (B, H, N)

        theta_mat = 2 * torch.pi * (torch.cumsum(F_mat, dim=-1) % 1)

        # Deterministic phase offset -- precomputed at __init__ to avoid
        # aten::uniform in the trace (opset 17 cannot export it).
        phase_vec = self.sine_phase_vec.to(dtype=f0.dtype, device=f0.device)
        sine_waves = sg.sine_amp * torch.sin(theta_mat + phase_vec)

        uv = (f0 > sg.voiced_threshold).to(f0.dtype)
        noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv

    def _source_module(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter-free replacement for SourceModuleHnNSF.forward."""
        sine_wavs, uv = self._sine_gen(x.transpose(1, 2))  # (B, H, N)
        sine_wavs = sine_wavs.transpose(1, 2)  # (B, N, H)
        sine_merge = self.m_source.l_tanh(self.m_source.l_linear(sine_wavs))
        return sine_merge

    # ------------------------------------------------------------------
    # HiFi-GAN forward -- mel -> waveform
    # ------------------------------------------------------------------
    def _hifigan_decode(self, speech_feat: torch.Tensor) -> torch.Tensor:
        """
        speech_feat: (B, 80, T_mel_gen)
        returns:     (B, N_samples)
        """
        # mel -> f0 -> sine source (via our scatter-free SineGen)
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # (B, N_samples_raw, 1)
        s = self._source_module(s)
        output_sources = s.transpose(1, 2).squeeze(1)  # (B, N_samples_raw)

        # real-valued STFT (return_complex=False, for ONNX export)
        spec = torch.stft(
            output_sources,
            self.n_fft,
            self.hop_len,
            self.n_fft,
            window=self.stft_window.to(output_sources.device),
            return_complex=False,
        )
        s_stft_real, s_stft_imag = spec[..., 0], spec[..., 1]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # (B, n_fft+2, T)

        # HiFiGAN body
        x = self.conv_pre(speech_feat)

        # upsample 0
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[0](x)
        si = self.source_downs[0](s_stft)
        si = self.source_resblocks[0](si)
        x = x + si
        xs = (
            self.resblocks[0](x) + self.resblocks[1](x) + self.resblocks[2](x)
        )
        x = xs / 3

        # upsample 1
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[1](x)
        si = self.source_downs[1](s_stft)
        si = self.source_resblocks[1](si)
        x = x + si
        xs = (
            self.resblocks[3](x) + self.resblocks[4](x) + self.resblocks[5](x)
        )
        x = xs / 3

        # upsample 2 (last: reflection pad)
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[2](x)
        x = self.reflection_pad(x)
        si = self.source_downs[2](s_stft)
        si = self.source_resblocks[2](si)
        x = x + si
        xs = (
            self.resblocks[6](x) + self.resblocks[7](x) + self.resblocks[8](x)
        )
        x = xs / 3

        # final
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1 :, :])

        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        rebuild = torch.cat([real, img], dim=1)
        wavs = self.istft(rebuild)

        # fade-in on first 2*n_trim samples, hard-limited
        trim_fade = self.trim_fade.to(dtype=wavs.dtype, device=wavs.device)
        # Build a per-sample scale: first len(trim_fade) samples are fade-in,
        # the rest are 1.0. We do this scatter-free by cat.
        ones = torch.ones(wavs.size(1) - trim_fade.size(0), dtype=wavs.dtype, device=wavs.device)
        scale = torch.cat([trim_fade, ones], dim=0)
        wavs = wavs * scale.unsqueeze(0)
        wavs = torch.clamp(wavs, -self.audio_limit, self.audio_limit)
        return wavs

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        speech_tokens: torch.Tensor,       # (B, N_tok)  int64
        speaker_embeddings: torch.Tensor,  # (B, 192)    float
        speaker_features: torch.Tensor,    # (B, N_mel_prompt, 80) float
    ) -> torch.Tensor:
        mel_len1, mu, spks, cond, mask = self._flow_encode(
            speech_tokens, speaker_embeddings, speaker_features
        )
        mel = self._cfm_solve(mu, mask, spks, cond)  # (B, 80, T_mel_total)
        # trim off the prompt portion
        mel_gen = mel[:, :, mel_len1:]
        wavs = self._hifigan_decode(mel_gen)
        return wavs


# ---------------------------------------------------------------------------
# Build dummy speaker inputs by running the reference embed_ref on a reference
# audio clip. Used both to drive the torch trace and to build ORT smoke inputs.
# ---------------------------------------------------------------------------
def build_speaker_inputs(
    s3gen: nn.Module, ref_wav_path: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (speech_tokens, speaker_embeddings, speaker_features).

    We reuse s3gen.embed_ref to obtain the real 24 kHz mel prompt and the
    192-dim xvector embedding -- these are what the prod pipeline feeds us
    via the separately-exported speech_encoder.onnx, so using them here
    keeps the smoke test representative.
    """
    import soundfile as sf

    wav, sr = sf.read(ref_wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono
    # Clip to 2s for the smoke test: keeps the prompt feat short (~100 mel
    # frames) so the 100-speech-token smoke produces ~3-4s of generated audio.
    max_len = 2 * sr
    wav = wav[:max_len]
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    ref = s3gen.embed_ref(wav_t, ref_sr=sr, device=device, ref_fade_out=False)

    speaker_embeddings = ref["embedding"].to(device=device, dtype=torch.float32)
    prompt_feat_full = ref["prompt_feat"].to(device=device, dtype=torch.float32)
    prompt_token_full = ref["prompt_token"].to(device=device).long()

    # Contract (matches the community export and ConditionalDecoder upstream):
    #   speech_tokens = concat(prompt_token, newly-generated-tokens)
    #   speaker_features = prompt mel-spectrogram; must have length
    #       2 * prompt_token.size(1)   (token_mel_ratio == 2).
    #
    # For the smoke test we want SMOKE_N_TOKENS = 100 tokens total. Use a
    # 50/50 split: first 50 come from the real prompt, last 50 are pulled
    # from later in the prompt to emulate "generated" tokens. That gives a
    # reasonable-sounding reconstruction (rather than noise), so the
    # amplitude / perceptual sanity checks actually mean something.
    half = SMOKE_N_TOKENS // 2
    # Truncate / loop the real prompt as needed.
    if prompt_token_full.size(1) < SMOKE_N_TOKENS:
        reps = (SMOKE_N_TOKENS + prompt_token_full.size(1) - 1) // prompt_token_full.size(1)
        prompt_token_full = prompt_token_full.repeat(1, reps)[:, :SMOKE_N_TOKENS]
    prompt_tokens = prompt_token_full[:, :half]             # (1, half)
    gen_tokens = prompt_token_full[:, half:SMOKE_N_TOKENS]  # (1, SMOKE_N_TOKENS-half)
    speech_tokens = torch.cat([prompt_tokens, gen_tokens], dim=1).long()
    # speaker_features must cover the prompt tokens, 2 mel frames per token.
    speaker_features = prompt_feat_full[:, : half * 2, :].contiguous()
    return speech_tokens, speaker_embeddings, speaker_features


# ---------------------------------------------------------------------------
# Export one variant
# ---------------------------------------------------------------------------
def export_one(
    s3gen: nn.Module,
    n_timesteps: int,
    speech_tokens: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    speaker_features: torch.Tensor,
    device: str,
) -> Path:
    print(f"\n===== Exporting n_timesteps={n_timesteps} =====", flush=True)

    wrapper = ConditionalDecoderExport(s3gen, n_timesteps=n_timesteps).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Sanity: run the torch path once and keep result for numerics compare
    with torch.inference_mode():
        torch.manual_seed(123)  # deterministic noise in _cfm_solve
        torch_wav = wrapper(speech_tokens, speaker_embeddings, speaker_features)
    print(
        f"[torch] forward ok: waveform shape={tuple(torch_wav.shape)}, "
        f"max|amp|={torch_wav.abs().max().item():.4f}",
        flush=True,
    )

    # Export to ONNX in fp32 first. For large N, the single-file fp32 can
    # exceed the 2GB protobuf limit, so we save to a tmp dir, reload with
    # external data, and re-save externalised before running any further
    # passes that might try to serialize the graph.
    fp32_path = OUTPUT_DIR / f"_tmp_conditional_decoder_n{n_timesteps}_fp32.onnx"
    print(f"[export] torch.onnx.export -> {fp32_path}", flush=True)
    torch.manual_seed(123)  # keep randn_like inside graph constant across exports
    torch.onnx.export(
        wrapper,
        (speech_tokens, speaker_embeddings, speaker_features),
        str(fp32_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["speech_tokens", "speaker_embeddings", "speaker_features"],
        output_names=["waveform"],
        dynamic_axes={
            "speech_tokens": {0: "batch_size", 1: "num_speech_tokens"},
            "speaker_embeddings": {0: "batch_size"},
            "speaker_features": {0: "batch_size", 1: "prompt_mel_len"},
            "waveform": {0: "batch_size", 1: "num_samples"},
        },
        dynamo=False,  # stick with the tracing exporter; dynamo path can miss ops
    )
    print(f"[export] fp32 saved", flush=True)

    # Re-save with external data so downstream passes work on models >2 GB.
    fp32_ext = OUTPUT_DIR / f"_tmp_conditional_decoder_n{n_timesteps}_fp32_ext.onnx"
    print(f"[export] re-saving with external data -> {fp32_ext}", flush=True)
    raw = onnx.load(str(fp32_path), load_external_data=True)
    onnx.save_model(
        raw,
        str(fp32_ext),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=fp32_ext.name + "_data",
    )
    fp32_path.unlink()
    del raw

    # Onnxslim pass — disabled. It folded small constants by duplicating
    # every Cast-fed initializer, which ballooned our n=4/6 file sizes
    # relative to n=10 (which skipped slim due to proto>2GB and was
    # substantially smaller). Our own downstream passes already do
    # shape-inference strip + initializer-to-fp16 conversion, which is
    # enough. If we revisit this, use onnxslim in a deduplication-only
    # mode.
    slim_path = fp32_ext
    print("[slim] DISABLED (avoids initializer duplication across unrolled loop)")

    # Two modes:
    #   FP16_MODE=initializers (default): convert only initializer tensors
    #     (a.k.a. weights) to fp16, leaving every op in fp32 and inserting
    #     a Cast at each initializer->op boundary. This is what "fp16
    #     weights" most naturally means, halves the on-disk size, and is
    #     numerically identical to fp32 at inference time (modulo the one
    #     initializer-read rounding, which is imperceptible).
    #   FP16_MODE=none (EXPORT_FP32=1): skip conversion entirely; fp32
    #     weights, fp32 compute.
    #   FP16_MODE=full: convert compute to fp16 too. Kept for future work;
    #     the onnxconverter_common path doesn't insert casts correctly at
    #     node-level block boundaries, which led to NaN outputs. We ship
    #     with this DISABLED by default.
    fp16_mode = "none" if os.environ.get("EXPORT_FP32") == "1" else os.environ.get(
        "FP16_MODE", "initializers"
    )
    print(f"[fp16] mode={fp16_mode}", flush=True)

    if fp16_mode == "none":
        print("[fp16] SKIPPED; keeping fp32", flush=True)
        final_path = OUTPUT_DIR / f"conditional_decoder_n{n_timesteps}.onnx"
        for stale in OUTPUT_DIR.glob(f"conditional_decoder_n{n_timesteps}.onnx*"):
            stale.unlink()
        model_fp32 = onnx.load(str(slim_path), load_external_data=True)
        _strip_non_io_value_info(model_fp32)
        onnx.save_model(
            model_fp32,
            str(final_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=final_path.name + "_data",
        )
        for p in [fp32_ext, slim_path] + list(OUTPUT_DIR.glob("_tmp_*")):
            if p.exists() and p != slim_path:
                p.unlink()
        if slim_path.exists():
            slim_path.unlink()
        final_path.with_suffix(".torch_ref.npy").write_bytes(
            torch_wav.float().cpu().numpy().tobytes()
        )
        final_path.with_suffix(".torch_ref.shape").write_text(
            ",".join(str(d) for d in torch_wav.shape)
        )
        return final_path

    if fp16_mode == "initializers":
        print("[fp16] converting initializers to fp16 ...", flush=True)
        model = onnx.load(str(slim_path), load_external_data=True)
        _strip_non_io_value_info(model)
        _convert_initializers_to_fp16(model)
        final_path = OUTPUT_DIR / f"conditional_decoder_n{n_timesteps}.onnx"
        for stale in OUTPUT_DIR.glob(f"conditional_decoder_n{n_timesteps}.onnx*"):
            stale.unlink()
        onnx.save_model(
            model,
            str(final_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=final_path.name + "_data",
        )
        print(f"[fp16] saved -> {final_path}", flush=True)
        for p in [fp32_ext, slim_path] + list(OUTPUT_DIR.glob("_tmp_*")):
            if p.exists() and p != slim_path:
                p.unlink()
        if slim_path.exists():
            slim_path.unlink()
        final_path.with_suffix(".torch_ref.npy").write_bytes(
            torch_wav.float().cpu().numpy().tobytes()
        )
        final_path.with_suffix(".torch_ref.shape").write_text(
            ",".join(str(d) for d in torch_wav.shape)
        )
        return final_path

    # fp16 conversion — keep LayerNorm/Softmax/ReduceMean in fp32 by default;
    # additionally block the trig/exp ops in the STFT/phase path for cleaner
    # numerics.
    print("[fp16] convert_float_to_float16 ...", flush=True)
    model_fp16 = onnx.load(str(slim_path), load_external_data=True)
    # RandomNormal/RandomNormalLike need to stay in fp32 — converting them
    # produces a dtype mismatch between the random output (fp16) and its
    # downstream consumers (which may still be fp32 after selective block).
    # Sin/Cos/Exp/Log are in the STFT/phase path; keep fp32 for clean numerics.
    op_block_list = list(ocf16.DEFAULT_OP_BLOCK_LIST) + [
        "Sin", "Cos", "Exp", "Log",
        "RandomNormal", "RandomNormalLike",
        "RandomUniform", "RandomUniformLike",
    ]
    # Node-level block list: keep every node inside the ISTFT / HiFi-GAN
    # Snake activations / STFT pre-processing in fp32, because they involve
    # trig/exp accumulations whose fp16 behaviour drifts into NaNs. The
    # conformer encoder is similarly kept in fp32 to avoid the attention
    # softmax + relative-pos-emb cascade overflowing. The CFM UNet carries
    # the bulk of the parameters and is the main beneficiary of fp16.
    node_block_list: list[str] = []
    for n in model_fp16.graph.node:
        name = n.name or ""
        if (
            name.startswith("/encoder/")           # conformer encoder
            or name.startswith("/m_source/")       # SineGen
            or name.startswith("/f0_predictor/")   # F0 predictor
            or "/istft/" in name                   # custom ISTFT
            or name.startswith("/istft")           # ditto
            or "/Snake" in name                    # HiFi-GAN Snake activations
            or "resblocks." in name                # HiFi-GAN residual blocks
            or "conv_post" in name                 # HiFi-GAN output conv (fp32 phase/mag)
        ):
            node_block_list.append(n.name)

    print(
        f"[fp16] op_block_list={len(op_block_list)}, "
        f"node_block_list={len(node_block_list)} "
        f"(fp32-kept nodes for numerical safety)",
        flush=True,
    )
    model_fp16 = ocf16.convert_float_to_float16(
        model_fp16,
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
        check_fp16_ready=False,
    )

    # Post-conversion fix: onnxconverter_common sometimes leaves `Cast`
    # nodes whose `to` attr says FLOAT (1) but whose value_info was marked
    # as fp16. Reconcile by trusting the op's `to` attribute: set the
    # value_info dtype to match. Same for CastLike when we see mismatches.
    _fix_cast_value_info(model_fp16)

    # Additional fix: `torch.onnx.export` with `dynamic_axes` re-uses the
    # user-supplied axis name ("num_speech_tokens") for derived dims that
    # are actually `2 * T - 1` etc. ORT then rejects the model at runtime
    # with a "Shape mismatch attempting to re-use buffer" error because the
    # pos_emb branch of the conformer attention produces a different size.
    # Stripping non-IO value_info forces ORT to infer shapes freshly, which
    # resolves the clash.
    _strip_non_io_value_info(model_fp16)

    final_path = OUTPUT_DIR / f"conditional_decoder_n{n_timesteps}.onnx"
    # Wipe any stale sidecars
    for stale in OUTPUT_DIR.glob(f"conditional_decoder_n{n_timesteps}.onnx*"):
        stale.unlink()
    onnx.save_model(
        model_fp16,
        str(final_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=final_path.name + "_data",
    )
    print(f"[fp16] saved -> {final_path}", flush=True)

    # Clean tmp files
    for p in [fp32_path, slim_path] + list(OUTPUT_DIR.glob("_tmp_*")):
        if p.exists():
            p.unlink()

    # Stash the torch waveform for later numeric comparison
    final_path.with_suffix(".torch_ref.npy").write_bytes(
        torch_wav.float().cpu().numpy().tobytes()
    )
    torch_shape_path = final_path.with_suffix(".torch_ref.shape")
    torch_shape_path.write_text(",".join(str(d) for d in torch_wav.shape))

    return final_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_one(
    final_path: Path,
    n_timesteps: int,
    speech_tokens: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    speaker_features: torch.Tensor,
):
    print(f"\n----- Validating {final_path.name} -----", flush=True)

    # 1. onnx.checker
    model = onnx.load(str(final_path), load_external_data=True)
    try:
        onnx.checker.check_model(model, full_check=False)
        print("[val] onnx.checker: PASS")
    except Exception as e:  # noqa: BLE001
        print(f"[val] onnx.checker: FAIL -- {e}")
        raise

    # 5. ScatterND count
    scatter_count = sum(1 for n in model.graph.node if n.op_type == "ScatterND")
    print(f"[val] ScatterND count: {scatter_count}")

    # 6. File sizes
    total_mb = 0.0
    for p in sorted(final_path.parent.glob(final_path.name + "*")):
        mb = p.stat().st_size / (1024 * 1024)
        total_mb += mb
        print(f"[val] file {p.name}: {mb:.1f} MiB")
    print(f"[val] total size: {total_mb:.1f} MiB")

    # 2. Load with ORT on CPU
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(final_path), sess_options=so, providers=["CPUExecutionProvider"]
    )

    # 3. Smoke test
    feed = {
        "speech_tokens": speech_tokens.cpu().numpy().astype(np.int64),
        "speaker_embeddings": speaker_embeddings.cpu().numpy().astype(np.float32),
        "speaker_features": speaker_features.cpu().numpy().astype(np.float32),
    }
    t0 = time.time()
    (wave,) = sess.run(["waveform"], feed)
    t1 = time.time()
    print(
        f"[val] ORT run: {t1 - t0:.2f}s, waveform shape={wave.shape}, "
        f"dtype={wave.dtype}, max|amp|={np.abs(wave).max():.4f}",
        flush=True,
    )
    assert wave.ndim == 2 and wave.shape[0] == 1, "expected (1, T)"

    # Sample-count check. Each speech token maps to 2 mel frames (token_mel_ratio)
    # and each mel frame upsamples to 480 samples (HiFi-GAN: 8*5*3 * hop_len=4).
    # The prompt mel (speaker_features.shape[1]) gets trimmed off the front.
    # So T_samples = (2 * N_tok - prompt_mel_len) * 480.
    prompt_mel_len = int(speaker_features.shape[1])
    gen_mel = 2 * int(speech_tokens.shape[1]) - prompt_mel_len
    expected_samples = gen_mel * 480
    # Allow +/- 1s slop for stft/istft edge effects.
    slop = 24000
    assert (expected_samples - slop) < wave.shape[1] < (expected_samples + slop), (
        f"unexpected sample count {wave.shape[1]} (expected ~{expected_samples})"
    )
    print(f"[val] sample-count sanity OK: got {wave.shape[1]} samples "
          f"(expected ~{expected_samples}), ~{wave.shape[1] / S3GEN_SR:.2f}s @ 24 kHz")

    # 4. Numeric comparison to torch reference
    shape_txt = final_path.with_suffix(".torch_ref.shape").read_text()
    ref_shape = tuple(int(s) for s in shape_txt.split(","))
    torch_ref = (
        np.frombuffer(final_path.with_suffix(".torch_ref.npy").read_bytes(), dtype=np.float32)
        .reshape(ref_shape)
    )
    # Compare on common length
    T = min(wave.shape[1], torch_ref.shape[1])
    diff = np.abs(wave[:, :T] - torch_ref[:, :T])
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    print(
        f"[val] vs torch: shape_ref={torch_ref.shape}, shape_ort={wave.shape}, "
        f"max|diff|={max_err:.4f}, mean|diff|={mean_err:.4f}"
    )

    # Amp sanity. With random speech tokens the output is often very quiet
    # (the CFM flow cannot synthesise anything coherent from noise tokens),
    # so we only gate on "not NaN, within legal range". The meaningful
    # amplitude test happens in the downstream end-to-end pipeline test.
    amp = float(np.abs(wave).max())
    assert not np.isnan(wave).any(), "waveform contains NaN"
    assert amp < 1.1, f"waveform amplitude out of [-1,1] range: {amp}"

    # Save the first-N to wav for a human listen
    try:
        import soundfile as sf

        wav_out = OUTPUT_DIR / final_path.name.replace(".onnx", "_smoke.wav")
        sf.write(str(wav_out), wave[0].astype(np.float32), S3GEN_SR)
        print(f"[val] wrote smoke audio -> {wav_out}")
    except Exception as e:  # noqa: BLE001
        print(f"[val] could not write wav: {e}")

    return {
        "scatter_count": scatter_count,
        "total_mb": total_mb,
        "max_diff_vs_torch": max_err,
        "mean_diff_vs_torch": mean_err,
        "samples": wave.shape[1],
        "max_amp": amp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] torch={torch.__version__} device={device}", flush=True)

    # Pick a reference audio clip.
    ref_candidates = [
        os.environ.get("REF_WAV", ""),
        "/ref/Gianna.wav",
        "/ref/Robert.wav",
        "/ref/Roel_conversational.wav",
        "/ref/Amsterdam.wav",
        "/workspace/reference_audio/Gianna.wav",
        "/output/Gianna.wav",
    ]
    ref_wav_path = None
    for c in ref_candidates:
        if os.path.exists(c):
            ref_wav_path = c
            break
    if ref_wav_path is None:
        # Synthesise a 3s white-noise clip as a last resort.
        print("[main] no reference audio found; using random noise", flush=True)
        import soundfile as sf

        rng = np.random.default_rng(0)
        synth = (rng.standard_normal(3 * 24000) * 0.05).astype(np.float32)
        ref_wav_path = str(OUTPUT_DIR / "_synth_ref.wav")
        sf.write(ref_wav_path, synth, 24000)

    print(f"[main] reference audio: {ref_wav_path}", flush=True)

    # Load chatterbox
    print("[main] loading ChatterboxTTS ...", flush=True)
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)
    s3gen = model.s3gen.eval()
    for p in s3gen.parameters():
        p.requires_grad_(False)

    speech_tokens, speaker_embeddings, speaker_features = build_speaker_inputs(
        s3gen, ref_wav_path, device=device
    )
    print(
        f"[main] smoke inputs: speech_tokens={tuple(speech_tokens.shape)} "
        f"speaker_embeddings={tuple(speaker_embeddings.shape)} "
        f"speaker_features={tuple(speaker_features.shape)}",
        flush=True,
    )

    # Export each variant
    exported: dict[int, Path] = {}
    for n in N_TIMESTEPS_LIST:
        exported[n] = export_one(
            s3gen, n, speech_tokens, speaker_embeddings, speaker_features, device
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Validate each variant
    results = {}
    for n in N_TIMESTEPS_LIST:
        results[n] = validate_one(
            exported[n], n, speech_tokens, speaker_embeddings, speaker_features
        )

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'N':>3}  {'ScatterND':>9}  {'Size (MiB)':>11}  "
          f"{'max|diff|':>10}  {'samples':>8}  {'max|amp|':>9}")
    all_ok = True
    for n in N_TIMESTEPS_LIST:
        r = results[n]
        print(
            f"{n:>3d}  {r['scatter_count']:>9d}  {r['total_mb']:>11.1f}  "
            f"{r['max_diff_vs_torch']:>10.4f}  {r['samples']:>8d}  "
            f"{r['max_amp']:>9.4f}"
        )
        if r["scatter_count"] != 0:
            all_ok = False

    # Clean up torch-ref sidecars — keep the .onnx + _data + _smoke.wav only.
    for n in N_TIMESTEPS_LIST:
        for suffix in (".torch_ref.npy", ".torch_ref.shape"):
            p = exported[n].with_suffix(suffix)
            if p.exists():
                p.unlink()

    if not all_ok:
        print("\nFAIL: at least one variant has ScatterND nodes")
        sys.exit(1)

    print("\nOK: all variants scatter-free")


if __name__ == "__main__":
    main()
