"""
Numpy port of chatterbox.models.t3.inference.alignment_stream_analyzer.AlignmentStreamAnalyzer.

Reads attention output from the new T3 ONNX export (which exposes layers 9, 12, 13)
and applies the same heuristics as the PyTorch path to detect hallucinations and
force EOS at the right moment. Runs entirely in numpy / Python — no torch dependency.

Usage:
    analyzer = AlignmentStreamAnalyzer(
        text_tokens_slice=(len_cond, len_cond + len_text),
        eos_idx=STOP_SPEECH_TOKEN,
    )
    # In the decode loop, after the LM step:
    logits = analyzer.step(logits_np, attn_layers_np, next_token=last_token_id)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Heads selected by the original Chatterbox impl. The new T3 export exposes attention
# from layers 9, 12, 13 (in that index order), so the head selection here is per-layer.
# Original: LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)] — but the new export
# already exposes only those layers in order [9, 12, 13], so we reindex.
ALIGNED_HEADS_BY_EXPORT_LAYER_IDX = {
    0: 2,   # corresponds to layer 9, head 2
    1: 15,  # corresponds to layer 12, head 15
    2: 11,  # corresponds to layer 13, head 11
}


@dataclass
class AlignmentAnalysisResult:
    false_start: bool
    long_tail: bool
    repetition: bool
    discontinuity: bool
    complete: bool
    position: int


class AlignmentStreamAnalyzer:
    """Numpy port of Chatterbox's AlignmentStreamAnalyzer.

    The PyTorch original injects forward hooks into specific attention layers.
    Here we expect the ONNX graph to emit `attn_layers` of shape
    `(num_aligned_layers=3, num_heads=16, T_q, T_kv)` — already filtered to the
    conditional batch row only (CFG batch=2 produces logits for cond only after combine).
    """

    # Hard cap: roughly how many speech tokens per text token is ever reasonable.
    # Natural rate is ~3-6 tokens/char; 25 gives a large safety margin. Going over this
    # always means the model is hallucinating and we should force EOS.
    MAX_STEPS_PER_TEXT_TOKEN = 25
    # Absolute floor for short inputs — don't over-truncate on "Hi." either.
    MIN_HARD_CAP = 150

    def __init__(self, text_tokens_slice: tuple[int, int], eos_idx: int):
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx

        self.alignment = np.zeros((0, j - i), dtype=np.float32)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at: Optional[int] = None

        self.complete = False
        self.completed_at: Optional[int] = None

        self.generated_tokens: list[int] = []

        # Deterministic upper bound on step count — prevents the pathological 400+ step
        # runs that happen when the attention-based `complete` heuristic never trips.
        self.hard_step_cap = max(self.MIN_HARD_CAP, (j - i) * self.MAX_STEPS_PER_TEXT_TOKEN)

    @staticmethod
    def _select_aligned_heads(attn_layers: np.ndarray) -> np.ndarray:
        """Pick the configured head from each aligned layer and average → (T_q, T_kv).

        attn_layers shape: (num_aligned_layers, num_heads, T_q, T_kv)
        """
        slices = []
        for layer_idx, head_idx in ALIGNED_HEADS_BY_EXPORT_LAYER_IDX.items():
            slices.append(attn_layers[layer_idx, head_idx])
        # Stack and average over the 3 selected (layer, head) pairs.
        return np.stack(slices, axis=0).mean(axis=0)  # (T_q, T_kv)

    def step(
        self,
        logits: np.ndarray,
        attn_layers: np.ndarray,
        next_token: Optional[int] = None,
    ) -> np.ndarray:
        """Apply alignment-aware integrity checks and possibly force EOS in logits.

        Args:
            logits: (1, vocab) — logits for the next token after CFG combine
            attn_layers: (3, num_heads, T_q, T_kv) — attention from layers [9, 12, 13]
            next_token: optionally, the most recently emitted token id (for repetition tracking)

        Returns:
            Modified logits (same shape).
        """
        aligned_attn = self._select_aligned_heads(attn_layers)  # (T_q, T_kv)
        i, j = self.text_tokens_slice

        if self.curr_frame_pos == 0:
            # First chunk has conditioning + text + BOS
            A_chunk = aligned_attn[j:, i:j].copy()  # (T, S)
        else:
            # Subsequent steps: KV-cached, T_q==1
            A_chunk = aligned_attn[:, i:j].copy()  # (1, S)

        # Monotonic cleanup: zero out alignment past the current position + 1
        if self.curr_frame_pos + 1 < A_chunk.shape[1]:
            A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = np.concatenate([self.alignment, A_chunk], axis=0)

        A = self.alignment
        T, S = A.shape

        # Position update
        cur_text_posn = int(np.argmax(A_chunk[-1]))
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # False-start detection
        false_start = (not self.started) and (
            A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5
        )
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Completion check
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # Long-tail detection
        long_tail = self.complete and (
            self.completed_at is not None and
            A[self.completed_at:, -3:].sum(axis=0).max() >= 5
        )

        # Repetition via alignment
        alignment_repetition = self.complete and (
            self.completed_at is not None and
            A[self.completed_at:, :-5].max(axis=1).sum() > 5
        )

        # Token-level repetition (3 of same token in a row)
        if next_token is not None:
            self.generated_tokens.append(int(next_token))
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = (
            len(self.generated_tokens) >= 3
            and len(set(self.generated_tokens[-2:])) == 1
        )

        if token_repetition:
            logger.warning(
                f"Detected 2x repetition of token {self.generated_tokens[-1]}"
            )

        # Hard step cap — reliability safety net when the attention-based heuristics
        # don't detect completion (happens occasionally on some inputs). Without this,
        # we can burn through 400+ steps before token_repetition catches the runaway.
        hard_cap_hit = self.curr_frame_pos >= self.hard_step_cap

        # Suppress EOS until we've reached the end of text — but NEVER when the hard
        # cap has fired, otherwise we can't force-terminate.
        if not hard_cap_hit and cur_text_posn < S - 3 and S > 5:
            logits = logits.copy()
            logits[..., self.eos_idx] = -(2 ** 15)

        # Force EOS on bad-ending detection or hard cap
        if long_tail or alignment_repetition or token_repetition or hard_cap_hit:
            logger.warning(
                f"forcing EOS at step {self.curr_frame_pos}, long_tail={long_tail} "
                f"alignment_rep={alignment_repetition} token_rep={token_repetition} "
                f"hard_cap={hard_cap_hit} (cap={self.hard_step_cap})"
            )
            logits = -(2 ** 15) * np.ones_like(logits)
            logits[..., self.eos_idx] = 2 ** 15

        self.curr_frame_pos += 1
        return logits
