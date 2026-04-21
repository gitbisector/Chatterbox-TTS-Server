"""Differential test: INT4 language_model_v2 vs fp16 reference.

Runs both ONNX graphs through ORT CUDA EP on the same captured decode samples
and reports cosine similarity on logits + past_key_values, logit range, and
top-1 token agreement. Pass criteria in project_chatterbox_int4_next.md:
    logits cos   > 0.98
    logit range  within 10% of fp16
    top-1 agree  >= 3/5 sampled positions
    present.N.{k,v} cos > 0.95 on intermediate layers
"""
import glob
import os
import time

import numpy as np
import onnxruntime as ort

FP16_PATH = os.environ.get("FP16_PATH", "/app/onnx-models/language_model_v2.onnx")
INT4_PATH = os.environ.get("INT4_PATH", "/app/onnx-models/language_model_v2.int4.onnx")
CALIB_DIR = os.environ.get("CALIB_DIR", "/tmp/calib_samples")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "5"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def make_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path,
        sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )


def pick_samples(n: int):
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "sample_*.npz")))
    if not files:
        raise RuntimeError(f"no samples under {CALIB_DIR}")
    # Spread across the list to get varied past_lens
    idxs = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idxs]


def run(sess: ort.InferenceSession, inputs: dict):
    name_set = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in inputs.items() if k in name_set}
    outs = sess.run(None, feeds)
    return {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}


def main():
    print(f"fp16: {FP16_PATH}")
    print(f"int4: {INT4_PATH}")

    t0 = time.time()
    s_fp16 = make_session(FP16_PATH)
    print(f"fp16 session ready in {time.time()-t0:.1f}s")
    t0 = time.time()
    s_int4 = make_session(INT4_PATH)
    print(f"int4 session ready in {time.time()-t0:.1f}s")

    out_names = [o.name for o in s_fp16.get_outputs()]
    logit_out = next(n for n in out_names if "logit" in n.lower())
    kv_outs = [n for n in out_names if n.startswith("present")]
    print(f"logit output: {logit_out}")
    print(f"kv outputs  : {len(kv_outs)}")

    samples = pick_samples(N_SAMPLES)
    print(f"testing on {len(samples)} samples: {[os.path.basename(s) for s in samples]}")

    per_sample = []
    top1_agreements = 0
    kv_cos_all = []

    for sf in samples:
        with np.load(sf) as npz:
            inputs = {k: npz[k].copy() for k in npz.files}
        past_len = inputs["past_key_values.0.key"].shape[2]

        o_fp16 = run(s_fp16, inputs)
        o_int4 = run(s_int4, inputs)

        l_fp16 = o_fp16[logit_out]
        l_int4 = o_int4[logit_out]
        lc = cosine(l_fp16, l_int4)

        fp16_range = float(l_fp16.max() - l_fp16.min())
        int4_range = float(l_int4.max() - l_int4.min())
        range_ratio = int4_range / fp16_range if fp16_range > 0 else float("nan")

        # top-1 token: logits shape (B, S, V) — take B=0 (cond), last seq pos.
        top1_fp16 = int(l_fp16[0, -1].argmax())
        top1_int4 = int(l_int4[0, -1].argmax())
        agree = top1_fp16 == top1_int4
        if agree:
            top1_agreements += 1

        # KV cos on a few layers: 0, middle, last
        kv_cs = []
        for ln in (0, len(kv_outs) // 4, len(kv_outs) // 2):
            name = kv_outs[ln]
            kv_cs.append((name, cosine(o_fp16[name], o_int4[name])))
        kv_cos_all.extend(c for _, c in kv_cs)

        per_sample.append(
            dict(
                sample=os.path.basename(sf),
                past_len=past_len,
                logits_cos=lc,
                fp16_range=fp16_range,
                int4_range=int4_range,
                range_ratio=range_ratio,
                top1_fp16=top1_fp16,
                top1_int4=top1_int4,
                top1_agree=agree,
                kv_cos=kv_cs,
            )
        )

    print()
    print(
        f"{'sample':<20} {'past':>5} {'cos(logits)':>12} {'fp16 rng':>10} {'int4 rng':>10} {'ratio':>7} {'top1':>14} {'ok':>4}"
    )
    for r in per_sample:
        print(
            f"{r['sample']:<20} {r['past_len']:>5d} {r['logits_cos']:>12.5f} {r['fp16_range']:>10.3f} {r['int4_range']:>10.3f} {r['range_ratio']:>7.3f} {r['top1_fp16']:>6d}->{r['top1_int4']:<5d} {'Y' if r['top1_agree'] else 'N':>4}"
        )

    print()
    print("KV cos (layer 0, q1, q2) per sample:")
    for r in per_sample:
        strs = " ".join(f"{n}:{c:.3f}" for n, c in r["kv_cos"])
        print(f"  {r['sample']}: {strs}")

    mean_logit_cos = np.mean([r["logits_cos"] for r in per_sample])
    mean_kv_cos = float(np.mean(kv_cos_all))
    mean_range_ratio = float(np.mean([r["range_ratio"] for r in per_sample]))

    print()
    print(f"mean logits cos   : {mean_logit_cos:.5f}")
    print(f"mean KV cos       : {mean_kv_cos:.5f}")
    print(f"mean range ratio  : {mean_range_ratio:.3f}  (INT4 / fp16)")
    print(f"top-1 agreement   : {top1_agreements}/{len(samples)}")

    # Pass-fail per spec
    passed = (
        mean_logit_cos > 0.98
        and top1_agreements >= 3
        and mean_kv_cos > 0.95
        and 0.9 <= mean_range_ratio <= 1.1
    )
    print()
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
