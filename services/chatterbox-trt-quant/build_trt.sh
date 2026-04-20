#!/bin/bash
set -e
export PATH=/usr/local/tensorrt/bin:$PATH

# Build min/opt/max shape strings programmatically.
# We have 60 KV inputs (30 layers x {key, value}). All share dynamic axes
# (batch=0, past_len=2). For min: batch=2, past=0; opt: batch=2, past=100;
# max: batch=2, past=600.
KV_MIN=""
KV_OPT=""
KV_MAX=""
for i in $(seq 0 29); do
  for kv in key value; do
    KV_MIN+=",past_key_values.${i}.${kv}:2x16x0x64"
    KV_OPT+=",past_key_values.${i}.${kv}:2x16x100x64"
    KV_MAX+=",past_key_values.${i}.${kv}:2x16x600x64"
  done
done

# inputs_embeds: batch=2, q_len varies. attention_mask: batch=2, total_len varies.
MIN="inputs_embeds:2x1x1024,attention_mask:2x1${KV_MIN}"
OPT="inputs_embeds:2x100x1024,attention_mask:2x200${KV_OPT}"
MAX="inputs_embeds:2x500x1024,attention_mask:2x800${KV_MAX}"

trtexec \
  --onnx=/work/language_model_v2.fp8.onnx \
  --saveEngine=/work/language_model_v2.fp8.engine \
  --stronglyTyped \
  --memPoolSize=workspace:4096 \
  --minShapes="$MIN" \
  --optShapes="$OPT" \
  --maxShapes="$MAX" \
  --skipInference 2>&1 | tail -30
