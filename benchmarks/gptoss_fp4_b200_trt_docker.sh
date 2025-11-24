#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
# PORT

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

# GPTOSS TRTLLM Deployment Guide:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/quick-start-recipe-for-gpt-oss-on-trtllm.md

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
EP_SIZE="1"
MOE_BACKEND="TRTLLM"
DP_ATTENTION=false

# Higher concurrencies: Concurrency >= 256
#   MoE Backend = CUTLASS
#   Use DP attention with expert parallel MoE
if [[ $CONC -ge 256 ]]; then
    EP_SIZE="$TP"
    DP_ATTENTION=true
fi

echo "Final configuration: EP_SIZE='$EP_SIZE', MOE_BACKEND='$MOE_BACKEND', DP_ATTENTION='$DP_ATTENTION'"

EXTRA_CONFIG_FILE="gptoss-fp4.yml"
export TRTLLM_ENABLE_PDL=1
export NCCL_GRAPH_REGISTER=0

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONC
enable_attention_dp: $DP_ATTENTION
kv_cache_config:
    dtype: fp8
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.85
print_iter_log: true
stream_interval: 20
num_postprocess_workers: 4
moe_config:
    backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    enable_balance: true
EOF
fi

echo "Generated config file contents:"
cat $EXTRA_CONFIG_FILE

set -x

MAX_NUM_TOKENS=20000

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size 512 \
    --max_seq_len=$MAX_MODEL_LEN \
    --max_num_tokens=$MAX_NUM_TOKENS \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --extra_llm_api_options=$EXTRA_CONFIG_FILE > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/