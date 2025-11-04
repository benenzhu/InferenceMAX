#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# TP
# NUM_FEWSHOT (optional, defaults to 5)
# LIMIT (optional, empty for all examples)
# RESULT_FILENAME (optional, for output filename)

set -e

# Set defaults
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
PORT=${PORT:-8000}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace}

# Start SGLang server in background
echo "Starting SGLang server for model: $MODEL"
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
--model-path $MODEL \
--host 0.0.0.0 \
--port $PORT \
--tp-size $TP \
--mem-fraction-static 0.9 &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:$PORT/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Server failed to start within timeout"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

echo "Server started successfully"

# Run lm-eval with GSM8k
echo "Running GSM8k evaluation..."

# Create temporary output directory
TEMP_OUTPUT_DIR=$(mktemp -d)

# Build lm-eval command
LMEVAL_CMD="lm_eval --model vllm \
--model_args pretrained=$MODEL,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,base_url=http://localhost:$PORT/v1 \
--tasks gsm8k \
--num_fewshot $NUM_FEWSHOT \
--batch_size auto \
--output_path $TEMP_OUTPUT_DIR"

# Add limit if specified
if [ -n "$LIMIT" ]; then
    LMEVAL_CMD="$LMEVAL_CMD --limit $LIMIT"
fi

# Run evaluation
eval $LMEVAL_CMD

# Copy results to expected location
if [ -n "$RESULT_FILENAME" ]; then
    # lm-eval creates results.json in output directory
    if [ -f "$TEMP_OUTPUT_DIR/results.json" ]; then
        cp "$TEMP_OUTPUT_DIR/results.json" "$OUTPUT_DIR/$RESULT_FILENAME.json"
        echo "Results saved to $OUTPUT_DIR/$RESULT_FILENAME.json"
    else
        echo "Error: lm-eval output not found"
        ls -la $TEMP_OUTPUT_DIR
    fi
fi

# Shutdown server
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "GSM8k evaluation completed"
