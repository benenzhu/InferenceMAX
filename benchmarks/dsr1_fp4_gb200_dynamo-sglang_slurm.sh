
#!/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars CONC_LIST ISL OSL IMAGE SPEC_DECODING MODEL_PATH \
    PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN \
    PREFILL_NODES DECODE_NODES N_ADDITIONAL_FRONTENDS

# Always clone and setup Dynamo
echo "Cloning Dynamo repository..."
git clone --branch update-result-file-name https://github.com/Elnifio/dynamo.git
cd "dynamo/components/backends/sglang/slurm_jobs"

# Set up SGL launch script-specific environment variables
export TIME_LIMIT="04:00:00"
export MODEL_PATH=$MODEL_PATH
export CONFIG_DIR=$CONFIG_DIR
export CONTAINER_IMAGE=$IMAGE

# Launch jobs based on ISL/OSL
# Replace ' ' in CONC_LIST with 'x' such that the concurrency list is represented
# by a list of numbers delimted by 'x'. This is because of how the underlying launch script
# expects the concurrencies.
bash ./submit_disagg.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $DECODE_NUM_WORKERS \
    $N_ADDITIONAL_FRONTENDS \
    $ISL $OSL "${CONC_LIST// /x}" inf

# if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
#     bash ./submit_disagg.sh $PREFILL_NODES \
#         $PREFILL_NUM_WORKERS \
#         $DECODE_NODES \
#         $DECODE_NUM_WORKERS \
#         $DECODE_NUM_WORKERS \
#         $ISL $OSL "${CONC_LIST// /x}" inf
# elif [ "$ISL" = "8192" ] && [ "$OSL" = "1024" ]; then
#     concurrency_list="128x256x384x448x512x576x1024x2048x4096"
#     bash ./submit_disagg.sh 12 6 6 1 8 $ISL $OSL "${CONC_LIST// /x}" inf
# else
#     echo "Unsupported ISL/OSL combination: $ISL/$OSL"
#     exit 1
# fi