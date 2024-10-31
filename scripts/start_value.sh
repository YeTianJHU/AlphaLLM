#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
SERVER_DIR="${SCRIPT_DIR}/../"

GPU_ID=$1
# Set the model path
export MODEL_PATH='/apdcephfs/share_300000800/user/yudian/math-search/output/value_net_math_llama2_13b_lr1e-6_maxseq1024_nodes2_devicebatch8_nsamples30_success_rate_full_data'

# Set the port number based on the GPU_ID value
if [[ $GPU_ID == "0,1" ]]; then
  PORT=8009
elif [[ $GPU_ID == "2,3" ]]; then
  PORT=8109
elif [[ $GPU_ID == "4,5" ]]; then
  PORT=8209
elif [[ $GPU_ID == "6,7" ]]; then
  PORT=8309
else
  echo "Invalid GPU_ID value"
  exit 1
fi

# Start the uvicorn server with the specified GPU_ID and port number
CUDA_VISIBLE_DEVICES=$GPU_ID uvicorn --app-dir "${SERVER_DIR}" server.value_server:app --host 0.0.0.0 --port $PORT >& value_log &