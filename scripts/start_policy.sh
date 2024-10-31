#!/bin/bash

g=$1

if [ "$g" == "0" ]; then
  gpus="0,1,2,3"
  port="8000"
elif [ "$g" == "1" ]; then
  gpus="4,5,6,7"
  port="8010"
else
  echo "Invalid argument. Please provide 0 or 1."
  exit 1
fi

CUDA_VISIBLE_DEVICES=$gpus python3 -u -m vllm.entrypoints.openai.api_server --model /apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-hf --tensor-parallel-size 4 --port $port >& vllm_log &