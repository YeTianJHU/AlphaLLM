#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
SERVER_DIR="${SCRIPT_DIR}/../"

USAGE="ORM" # ORM or SRM
DATASET="gsm8k" # gsm8k or math or jiping
GPU_GROUP=0 # 0 or 1. 0 for 0,1,2,3 and 1 for 4,5,6,7
usage() {
  echo "This script is for value network deployment."
  echo "Options:"
  echo "-m     model path"
  echo "-s     server dir"
  echo "-u    usage (ORM or SRM)"
  echo "-d    dataset (gsm8k or math)"
  echo "-g     gpu group (0 or 1)"
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":m:s:u:d:g:" options; do
  case "${options}" in
    m)
      MODEL_PATH=${OPTARG}
      ;;
    s)
      SERVER_DIR=${OPTARG}
      ;;
    u)
      USAGE=${OPTARG}
      ;;
    d)
      DATASET=${OPTARG}  
      ;;
    g)
      GPU_GROUP=${OPTARG}
      ;;
    :)                                    # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal                       # Exit abnormally.
      ;;
    *)                                    # If unknown (any other) option:
      exit_abnormal                       # Exit abnormally.
      ;;
  esac
done

if [ "$GPU_GROUP" == "0" ]; then
  gpus="0,1,2,3"
  tp=4
elif [ "$GPU_GROUP" == "1" ]; then
  gpus="4,5,6,7"
  tp=4
elif [ "$GPU_GROUP" == "all" ]; then
  gpus="0,1,2,3,4,5,6,7" # llama3-70b based models: tp=8
  tp=8
else
  echo "Invalid argument. Please provide 0 or 1."
  exit 1
fi

if [ "$DATASET" == "gsm8k" ]; then
  ORM_MODEL_PATH=/apdcephfs/share_300000800/user/haitaomi/exp.tencent_chat/vllm_inference/models/outcome_gsm8k_llama2_70b_lr1e6_maxseq2048_thre20_lmonly
  # ORM_MODEL_PATH=/apdcephfs/share_300000800/user/haitaomi/exp.tencent_chat/vllm_inference/models/outcome_gsm8k_math_llama2_70b_lr1e6_maxseq2048_thre20_lmonly
  # ORM_MODEL_PATH=/apdcephfs/share_300000800/user/haitaomi/exp.tencent_chat/vllm_inference/models/outcome_gsm8k_self_improved_llama2_70b_lr1e6_maxseq2048_thre20_lmonly_epoch1
  SRM_MODEL_PATH=/apdcephfs/share_300000800/user/haitaomi/exp.tencent_chat/vllm_inference/models/step_reward_gsm8k_llama2_70b_lr5e6_maxseq2048_thre20_lmonly # base
  # SRM_MODEL_PATH=/apdcephfs/share_300000800/user/haitaomi/exp.tencent_chat/vllm_inference/models/step_reward_gsm8k_cal_enc_llama2_70b_lr1e6_maxseq2048_thre20_lmonly # calc
  ORM_PORT=8103
  # ORM_PORT=8104
  # ORM_PORT=8300
  SRM_PORT=8102
else
  echo "Invalid argument. Please provide gsm8k or math."
  exit 1
fi

echo ${SERVER_DIR}

if [ "$USAGE" == "ORM" ]; then 
  echo "Using ORM $ORM_MODEL_PATH on GPUs $gpus fot dataset $DATASET, port $ORM_PORT"
  CUDA_VISIBLE_DEVICES=$gpus python3 "${SERVER_DIR}/server/step_server_70b.py" --host 0.0.0.0 --port $ORM_PORT --model $ORM_MODEL_PATH --tensor-parallel-size $tp >& vllm_log_orm &
elif [ "$USAGE" == "SRM" ]; then 
  echo "Using SRM $SRM_MODEL_PATH on GPUs $gpus fot dataset $DATASET, port $SRM_PORT"
  CUDA_VISIBLE_DEVICES=$gpus python3 "${SERVER_DIR}/server/step_server_70b.py" --host 0.0.0.0 --port $SRM_PORT --model $SRM_MODEL_PATH --tensor-parallel-size 4 >& vllm_log_step &
fi