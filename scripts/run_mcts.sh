#!/bin/bash

USER=yaptian
LOGGING_LEVEL=3

DATASET="gsm8k" # gsm8k only
IS_IMPROVED=false
N_PROCESS=200
SEARCH_CONTINUE=true
MACHINE_ID="0"
SUFFIX="_test"   # other inform0tion to add 

VALUE_WEIGHT=1.0 # weight of value in the total reward. 0 means no value.
N_SIMULATIONS=4 # number of simulations with simulation policy model. 0 means no simulation.
FASTROLLOUT_WEIGHT=1.0 # weight of fast rollout reward in the total reward. 0 means no simulation.
STEP_REWARD_WEIGHT=0.0 # weight of step reward in the total reward. 0 means no step reward.

SEARCH_TIME=40 # total search times (equal or larger than than the number of nodes expanded)
MIN_TERMINALS=40 # minimum number of terminals must search
INIT_CHILDREN=40 # initial number of children to expand at the root node. if -1, use N_TOTAL_CHILDREN. This is a piror knowledge of the problem.

# majory parameters
N_TOTAL_CHILDREN=3 # number of children to expand at each node
C_PARAM=1 # exploration parameter
WIDTH_INCREASE_FACTOR=2  # increase the width of the tree by this factor in Adaptive child allocation
MAX_SPLIT_DEPTH=10 # maximum depth to split the tree. If larger, only single path will be expanded. If -1, no limit. This is a piror knowledge of the problem.

SEARCH_EVERY_N=1
SEARCH_MOD_M=0
MAX_DEPTH=10 # max search depth. We will stop search when depth is reached.

EXEC_CODE=false
USE_CONSISTENCY_AS_ORM=false
CONTINUE_FROM=0 # 38500
FILTER_PREV_ITER=false

if [ "$DATASET" == "gsm8k" ]; then
  DATA_PATH="/apdcephfs/share_300000800/user/baolinpeng/local_exp/codes/vllm-inference/llama-70b-gsm8k_cot_8_shot_test.decoded.n50.topp1.temp0.8.jsonl"
  if [ "$DATA_PATH" != "/apdcephfs/share_300000800/user/baolinpeng/local_exp/codes/vllm-inference/llama-70b-gsm8k_cot_8_shot_test.decoded.n50.topp1.temp0.8.jsonl" ]; then
    echo "Attention: -d ${DATA_PATH} is not testing!!!!"
    echo "Attention: -d ${DATA_PATH} is not testing!!!!"
    echo "Attention: -d ${DATA_PATH} is not testing!!!!"
  fi
else
  echo "Error: -d ${DATASET} is not gsm8k"
  exit 1
fi

GREEDY_PATH=true # whether to decode greedyly for the first child
USE_CALCULATOR=false # whether to use calculator 
USE_MATH_EXTRACTOR=false # whether to use math extractor
STOP_EXPANSION_ROLLOUT_VAR=-1 # stop expanding a node if the fastrollout variance is smaller than this value

# state merge
MAX_N_REPEAT=5 # maximum number of repeated in Diverse Node Expansion
STOP_REPEAT_NODE_THRES=10 # stop expanding a node if it has been repeated for this number of times. If this is larger than MAX_N_REPEAT, it will be deactivated.
STATE_MERGE_METHOD="edit_distance" # method to merge states for Diverse Node Expansion . Options: "model_based", "edit_distance"
EDIT_DIST_THRES=20 # edit distance threshold for Diverse Node Expansion 50
SIMILARITY_THRES=0.2 # similarity threshold for Diverse Node Expansion. This rate is the (n_similar_action / n_total_children).
SAVE_DATA=true # whether to save the data

PREV_ITER_FILE='/apdcephfs/share_300000800/user/yaptian/exp.tencent_chat/math-search/prev_iter_files/tmp.p'



if [ "$DATASET" == "gsm8k" ]; then
  USE_RULE_BASED_STEP_EXTEND=false # whether to use rule based step extend 
  # sympy
  ORM_USE_SYMPY=false # whether to use sympy to calculate the outcome
  ORM_PRESET_CODE_PATH=""
  ORM_CALIBRATE_LOGITS=false
else 
  USE_RULE_BASED_STEP_EXTEND=true # whether to use rule based step extend 
  # sympy
  ORM_USE_SYMPY=false # whether to use sympy to calculate the outcome
  ORM_PRESET_CODE_PATH="/apdcephfs/share_300000800/user/baolinpeng/exp.tencent_chat/data/math_search/MATH_preset_code_all.json" 
  # ORM_PRESET_CODE_PATH="/apdcephfs/share_300000800/user/baolinpeng/exp.tencent_chat/data/math_search/MATH_preset_code_codellama34b_lr1e5_majority_voted_subset1000.json"  # codellama
  ORM_CALIBRATE_LOGITS=false
fi


GSM8K_BASE_POLICY_URL="http://localhost:8000/v1"
GSM8K_BASE_POLICY_MODEL="/apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-hf"
GSM8K_BASE_POLICY_TYPE="llama2_70b"
GSM8K_IP_POLICY_URL="http://localhost:8004/v1"
# GSM8K_IP_POLICY_MODEL="/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/megatronlm-llama2/ckpt_math/llama2_70b_hf_gsm8k_global_best_correct_only_lr5e6_400" # v1
# GSM8K_IP_POLICY_TYPE="llama2_70b_improved"
# GSM8K_IP_POLICY_MODEL="/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/megatronlm-llama2/ckpt_math/llama2_70b_from_r0_hf_r0r2_global_best_correct_only_lr5e6_200"
# GSM8K_IP_POLICY_TYPE="llama2_70b_improved_meta"
GSM8K_IP_POLICY_MODEL="/apdcephfs/share_300000800/user/yaptian/exp.tencent_chat/models/llama2_70b_from_r0_hf_r0_gb_co_r2_gb_ost098_lr5e6_200"
GSM8K_IP_POLICY_TYPE="llama2_70b_improved_metanoco"
# GSM8K_IP_POLICY_MODEL="/apdcephfs/share_300000800/user/yaptian/exp.tencent_chat/models/llama2_70b_from_rerank_r1_hf_orm_scored_t90_top1_lr5e6_150"
# GSM8K_IP_POLICY_TYPE="llama2_70b_improved_rerank"
# GSM8K_IP_POLICY_MODEL="/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/megatronlm-llama2/ckpt_math/llama2_70b_from_r0_hf_r0r2round3_global_best_correct_only_lr5e6_200"
# GSM8K_IP_POLICY_TYPE="llama2_70b_improved_metamore"
GSM8K_STEP_REWARD_URL="http://30.159.161.178:8102/generate" # all
GSM8K_STEP_REWARD_TYPE="llama2_70b"
GSM8K_STEP_REWARD_MODE="latest" # how to calculate step reward. Options: "latest", "avg", "max", "min"
GSM8K_ORM_URL="http://localhost:8103/generate"  
GSM8K_ORM_TYPE="llama2_70b_lm" 

R3="11.220.10.237"
R4="11.220.28.24"
R8="11.216.42.111"

POLICY_URLS=(
  "http://${R3}:8002/v1"
  "http://${R3}:8012/v1"
  "http://${R3}:8022/v1"
  "http://${R4}:8002/v1"
  "http://${R4}:8012/v1"
  "http://${R4}:8022/v1"
  "http://${R4}:8032/v1"
  "http://${R8}:8002/v1"
  "http://${R8}:8012/v1"
  "http://${R8}:8022/v1"
  "http://${R8}:8032/v1"
  "http://${R8}:8042/v1"
  "http://${R8}:8052/v1"
  "http://${R8}:8062/v1"
  "http://${R8}:8072/v1"
)
POLICY_URL=$(IFS=,; echo "${POLICY_URLS[*]}")


if [ "$DATASET" == "gsm8k" ]; then

  if [ "$IS_IMPROVED" = true ]; then
    POLICY_URL=$GSM8K_IP_POLICY_URL
    POLICY_MODEL=$GSM8K_IP_POLICY_MODEL
    POLICY_TYPE=$GSM8K_IP_POLICY_TYPE
  else
    POLICY_URL=$GSM8K_BASE_POLICY_URL
    POLICY_MODEL=$GSM8K_BASE_POLICY_MODEL
    POLICY_TYPE=$GSM8K_BASE_POLICY_TYPE
  fi

  STEP_REWARD_URL=$GSM8K_STEP_REWARD_URL
  STEP_REWARD_TYPE=$GSM8K_STEP_REWARD_TYPE
  STEP_REWARD_MODE=$GSM8K_STEP_REWARD_MODE

  ORM_URL=$GSM8K_ORM_URL
  ORM_TYPE=$GSM8K_ORM_TYPE

elif [ "$DATASET" == "math" ]; then

  if [ "$IS_IMPROVED" = true ]; then
    POLICY_URL=$MATH_IP_POLICY_URL
    POLICY_MODEL=$MATH_IP_POLICY_MODEL
    POLICY_TYPE=$MATH_IP_POLICY_TYPE
  else
    POLICY_URL=$MATH_BASE_POLICY_URL
    POLICY_MODEL=$MATH_BASE_POLICY_MODEL
    POLICY_TYPE=$MATH_BASE_POLICY_TYPE
  fi

  STEP_REWARD_URL=$MATH_STEP_REWARD_URL
  STEP_REWARD_TYPE=$MATH_STEP_REWARD_TYPE
  STEP_REWARD_MODE=$MATH_STEP_REWARD_MODE

  ORM_URL=$MATH_ORM_URL
  ORM_TYPE=$MATH_ORM_TYPE
fi

# VALUE_TYPE="llama2_13b_calcrub"
# SIMULATE_POLICY_MODEL="/apdcephfs/share_300000800/user/baolinpeng/exp.tencent_chat/models/hf_models/Abel-7B-002"
# SIMULATE_POLICY_TYPE="abel_7b_002"
# VALUE_TYPE="llama2_13b_new" 
VALUE_TYPE="llama3_8b" 
# VALUE_URL="http://30.159.160.32:8006/predict,http://30.159.160.32:8016/predict,http://30.159.160.32:8036/predict,http://30.159.160.32:8036/predict,http://30.159.162.228:8006/predict,http://30.159.162.228:8016/predict"
VALUE_URL="http://30.159.160.32:8006/predict,http://30.159.160.32:8016/predict,http://30.159.160.32:8036/predict,http://30.159.160.32:8036/predict"
# SIMULATE_POLICY_MODEL="/apdcephfs/share_300000800/user/baolinpeng/exp.tencent_chat/models/hf_models/Meta-Llama-3-8B-Instruct"
SIMULATE_POLICY_MODEL="/apdcephfs_cq10/share_1150325/yaptian/exp.tencent_chat/models/Meta/Meta-Llama-3-8B-Instruct"
SIMULATE_POLICY_TYPE="llama3_8b"
SIMULATE_POLICY_URL=$MATH_BASE_POLICY_URL

ADD_VALUE_WEIGHT=0.0
ADD_VALUE_URL="http://30.159.162.126:8106/generate" # predict for regression, generate for lm
ADD_VALUE_TYPE="llama2_70b_lm"
MATH_EXTRACTOR_URL="http://9.206.62.242:8000/generate" 
STATE_MERGE_URL="http://30.159.160.186:8200/v1"
STATE_MERGE_MODEL="/apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-chat-hf"
STATE_MERGE_MODEL_TYPE="llama2_70b_chat"

EXP_NAME=${DATASET}_search_${SEARCH_TIME}_init_${INIT_CHILDREN}_msd_${MAX_SPLIT_DEPTH}_child_${N_TOTAL_CHILDREN}_depth_${MAX_DEPTH}_alpha_${WIDTH_INCREASE_FACTOR}_c_${C_PARAM}_minter_${MIN_TERMINALS}_p_${POLICY_TYPE}


# value
if [ $(echo "$VALUE_WEIGHT > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_v_${VALUE_WEIGHT}_${VALUE_TYPE}"
fi
# additional value
if [ $(echo "$ADD_VALUE_WEIGHT > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_addv_${ADD_VALUE_WEIGHT}_${ADD_VALUE_TYPE}"
fi
# step reward
if [ $(echo "$STEP_REWARD_WEIGHT > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_step_${STEP_REWARD_WEIGHT}_${STEP_REWARD_MODE}_${STEP_REWARD_TYPE}"
fi
# fast rollout
if [ $(echo "$N_SIMULATIONS > 0" | bc -l) -eq 1 ] && [ $(echo "$FASTROLLOUT_WEIGHT > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_fr_${FASTROLLOUT_WEIGHT}_${N_SIMULATIONS}_${ORM_TYPE}"
fi
# state merge
if [ $MAX_N_REPEAT -gt 1 ]; then
    EXP_NAME+="_merge_${STATE_MERGE_METHOD}_r_${MAX_N_REPEAT}"
    if [ "$STATE_MERGE_METHOD" == "edit_distance" ]; then
        EXP_NAME+="_${EDIT_DIST_THRES}"
    elif [ "$STATE_MERGE_METHOD" == "model_based" ]; then
        EXP_NAME+="_${SIMILARITY_THRES}_${STATE_MERGE_MODEL_TYPE}"
    fi
    if [ $STOP_REPEAT_NODE_THRES -le $MAX_N_REPEAT ]; then
        EXP_NAME+="_stop_expand_${STOP_REPEAT_NODE_THRES}"
    fi
fi
# # greedy path is default
# if [ $GREEDY_PATH = true ]; then
#     EXP_NAME+="_greedypath"
# fi
# calculator  
# if [ $USE_CALCULATOR = true ]; then
#     EXP_NAME+="_calc"
# fi
# math extractor
if [ $USE_MATH_EXTRACTOR = true ]; then
    EXP_NAME+="_extr"
fi
# rule based step extend
if [ $USE_RULE_BASED_STEP_EXTEND = true ]; then
    EXP_NAME+="_rulestep"
fi
# stop_expansion_rollout_var
if [ $(echo "$STOP_EXPANSION_ROLLOUT_VAR > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_stopvar_${STOP_EXPANSION_ROLLOUT_VAR}"
fi
# ORM_CALIBRATE_LOGITS
if [ $ORM_CALIBRATE_LOGITS = true ]; then
    EXP_NAME+="_calib"
fi
# math subset
if [ "$DATASET" == "math" ]; then
    EXP_NAME+="_subset${MATH_SUBSET}"
fi
# use consistency as orm
if [ $USE_CONSISTENCY_AS_ORM = true ]; then
    EXP_NAME+="_consasorm"
fi
# code execution
if [ $EXEC_CODE = true ]; then
    EXP_NAME+="_exec"
fi
# continue form
if [ $(echo "$CONTINUE_FROM > 0" | bc -l) -eq 1 ]; then
    EXP_NAME+="_from${CONTINUE_FROM}"
fi
echo $EXP_NAME$SUFFIX

WORKSPACE_DIR=/apdcephfs/share_300000800/user/${USER}/exp.tencent_chat
SCRIPT_DIR=${WORKSPACE_DIR}/math-search
OUTPUT_DIR=${WORKSPACE_DIR}/output/mcts_${EXP_NAME}

LOG_FILE_NAME=log_${EXP_NAME}${SUFFIX}.log

mkdir -p $OUTPUT_DIR

export PYTHONPATH=${SCRIPT_DIR}:$PYTHONPATH

export http_proxy=""
export https_proxy=""


python3 -u ${SCRIPT_DIR}/search/experiment.py \
  --dataset "$DATASET" \
  --search_every_n ${SEARCH_EVERY_N} \
  --search_mod_m ${SEARCH_MOD_M} \
  --search_continue ${SEARCH_CONTINUE} \
  --data_path "$DATA_PATH" \
  --max_depth "$MAX_DEPTH" \
  --n_total_children "$N_TOTAL_CHILDREN" \
  --init_children "$INIT_CHILDREN" \
  --max_split_depth "$MAX_SPLIT_DEPTH" \
  --c_param "$C_PARAM" \
  --search_time "$SEARCH_TIME" \
  --min_terminals ${MIN_TERMINALS} \
  --save_data "$SAVE_DATA" \
  --state_merge_method "$STATE_MERGE_METHOD" \
  --max_n_repeat "$MAX_N_REPEAT" \
  --edit_dist_thres "$EDIT_DIST_THRES" \
  --similarity_thres "$SIMILARITY_THRES" \
  --width_increase_factor "$WIDTH_INCREASE_FACTOR" \
  --policy_url "$POLICY_URL" \
  --value_url "$VALUE_URL" \
  --policy_type "$POLICY_TYPE" \
  --value_type "$VALUE_TYPE" \
  --policy_model "$POLICY_MODEL" \
  --simulate_policy_url "$SIMULATE_POLICY_URL" \
  --simulate_policy_type "$SIMULATE_POLICY_TYPE" \
  --simulate_policy_model "$SIMULATE_POLICY_MODEL" \
  --step_reward_url "$STEP_REWARD_URL" \
  --step_reward_type "$STEP_REWARD_TYPE" \
  --step_reward_weight "$STEP_REWARD_WEIGHT" \
  --step_reward_mode "$STEP_REWARD_MODE" \
  --state_merge_url "$STATE_MERGE_URL" \
  --state_merge_model "$STATE_MERGE_MODEL" \
  --state_merge_model_type "$STATE_MERGE_MODEL_TYPE" \
  --add_value_url "$ADD_VALUE_URL" \
  --add_value_type "$ADD_VALUE_TYPE" \
  --stop_repeat_node_thres "$STOP_REPEAT_NODE_THRES" \
  --n_simulations "$N_SIMULATIONS" \
  --fastrollout_weight "$FASTROLLOUT_WEIGHT" \
  --orm_url "$ORM_URL" \
  --orm_type "$ORM_TYPE" \
  --value_weight "$VALUE_WEIGHT" \
  --greedy_path "$GREEDY_PATH" \
  --use_rule_based_step_extend "$USE_RULE_BASED_STEP_EXTEND" \
  --stop_expansion_rollout_var "$STOP_EXPANSION_ROLLOUT_VAR" \
  --use_calculator "$USE_CALCULATOR" \
  --use_math_extractor "$USE_MATH_EXTRACTOR" \
  --math_extractor_url "$MATH_EXTRACTOR_URL" \
  --orm_use_sympy "$ORM_USE_SYMPY" \
  --orm_preset_code_path "$ORM_PRESET_CODE_PATH" \
  --orm_calibrate_logits "$ORM_CALIBRATE_LOGITS" \
  --consistency_url "$CONSISTENCY_URL" \
  --consistency_model "$CONSISTENCY_MODEL" \
  --use_consistency_as_orm "$USE_CONSISTENCY_AS_ORM" \
  --continue_from "$CONTINUE_FROM" \
  --exec_code "$EXEC_CODE" \
  --output-dir $OUTPUT_DIR \
  --debug-log-level ${LOGGING_LEVEL} \
  --exp_name ${EXP_NAME} \
  --suffix "$SUFFIX" \
  --n_process ${N_PROCESS} \
  --filter_prev_iter ${FILTER_PREV_ITER} \
  --prev_iter_file ${PREV_ITER_FILE} \
  >& ${OUTPUT_DIR}/${LOG_FILE_NAME} &

ln -s ${OUTPUT_DIR}/${LOG_FILE_NAME} logs/${LOG_FILE_NAME}