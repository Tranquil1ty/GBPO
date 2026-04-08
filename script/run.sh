#!/bin/bash

# --- 1. 运行环境 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8
PORT=29501
SCRIPT="./main.py"

# --- 2. 基础配置 ---
DATASET=${1:-"Beauty"}     # 数据集: Beauty, Toys, etc.
MODE=${2:-"SFT"}           # 模式: SFT/GRPO/GBPO/GUPO/GSPO/GPPO/DUAL_PPO/SAPO/DAPO/DPO
RUN_TYPE=${3:-"nohup"}     # 运行方式: torchrun (前台) / nohup (后台)
BASE_PATH=".."

# --- 3. 构造 Python 参数 ---
PY_ARGS="--dataset $DATASET --mode $MODE --base_path $BASE_PATH"

BATCH_SIZE=128
LR="1e-3"
if [ "$MODE" == "SFT" ]; then
    BATCH_SIZE=1024
    LR="1e-3"
fi

PY_ARGS="$PY_ARGS --batch_size $BATCH_SIZE --learning_rate $LR --epochs 500"

# 可选开关 (通过 positional args 4, 5, 6, 7, 8, 9 传入)
IS_ON_POLICY=${4:-"false"}
IS_VERBOSE=${5:-"false"}       # 默认 true
GEN_STRATEGY=${6:-"rollout"}    # 生成策略 (rollout/beam)
USE_TRIE=${7:-"no_trie"}        # trie 模式: no_trie, trie_gen, trie_all
REWARD_TOPK=${8:-"5"}           # Reward 阈值个数 (默认 5)
NUM_CANDIDATES=${9:-"32"}       # 生成候选数量 (默认 32)

PY_ARGS="$PY_ARGS --num_candidates $NUM_CANDIDATES --rollout_batch_size $NUM_CANDIDATES --use_bf16"

[ "$IS_ON_POLICY" == "true" ] && PY_ARGS="$PY_ARGS --on_policy"
PY_ARGS="$PY_ARGS --verbose $IS_VERBOSE"
PY_ARGS="$PY_ARGS --reward_topk $REWARD_TOPK"
if [ "$MODE" != "SFT" ]; then
    # RL 阶段自动设置 dropout 为 0
    PY_ARGS="$PY_ARGS --gen_strategy $GEN_STRATEGY --dropout 0"
fi

# 处理兼容性: 如果传入 "true"，自动转为 "trie_all"
if [ "$USE_TRIE" == "true" ]; then
    USE_TRIE="trie_all"
elif [ "$USE_TRIE" == "false" ]; then
    USE_TRIE="no_trie"
fi
PY_ARGS="$PY_ARGS --use_trie $USE_TRIE"

# --- 4. 辅助函数 (还原原始 Log 路径逻辑) ---
create_log_path() {
    local script_path="$1"
    local mode="$2"
    local on_policy="$3"
    local gen_strategy="$4"
    local use_trie_mode="$5"
    local reward_topk="$6"
    local num_candidates="$7"
    
    local abs_script_path=$(realpath "$script_path")
    local script_dir=$(dirname "$abs_script_path")
    local parent_dir=$(dirname "$script_dir")
    local logs_dir="$parent_dir/logs_rl"
    local script_name=$(basename "$script_path" .py)
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local policy_str=""
    if [[ "$mode" == "GRPO" || "$mode" == "GBPO" || "$mode" == "GUPO" || "$mode" == "GSPO" || "$mode" == "GPPO" || "$mode" == "DUAL_PPO" || "$mode" == "SAPO" || "$mode" == "DAPO" || "$mode" == "DPO" ]]; then
        if [ "$on_policy" == "true" ]; then
            policy_str="_OnPolicy"
        else
            policy_str="_OffPolicy"
        fi
        policy_str="${policy_str}_${gen_strategy}_topk${reward_topk}_K${num_candidates}"
    fi
    
    local trie_suffix="free"
    if [ "$use_trie_mode" == "trie_all" ]; then
        trie_suffix="trie_all"
    elif [ "$use_trie_mode" == "trie_gen" ]; then
        trie_suffix="trie_gen"
    fi
    
    local log_file="$logs_dir/${DATASET}/${script_name}_${mode}${policy_str}_${trie_suffix}_${timestamp}.log"
    mkdir -p "$(dirname "$log_file")"
    echo "$log_file"
}

# --- 5. 启动逻辑 ---
if [ "$RUN_TYPE" == "nohup" ]; then
    LOG_FILE=$(create_log_path "$SCRIPT" "$MODE" "$IS_ON_POLICY" "$GEN_STRATEGY" "$USE_TRIE" "$REWARD_TOPK" "$NUM_CANDIDATES")
    
    echo "🚀 Starting background training..."
    echo "📝 Log: tail -f $LOG_FILE"
    
    nohup torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT $PY_ARGS > "$LOG_FILE" 2>&1 &
    echo "✅ PID: $!"
else
    echo "🚀 Starting foreground training..."
    torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT $PY_ARGS
fi
