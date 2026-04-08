#!/bin/bash

LOG_DIR="$(pwd)/../logs/process"
mkdir -p "$LOG_DIR"

datasets=("Beauty" "Sports" "Toys")
gpus=(3 4 5)
BASE_DIR=".."

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu=${gpus[$i]}
    
    # 数据来源路径
    DATA_DIR="${BASE_DIR}/data/processed_data/${dataset}/processed_loo"
    # 模型保存路径
    SAVE_DIR="${BASE_DIR}/DIN/checkpoint/${dataset}"

    # 检测机制：如果目录存在且不为空，认为已有 checkpoint，跳过运行
    if [ -d "$SAVE_DIR" ] && [ "$(ls -A "$SAVE_DIR" 2>/dev/null)" ]; then
        echo "✅ $dataset: Checkpoint already exists in $SAVE_DIR, skipping."
        continue
    fi

    mkdir -p "$SAVE_DIR"

    echo "🚀 Starting DIN training for $dataset on GPU $gpu..."
    
    # 使用指定 GPU 后台运行任务，日志存放在 logs/process 目录下
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u "${BASE_DIR}/DIN/functions/train_loo.py" \
      --data_dir_loo "${DATA_DIR}/" \
      --save_path "${SAVE_DIR}" \
      > "$LOG_DIR/train_${dataset}.log" 2>&1 &
done

echo "所有符合条件的训练任务已在后台启动。"
