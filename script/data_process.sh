#!/bin/bash

LOG_DIR="$(pwd)/../logs/process"
mkdir -p "$LOG_DIR"

datasets=("Beauty" "Sports" "Toys")
gpus=(0 1 2)

# 1. 预处理数据 (generate_sasrec_files.py)
for dataset in "${datasets[@]}"; do
    OUT_DIR="../data/processed_data/${dataset}/processed_loo"
    
    if [ -d "$OUT_DIR" ]; then
        echo "$dataset: $OUT_DIR exists, skipping."
    else
        echo "Generating SASRec files for $dataset..."
        mkdir -p "$OUT_DIR"
        python -u "../data/process/generate_sasrec_files.py" \
          --inter_file "../data/raw_data/${dataset}/${dataset}.inter.json" \
          --output_dir "$OUT_DIR" \
          --max_seq_len 20
    fi
done

# 2. 生成 Embedding (generate_emb.py)
cd ../data/process/RQ-VAE || exit

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu=${gpus[$i]}
    emb_file="../../../data/raw_data/$dataset/${dataset}.emb-llama-td.npy"

    if [ -f "$emb_file" ]; then
        echo "$dataset: $emb_file exists, skipping."
    else
        CUDA_VISIBLE_DEVICES=$gpu nohup python -u generate_emb.py \
            --root "../../../data/raw_data/$dataset" \
            --dataset "$dataset" \
            --plm_checkpoint "../../../models/llama3" \
            > "$LOG_DIR/${dataset}_step.log" 2>&1 &
    fi
done

echo "Waiting for embedding generation to complete..."
wait

# 3. 运行 RQ-VAE 训练 (main.py)
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu=${gpus[$i]}
    CKPT_DIR="../../../data/processed_data/${dataset}/rq_ckpt"
    DATA_PATH="../../../data/raw_data/${dataset}/${dataset}.emb-llama-td.npy"

    if [ -d "$CKPT_DIR" ] && [ "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
        echo "$dataset: $CKPT_DIR exists and not empty, skipping."
    else
        echo "Starting $dataset RQ-VAE training on GPU $gpu..."
        mkdir -p "$CKPT_DIR"
        nohup python -u ./main.py \
          --device "cuda:$gpu" \
          --data_path "$DATA_PATH" \
          --alpha 0 \
          --sk_epsilons 0.02 0.02 0.025 0.03 \
          --beta 0 \
          --ckpt_dir "$CKPT_DIR" \
          > "$LOG_DIR/${dataset}_main.log" 2>&1 &
    fi
done

echo "Waiting for RQ-VAE training to complete..."
wait

# 4. 生成 SID (generate_sid.py)
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu=${gpus[$i]}
    OUT_DIR="../../../data/processed_data/${dataset}"
    
    if [ -f "$OUT_DIR/${dataset}_llama.pkl" ] && [ -f "$OUT_DIR/${dataset}_llama.npy" ]; then
        echo "$dataset: SID files exist, skipping."
    else
        CKPT_BASE="$OUT_DIR/rq_ckpt"
        LATEST_DIR=$(ls -td "$CKPT_BASE"/*/ 2>/dev/null | head -n 1)
        
        if [ -n "$LATEST_DIR" ] && [ -f "${LATEST_DIR}best_collision_model.pth" ]; then
            echo "Generating SID for $dataset using latest checkpoint: $LATEST_DIR"
            CUDA_VISIBLE_DEVICES=$gpu nohup python -u ./generate_sid.py \
              --ckpt_path "${LATEST_DIR}best_collision_model.pth" \
              --output_dir "$OUT_DIR" \
              > "$LOG_DIR/${dataset}_sid.log" 2>&1 &
        else
            echo "$dataset: No valid checkpoint found in $CKPT_BASE, skipping SID generation."
        fi
    fi
done

echo "All tasks submitted and waited where necessary."
