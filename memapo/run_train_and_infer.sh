#!/bin/bash
set -euo pipefail

# ============ 环境 ============
source activate textgrad_dev
cd /home/admin/workspace/aop_lab/app_source/apo/memapo

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY=
export OPENAI_BASE_URL=
export WHALE_API_KEY=
export WHALE_BASE_URL=

# ============ 公共参数 ============
correct_threshold=0.3
embed_model=Qwen3-Embedding-8B
llm_model=gpt-4o-mini-0718

# ============ 训练参数 ============
train_dataset=math_group
train_exp_name=${train_dataset}_${llm_model}_${embed_model}_csim_${correct_threshold}_hope_luck

# ============ 推理参数 ============
infer_datasets=(agieval_aqua agieval_gaokao_math)   # 可按需增减
num_workers=2

# ============ Step 1: 训练 ============
echo "========== [Step 1] Training on ${train_dataset} =========="
python train.py \
    --dataset ${train_dataset} \
    --embed_model ${embed_model} \
    --llm_model ${llm_model} \
    --exp_name ${train_exp_name} \
    --correct_threshold ${correct_threshold}

# train.py 会自动写 logs/latest_checkpoint_{exp_name}.txt
checkpoint=$(cat /home/admin/workspace/aop_lab/app_source/apo/memapo/logs/latest_checkpoint_${train_exp_name}.txt)
echo "Found checkpoint: ${checkpoint}"

# ============ Step 2: 推理（可跑多个数据集） ============
correct_threshold=0.1
for infer_dataset in "${infer_datasets[@]}"; do
    infer_exp_name=${train_dataset}_${infer_dataset}_${llm_model}_${embed_model}_csim_${correct_threshold}_hope_luck
    echo "========== [Step 2] Inference on ${infer_dataset} =========="
    python infer.py \
        --dataset ${infer_dataset} \
        --exp_name ${infer_exp_name} \
        --checkpoint ${checkpoint} \
        --correct_threshold ${correct_threshold} \
        --num_workers ${num_workers}
done

echo "========== All done! =========="
