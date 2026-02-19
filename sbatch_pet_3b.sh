#!/bin/bash
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=train_job
#SBATCH --output=slurm-%j.out

gpustat

echo "### JOB STARTED: $(date)"
echo "### NODE: $(hostname)"
echo "### CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/anaconda3/etc/profile.d/conda.sh && conda activate spl

export HF_HOME=""
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
echo $HF_HOME

export WANDB_MODE=online
export WANDB_PROJECT=SPL_pet
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


model_name='meta-llama/Llama-3.2-3B-Instruct'

model_type=$1

if [ ${model_type} == "vpl" ]
then
python -m config.train_llm_vpl_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/llama-3.2-3B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=vpl \
        --data_subset=both \
        --log_dir="logs/llama-3.2-3B-instruct_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 3072 \
        --decoder_embed_dim 3072 \
        --max_length 1024 \
        --learning_rate 3e-4 \
        --use_annealing True \
        --kl_loss_weight 1e-4 \
        --guiding False \
        --guiding_weight 1e-5 \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --use_last_token_embedding True \
        --up_sampling True \
        --controversial_only False \
        --seed 31 \
        --fast_eval False


# Inverse Autoregressive Flow + Variational Preference Learning (VPL-IAF): ivpl
elif [ ${model_type} == "ivpl" ]
then
python -m config.train_llm_ivpl_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/llama-3.2-3B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=ivpl \
        --data_subset=both \
        --log_dir="logs/llama-3.2-3B-instruct_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 3072 \
        --decoder_embed_dim 3072 \
        --max_length 1024 \
        --learning_rate 3e-4 \
        --use_annealing False \
        --kl_loss_weight 1e-4 \
        --guiding False \
        --guiding_weight 1e-5 \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --use_last_token_embedding True \
        --up_sampling True \
        --controversial_only False \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2 \
        --fast_eval False

# Swap-guided Preference Learning (SPL): spl
elif [ ${model_type} == "spl" ]
then
python -m config.train_llm_spl_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/llama-3.2-3B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=spl \
        --data_subset=both \
        --log_dir="logs/llama-3.2-3B-instruct_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 3072 \
        --decoder_embed_dim 3072 \
        --max_length 1024 \
        --learning_rate 3e-4 \
        --use_annealing True \
        --kl_loss_weight 1e-4 \
        --guiding True \
        --guiding_weight 1e-5 \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --use_last_token_embedding True \
        --up_sampling True \
        --controversial_only False \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2 \
        --fast_eval False

# DPL (Distributional Preference Learning): catergorical / mean_and_variance
# BTL (Bradley-Terry-Luce RLHF): base
else
python -m config.train_llm_preference_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/llama-3.2-3B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=${model_type} \
        --data_subset=both \
        --log_dir="logs/llama-3.2-3B-instruct_simple_pets" \
        --bf16 True \
        --fp16 False \
        --max_length 1024 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --controversial_only False \
        --up_sampling True \
        --gradient_checkpointing True \
        --seed 31
fi

echo "### JOB ENDED: $(date)"