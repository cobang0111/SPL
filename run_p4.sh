export WANDB_MODE=online
export WANDB_PROJECT=SPL_p4

# huggingface original name
model_name='meta-llama/Llama-3.2-3B-Instruct'

python -m config.train_llm_spl_model \
        --model_name=${model_name} \
        --data_path="data/P_4_survey_16/llama-3.2-3B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=spl \
        --data_subset=all \
        --log_dir="logs/llama-3.2-3B-instruct_P_4_survey_16" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 3072 \
        --decoder_embed_dim 3072 \
        --max_length 1024 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --guiding True \
        --guiding_weight 1e-5 \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets single \
        --use_last_token_embedding True \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2 \
        --fast_eval False