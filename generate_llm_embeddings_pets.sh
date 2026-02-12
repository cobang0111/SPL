# Set model_type to be 'llama-3.1-8B-instruct' or 'llama-3.2-3B-instruct'
model_type="llama-3.2-3B-instruct"
#model_type="llama-3.1-8B-instruct"

# Generate Pets dataset
 python -m config.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split test --dataset_size 200 &

 python -m config.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split train --dataset_size 2000 &

 python -m config.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split test --dataset_size 200 &

 python -m config.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split train --dataset_size 2000 &

wait