<h1 align="center">Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback (ICLR 2026)</h1>

<p align="center"> Gihoon Kim<sup>1</sup> and Euntai Kim<sup>1, 2</sup> </p>

<p align="center"> <sup>1</sup> Yonsei University, <sup>2</sup> Korea Institute of Science and Technology </p>

## 1. Introduction
This repository provides the implementation of [Swap-guided Preference Learning (SPL)](https://openreview.net/forum?id=nc28mSbyVG).

## 2. Architecture
The architecture is shown below.
<p align="center">
  <img src="fig/spl.png" alt="Overview of SPL" width="1000"/>
</p>

## 3. Implementation

### ✨Environment Setting
Clone this repository and run:

```bash
conda create -n spl python=3.10
conda activate spl
pip install -r requirements.txt
```
### ✨Dataset Generation
Run the following commands with the model of your choice.
You may change the model identifiers in the script files if needed (default: Llama 3 3B).
You need an approved Hugging Face access token to download Llama 3 the first time.
```bash
# Put your authorized Hugging Face token
huggingface-cli login
```

And then, follow below command to generate datasets.
```bash
# For Pets (Dataset size: 3B ≈ 3GB, 8B ≈ 4GB)
bash generate_llm_embeddings_pets.sh

# For UF-P-2 (Dataset size: 3B ≈ 43GB, 8B ≈ 56GB)
python -m config.data_utils.ultrafeedback_augment -a 84 -n P
bash generate_llm_embeddings_UF_P_2.sh

# For UF-P-4 (Dataset size: 3B ≈ 61GB, 8B ≈ 79GB)
python -m config.data_utils.ultrafeedback_augment -a single -n P_4 -c
bash generate_llm_embeddings_UF_P_4.sh

```

(Optional) When you have problem with 
```bash
KeyError: 'type'
```
Put "type": "llama3" to config.json in hugging face cache transformers/model_name/snapshots. 


### ✨Simple Experiments

You can evaluate SPL on the generated datasets, alongside all baseline models included in the paper.

For example:

- **run_pets.sh** runs SPL on the Pets dataset with Llama-3.2-3B-instruct.

- **run_p4.sh** runs SPL on the UF-P-4 dataset with Llama-3.2-3B-instruct.
```bash
# SPL in Pets
bash run_pets.sh
# SPL in UF-P-4
bash run_p4.sh 
```
Additional examples for other datasets and model sizes can be found in the corresponding sbatch_[data]_[model_size].sh files.

You can check all the result metrics in Weights & Biases (wandb).
For first-time use, you’ll need to enter your wandb API key.

## Citation
If you find our work useful, please cite:
```bib
@inproceedings{
  kim2026swapguided,
  title={Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback},
  author={Gihoon Kim and Euntai Kim},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=nc28mSbyVG}
}
```