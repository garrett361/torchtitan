SHELL := /bin/bash

HF_HOME=/proj/data-eng/goon/.cache/huggingface
EXP_DIR=/proj/data-eng/goon/long-ctx-sft/experiments

TORCHTITAN_DIR=/proj/data-eng/goon/garrett361/torchtitan-yarn-sft
VENV=$(TORCHTITAN_DIR)/.venv/bin/activate
OPEN_INSTRUCT_DIR=/proj/data-eng/goon/open-instruct
OI_FINETUNE_SCRIPT=$(OPEN_INSTRUCT_DIR)/open_instruct/finetune.py

DATA_PATH=$(TORCHTITAN_DIR)/data
TULU=$(DATA_PATH)/tuluv3_data.jsonl
TOKENIZER_PATH=meta-llama/Llama-3.1-8B
TOKENIZE_NUM_PROC=32

OI_CACHE_DATASET_FLAGS = \
	--dataset_skip_cache False \
	--dataset_cache_mode local \
	--cache_dataset_only True \
	--dataset_local_cache_dir $(DATA_PATH) \
    --tokenizer_name_or_path $(TOKENIZER_PATH) \
	--push_to_hub False \
	--try_launch_beaker_eval_jobs False \
    --chat_template_name tulu \
	--dataset_transform_fn sft_tulu_tokenize_and_truncate_v1 sft_tulu_filter_v1


# NOTE: @goon - no need for sft_tulu_filter_truncated_v1 when running without a max len

help:
	@echo "Choose another target"

tokenize-tulu:
	export HF_HOME=$(HF_HOME) && \
	export BEAKER_ASSIGNED_CPU_COUNT=$(TOKENIZE_NUM_PROC) && \
	source $(VENV) && \
	python3 $(OI_FINETUNE_SCRIPT) \
    --dataset_mixer_list $(TULU) 1.0 \
	$(OI_CACHE_DATASET_FLAGS)

fsdp-8b:
	torchrun --nproc-per-node 8 train.py --job.config_file ./train_configs/llama3_8b.toml

fsdp-8b-sft:
	torchrun --nproc-per-node 8 train.py --job.config_file ./train_configs/llama3_8b.toml
