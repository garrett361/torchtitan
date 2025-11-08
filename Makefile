SHELL := /bin/bash

HF_HOME=/proj/data-eng/goon/.cache/huggingface
EXP_DIR=/proj/data-eng/goon/long-ctx-sft/experiments

TORCHTITAN_DIR=/proj/data-eng/goon/garrett361/torchtitan-yarn-sft
OI_VENV=/proj/data-eng/goon/open-instruct/.venv/bin/activate
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
	--dataset_transform_fn sft_tulu_tokenize_and_truncate_v1 sft_tulu_filter_v1 \
	--get_tokenizer_fn get_tokenizer_tulu_no_pad_tok_addition

SFT_DATA_PATH=/proj/data-eng/goon/garrett361/torchtitan-yarn-sft/data/14c47a6219
LLAMA_3_8B_DCP_PATH=/proj/data-eng/goon/garrett361/torchtitan-yarn-sft/data/llama_3_8b_dcp

# NOTE: @goon - no need for sft_tulu_filter_truncated_v1 when running without a max len

help:
	@echo "Choose another target"

tokenize-tulu:
	export HF_HOME=$(HF_HOME) && \
	export BEAKER_ASSIGNED_CPU_COUNT=$(TOKENIZE_NUM_PROC) && \
	source $(OI_VENV) && \
	python3 $(OI_FINETUNE_SCRIPT) \
    --dataset_mixer_list $(TULU) 1.0 \
	$(OI_CACHE_DATASET_FLAGS)

fsdp-8b:
	torchrun --nproc-per-node 8 train.py --job.config_file ./train_configs/llama3_8b.toml

# NOTE: @goon
# -  must enable_checkpoint to have loading work
# -  context_parallel_degree=world_size disables CP!
# -  want huge max norm for sum loss
fsdp-8b-sft:
	torchrun --nproc-per-node 8 train.py --job.config_file ./train_configs/llama3_8b.toml \
	    --training.batch_size 1 \
		--dataset.datasets $(SFT_DATA_PATH) \
		--dataset.dataset_weights 1.0 \
		--model.tokenizer_path meta-llama/Llama-3.1-8B \
		--training.sum_loss \
		# --checkpoint.warm_start_ckpt_path $(LLAMA_3_8B_DCP_PATH) \
		# --checkpoint.enable_checkpoint

fsdp-8b-sft-cp:
	torchrun --nproc-per-node 8 train.py --job.config_file ./train_configs/llama3_8b.toml \
		--dataset.dataset_weights 1.0 \
		--dataset.datasets $(SFT_DATA_PATH) \
		--experimental.context_parallel_degree 4 \
		--model.tokenizer_path meta-llama/Llama-3.1-8B \
		--optimizer.lr 1e-5 \
		--training.batch_size 1 \
		--training.warmup_steps 30 \
		--training.max_norm 1e20 \
		--training.seq_len 32768 \
		--training.epochs 0.05 \
		--training.sum_loss \
		# --training.debug \
		# --checkpoint.warm_start_ckpt_path $(LLAMA_3_8B_DCP_PATH) \
		# --checkpoint.enable_checkpoint

