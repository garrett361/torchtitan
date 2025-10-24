LOCAL_BATCH_SIZE ?= 2
STEPS ?= 100
CONFIG_FILE=./torchtitan/models/llama3gdn/train_configs/debug_model.toml
ENABLE_WANDB ?= False
GIT_HASH := $(shell git rev-parse --short HEAD)
NGPU := $(shell nvidia-smi --list-gpus | wc -l)
LOAD_BALANCE_COEFF ?= 1e-2
FLAVOR ?= debugmodel
ARGS ?=
LR ?= 8e-4
WARMUP_STEPS ?= 200
SEQ_LEN ?= 2048
LOG_FREQ ?= 5
DATE := $(shell date "+%Y-%m-%d-%H-%M")

# NOTE: @goon - Usage: to run a dev-model with, say, n_layers=2 and  num_experts=32 with EP,
# do make ep FLAVOR=n_layers=2|num_experts=32.

# Conditional wandb flag
ifeq ($(ENABLE_WANDB),True)
WANDB_FLAG = --metrics.enable_wandb
else
WANDB_FLAG =
endif

define run_fsdp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$(seq -s, 0 $$((NGPU-1))) && \
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-fsdp-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_fsdp_cp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$(seq -s, 0 $$((NGPU-1))) && \
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-fsdp-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--gdn_args.attn_freq 0 \
		--parallelism.context_parallel_degree $$NGPU \
		--model.flavor $(2) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


fsdp:
	$(call run_fsdp,debug,"$(FLAVOR)")

fsdp-cp:
	$(call run_fsdp_cp,debug,"$(FLAVOR)")
