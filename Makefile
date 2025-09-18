LOCAL_BATCH_SIZE ?= 2
STEPS ?= 100
CONFIG_FILE=./torchtitan/models/hybrid_moe/train_configs/debug_model.toml
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
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_ep
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$(seq -s, 0 $$((NGPU-1))) && \
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.expert_parallel_degree $$NGPU \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_ep_pp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$(seq -s, 0 $$((NGPU-1))) && \
	export PP=2 && \
	export EP=$$((NGPU/PP)) && \
	export WANDB_RUN_ID=$(1)-$(LR)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-$${PP}pp-$${EP}ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.expert_parallel_degree $$EP \
		--parallelism.pipeline_parallel_degree $$PP \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		$(WANDB_FLAG) \
		$(ARGS)
endef

help:
	@echo "Choose another target"

fsdp:
	$(call run_fsdp,debug,"$(FLAVOR)")

ep:
	$(call run_ep,debug,"$(FLAVOR)")

ep_pp:
	$(call run_ep_pp,debug,"$(FLAVOR)")
