LOCAL_BATCH_SIZE ?= 2
STEPS ?= 10
CONFIG_FILE=./torchtitan/models/llama3_moe/train_configs/llama3_moe.toml
ENABLE_WANDB ?= False
GIT_HASH := $(shell git rev-parse --short HEAD)
NGPU := $(shell echo "$(CUDA_VISIBLE_DEVICES)" | tr ',' '\n' | wc -l)
LOAD_BALANCE_COEFF ?= 1e-2
FLAVOR ?= 3B_8exp
ARGS ?=
LR ?= 1e-4
WARMUP_STEPS ?= 10
SEQ_LEN ?= 2048
LOG_FREQ ?= 5
DATE := $(shell date "+%Y-%m-%d-%H-%M")
LOG_RANK ?= $$(seq -s, 0 $$((NGPU-1)))

# Conditional wandb flag
ifeq ($(ENABLE_WANDB),True)
WANDB_FLAG = --metrics.enable_wandb
else
WANDB_FLAG =
endif

define run_fsdp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$LOG_RANK && \
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-fsdp-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(3) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.expert_parallel_degree 1 \
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
	export LOG_RANK=$$LOG_RANK && \
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(3) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.expert_parallel_degree $$NGPU \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		--checkpoint.enable $(ENABLE_CKPT) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_ep_pp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$LOG_RANK && \
	export PP=2 && \
	export EP=$$((NGPU/PP)) && \
	export WANDB_RUN_ID=$(1)-$(LR)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-$${PP}pp-$${EP}ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(3) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.expert_parallel_degree $$EP \
		--parallelism.pipeline_parallel_degree $$PP \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		--checkpoint.enable $(ENABLE_CKPT) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_pp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$LOG_RANK && \
	export PP=$(NGPU)&& \
	export WANDB_RUN_ID=$(1)-$(LR)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-$${PP}pp-$${EP}ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(3) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.pipeline_parallel_degree $$PP \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		--checkpoint.enable $(ENABLE_CKPT) \
		$(WANDB_FLAG) \
		$(ARGS)
endef


define run_fsdp_pp
	export NGPU=$(NGPU) && \
	export LOG_RANK=$$LOG_RANK && \
	export PP=2 && \
	export WANDB_RUN_ID=$(1)-$(LR)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-$${PP}pp-$${EP}ep-$(GIT_HASH)-$(DATE) && \
	export CONFIG_FILE=$(3) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(SEQ_LEN) \
		--optimizer.lr $(LR) \
		--metrics.log_freq $(LOG_FREQ) \
		--lr-scheduler.warmup-steps $(WARMUP_STEPS) \
		--model.flavor $(2) \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.pipeline_parallel_degree $$PP \
		--parallelism.fsdp_reshard_after_forward never \
		--custom-args.load-balance-coeff $(LOAD_BALANCE_COEFF) \
		--checkpoint.enable $(ENABLE_CKPT) \
		$(WANDB_FLAG) \
		$(ARGS)
endef

help:
	@echo "Choose another target"

fsdp:
	$(call run_fsdp,debug,"$(FLAVOR)","$(CONFIG_FILE)")

ep:
	$(call run_ep,debug,"$(FLAVOR)","$(CONFIG_FILE)")

ep_pp:
	$(call run_ep_pp,debug,"$(FLAVOR)","$(CONFIG_FILE)")

fsdp_pp:
	$(call run_fsdp_pp,debug,"$(FLAVOR)","$(CONFIG_FILE)")

pp:
	$(call run_pp,debug,"$(FLAVOR)","$(CONFIG_FILE)")

# For comparison
pp_llama4:
	$(call run_pp,debug,"debugmodel","./torchtitan/experiments/llama4/train_configs/debug_model.toml")
