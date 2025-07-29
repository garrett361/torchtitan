LOCAL_BATCH_SIZE=2
STEPS=100
CONFIG_FILE=./torchtitan/models/hybrid_moe/train_configs/debug_model.toml
# ENABLE_WANDB=True
ENABLE_WANDB=False
GIT_HASH := $(shell git rev-parse --short HEAD)

# Conditional wandb flag
ifeq ($(ENABLE_WANDB),True)
WANDB_FLAG = --metrics.enable_wandb
else
WANDB_FLAG =
endif

define run_fsdp
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-fsdp-$(GIT_HASH) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor $(2) \
		$(WANDB_FLAG)
endef


define run_ep
	export WANDB_RUN_ID=$(1)-$(LOCAL_BATCH_SIZE)bs-$(STEPS)step-ep-$(GIT_HASH) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor $(2) \
		--parallelism.expert_parallel_degree 8 \
		--parallelism.fsdp_reshard_after_forward never \
		$(WANDB_FLAG)
endef

help:
	@echo "Choose another target"

fsdp:
	$(call run_fsdp,debug,debugmodel)

fsdp_nope:
	$(call run_fsdp,debug-nope,debugmodel_nope)

ep:
	$(call run_ep,debug,debugmodel)
