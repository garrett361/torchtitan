LOCAL_BATCH_SIZE=2
# STEPS=500
STEPS=100
CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_dev_model.toml
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

fsdp_fl:
	$(call run_fsdp,16B-for-loop,16B_for_loop)

fsdp_gm:
	$(call run_fsdp,16B-grouped-mm,16B)

fsdp_cg:
	$(call run_fsdp,16B-cg-grouped-gemm,16B_cg_grouped_gemm)

ep_fl:
	$(call run_ep,16B-for-loop,16B_for_loop)

ep_gm:
	$(call run_ep,16B-grouped-mm,16B)

ep_cg:
	$(call run_ep,16B-cg-grouped-gemm,16B_cg_grouped_gemm)

