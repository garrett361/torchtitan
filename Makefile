LOCAL_BATCH_SIZE=4
STEPS=100
CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_dev_model.toml
ENABLE_WANDB=True
GIT_HASH := $(shell git rev-parse --short HEAD)

# Conditional wandb flag
ifeq ($(ENABLE_WANDB),True)
WANDB_FLAG = --metrics.enable_wandb
else
WANDB_FLAG =
endif

# Common function to run training with different flavors
define run_fsdp
	export WANDB_RUN_ID=$(1)-fsdp-$(GIT_HASH) && \
	export CONFIG_FILE=$(CONFIG_FILE) && \
	./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor $(2) \
		$(WANDB_FLAG)
endef

help:
	@echo "Available targets:"
	@echo "  16B_grouped_mm"
	@echo "  16B_for_loop"
	@echo "  16B_cg_grouped_gemm"

16B_grouped_mm:
	$(call run_fsdp,16B-grouped-mm,16B)

16B_for_loop:
	$(call run_fsdp,16B-for-loop,16B_for_loop)

16B_cg_grouped_gemm:
	$(call run_fsdp,16B-cg-grouped-gemm,16B_cg_grouped_gemm)

