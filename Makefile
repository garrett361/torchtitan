LOCAL_BATCH_SIZE=1
STEPS=100
CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_dev_model.toml

16B:
	export CONFIG_FILE=$(CONFIG_FILE) && ./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor 16B

16B_for_loop:
	export CONFIG_FILE=$(CONFIG_FILE) && ./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor 16B_for_loop

16B_cg_grouped_gemm:
	export CONFIG_FILE=$(CONFIG_FILE) && ./run_train.sh \
		--training.local_batch_size $(LOCAL_BATCH_SIZE) \
		--training.steps $(STEPS) \
		--model.flavor 16B_cg_grouped_gemm
