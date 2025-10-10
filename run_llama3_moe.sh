#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Set log folder
LOGDIR_ROOT=${LOGDIR_ROOT:-"/gpfs/afasoli/logs/llama3moe"}
DATE=$(date '+%Y%m%d_%H%M%S')
LOGDIR=${LOGDIR_ROOT}/${DATE}
mkdir -p $LOGDIR
cp $0 $LOGDIR
cp "./torchtitan/models/__init__.py" $LOGDIR

# !!! temporary hardcoding of NGPU and config for DEBUGGING
export NGPU=4
export CONFIG_FILE="./torchtitan/models/llama3_moe/train_configs/llama3_moe.toml"
cp $CONFIG_FILE $LOGDIR

# Define env var
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# Run
PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@" | tee ${LOGDIR}/debug.log
