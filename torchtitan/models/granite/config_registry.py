# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from . import model_registry
from .pretokenized_dataset import GranitePreTokenizedDataLoader
from .sft_dataset import GraniteSFTDataLoader


def granite_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def granite_sft_debugmodel() -> Trainer.Config:
    """SFT debug config with Granite debugmodel and local test data."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel", attn_backend="flex"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=GraniteSFTDataLoader.Config(
            dataset_path="tests/assets/sft_test/data.json",
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def granite_sft_pretokenized_debugmodel() -> Trainer.Config:
    """SFT debug config using pre-tokenized test assets."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel", attn_backend="flex"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=GranitePreTokenizedDataLoader.Config(
            dataset_path="tests/assets/pretok_truncate_last",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def granite_4_1_8b_sft() -> Trainer.Config:
    """SFT config for granite-4.1-8b on raw JSONL data.

    Requires --hf-assets-path and --dataloader.dataset-path on CLI.
    """
    return Trainer.Config(
        model_spec=model_registry("8B", attn_backend="flex"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.9,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=GraniteSFTDataLoader.Config(dataset_path=""),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def granite_4_1_8b_sft_pretokenized() -> Trainer.Config:
    """SFT config for granite-4.1-8b using pre-tokenized Arrow shards.

    Requires --hf-assets-path and --dataloader.dataset-path on CLI.
    dataset-path may be a single directory containing manifest.json, or a
    comma-separated list of such directories to merge into a single training pool.
    """
    return Trainer.Config(
        model_spec=model_registry("8B", attn_backend="flex"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.9,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=GranitePreTokenizedDataLoader.Config(dataset_path=""),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def granite_4_1_8b_base() -> Trainer.Config:
    """Pretraining/SFT config for granite-4.1-8b-base.

    Requires --hf-assets-path on CLI.
    """
    return Trainer.Config(
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.9,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def granite_debugmodel_float8() -> Trainer.Config:
    config = granite_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
            ),
        ],
    )
    return config


def granite_debugmodel_float8_rowwise() -> Trainer.Config:
    config = granite_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[Float8LinearConverter.Config(recipe_name="rowwise")],
    )
    return config



def granite_4_1_8b_sft_pretokenized_float8_filteroutput() -> Trainer.Config:
    config = granite_4_1_8b_sft_pretokenized()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output"],
            ),
        ],
    )
    return config


def granite_4_1_8b_sft_pretokenized_float8_filteroutput_autokn() -> Trainer.Config:
    config = granite_4_1_8b_sft_pretokenized()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output", "auto_filter_small_kn"],
            ),
        ],
    )
    return config


def granite_4_1_8b_sft_pretokenized_float8_rowwise() -> Trainer.Config:
    config = granite_4_1_8b_sft_pretokenized()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(recipe_name="rowwise"),
        ],
    )
    return config


def granite_4_1_8b_sft_pretokenized_fa4() -> Trainer.Config:
    """SFT config for granite-4.1-8b with FA4 attention backend."""
    config = granite_4_1_8b_sft_pretokenized()
    config.model_spec = model_registry("8B", attn_backend="fa4")
    return config


def granite_4_1_8b_sft_pretokenized_fa4_float8_filteroutput() -> Trainer.Config:
    """SFT config for granite-4.1-8b with FA4 + tensorwise float8."""
    config = granite_4_1_8b_sft_pretokenized_fa4()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output"],
            ),
        ],
    )
    return config
