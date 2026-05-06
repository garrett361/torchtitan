# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

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
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
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
            manifest_path="tests/assets/pretok_truncate_last/manifest.json",
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
    """SFT config for granite-4.1-8b on GLM-5.1 Reasoning with thinking template.

    Requires HF_ASSETS_PATH and DATA_PATH set in the environment or a .env file at the
    repo root.
    """
    from dotenv import load_dotenv

    load_dotenv()
    ckpt_path = os.getenv("HF_ASSETS_PATH")
    data_path = os.getenv("DATA_PATH")
    for name, val in [
        ("HF_ASSETS_PATH", ckpt_path),
        ("DATA_PATH", data_path),
    ]:
        if val is None:
            raise ValueError(f"{name} not set. Add it to .env or export it before running.")

    return Trainer.Config(
        hf_assets_path=ckpt_path,
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
        dataloader=GraniteSFTDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": str(Path(data_path) / "*.jsonl"),
                "split": "train",
            },
            sample_processor=lambda s: s["messages"],
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def granite_4_1_8b_sft_pretokenized() -> Trainer.Config:
    """SFT config for granite-4.1-8b using pre-tokenized Arrow shards.

    Requires HF_ASSETS_PATH and PRETOK_DATA_PATH set in the environment or a
    .env file at the repo root. PRETOK_DATA_PATH may be a single directory
    containing manifest.json, or a comma-separated list of such directories
    to merge into a single training pool.
    """
    from dotenv import load_dotenv

    load_dotenv()
    ckpt_path = os.getenv("HF_ASSETS_PATH")
    pretok_data_path = os.getenv("PRETOK_DATA_PATH")
    for name, val in [
        ("HF_ASSETS_PATH", ckpt_path),
        ("PRETOK_DATA_PATH", pretok_data_path),
    ]:
        if val is None:
            raise ValueError(f"{name} not set. Add it to .env or export it before running.")

    paths = [p.strip() for p in pretok_data_path.split(",") if p.strip()]
    manifest_path_str = ",".join(str(Path(p) / "manifest.json") for p in paths)

    return Trainer.Config(
        hf_assets_path=ckpt_path,
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
        dataloader=GranitePreTokenizedDataLoader.Config(
            manifest_path=manifest_path_str,
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def granite_4_1_8b_base() -> Trainer.Config:
    """Pretraining/SFT config for granite-4.1-8b-base.

    Requires HF_ASSETS_PATH set in the environment or a .env file at the repo root
    pointing to the HF checkpoint directory.
    """
    from dotenv import load_dotenv

    load_dotenv()
    ckpt_path = os.getenv("HF_ASSETS_PATH")
    if ckpt_path is None:
        raise ValueError(
            "HF_ASSETS_PATH not set. Add it to .env or export it before running."
        )

    return Trainer.Config(
        hf_assets_path=ckpt_path,
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


def granite_4_1_8b_sft_pretokenized_float8_rowwise() -> Trainer.Config:
    config = granite_4_1_8b_sft_pretokenized()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(recipe_name="rowwise"),
        ],
    )
    return config
