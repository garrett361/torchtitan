import pytest
import torch
import torch.distributed as dist

from dtest import DTest
import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional
from torchtitan.distributed import ParallelDims

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config import TORCH_DTYPE_MAP, ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.attention import init_attention_mask
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

from torchtitan.models.llama3_moe.custom_args import JobConfig
from torchtitan.models.llama3_moe import Transformer, TransformerModelArgs, llama3_moe_configs, parallelize_llama_moe, Llama3MoEStateDictAdapter, CustomCheckpointManager, TransformingHuggingFaceStorageReader
from copy import deepcopy


class TestHFReader(DTest):
    hf_assets_path = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"
    seqlen=64
    bsz=1
    """
    Test loading correctness
    """
    def test_non_moe_load_equivalence(self) -> None:
        model_args = llama3_moe_configs["3B"] 
        job_config = JobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Transformer(model_args)
        model_copy = deepcopy(model)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )


        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_copy = parallelize_llama_moe(model_copy, parallel_dims, job_config)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        model_copy.to_empty(device=self.device)
        with torch.no_grad():
            model_copy.init_weights(buffer_device=None)

        checkpointer = CheckpointManager(
            dataloader=None,
            model_parts=[model],
            optimizers=None, # HACK: @goon - ok to set to None for initial load
            lr_schedulers=None, # HACK: @goon - ok to set to None for initial load
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            base_folder="",
            ft_manager=None,

        )
        checkpointer.load()


        custom_checkpointer = CustomCheckpointManager(
            dataloader=None,
            model_parts=[model_copy],
            optimizers=None, # HACK: @goon - ok to set to None for initial load
            lr_schedulers=None, # HACK: @goon - ok to set to None for initial load
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            base_folder="",
            ft_manager=None,

        )
        custom_checkpointer.load()
        torch.manual_seed(42 + dist.get_rank())
        with torch.no_grad():
            inputs = torch.randint(model_args.vocab_size, size=(self.bsz, self.seqlen),
                                   device=self.device)
            out = model(inputs)
            out_copy = model(inputs)
            torch.testing.assert_close(out, out_copy)

