# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX, LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int
    _cached_epochs: float | None

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        cp_rank = (
            parallel_dims.get_mesh("cp").get_local_rank()
            if parallel_dims.cp_enabled
            else 0
        )
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            cp_rank=cp_rank,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Build the collection of model converters. No-op if converters empty
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
        model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )
        model_converters.convert(model)

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()

        # Check if any converter uses quantization (FP8, MX, etc.)
        has_quantization = any(
            isinstance(cc, QuantizationConverter.Config)
            for cc in config.model_converters.converters
        )

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=has_quantization,
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = model_spec.build_loss_fn(
            config.compile, parallel_dims=parallel_dims
        )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both Pipeline Parallel and SPMD-style scaling techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            if not config.checkpoint.create_seed_checkpoint:
                # Skip parallelize_fn for seed checkpoints — nothing from
                # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                model = model_spec.parallelize_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    model_converters=config.model_converters,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                )

            model.to_empty(device=init_device)
            with torch.no_grad():
                # TODO: Change this back to init_weights once
                # autoparallel contains the wrap_init_states
                cast(BaseModel, model).init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0
        self._cached_epochs = None
        self._rank_local_valid_tokens_per_step = 0

        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            attn_cost = input_dict.pop("attn_cost", None)
            if attn_cost is not None:
                if not self.metrics_processor.attn_cost_tracking_enabled:
                    self.metrics_processor.setup_attn_cost_tracking(
                        seq_len=input_dict["input"].shape[-1],
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        device=self.device,
                    )
                self.metrics_processor.record_attn_cost(attn_cost.item())
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.nvalid_tokens_since_last_log += int(
                (labels != IGNORE_INDEX).sum()
            )
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors from input_dict
                    (excluding "input" and "positions"). These are passed to the
                    model forward but are NOT forwarded across pipeline parallel
                    stages.
                - extra_kwargs: Dict of additional keyword arguments for model
                    forward (positions, attention_masks). These ARE forwarded
                    across all pipeline parallel stages.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage. Positions
            always go into extra_kwargs so every stage can apply RoPE correctly.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # extra_kwargs are forwarded to all PP stages; extra_inputs are only
        # available to the first stage.  Positions go into extra_kwargs so
        # every stage can apply RoPE correctly.
        extra_kwargs: dict[str, Any] = {}

        # Resolve positions once: per-document positions for block_causal,
        # sequential positions when CP needs them for shard indexing,
        # or None (model uses sequential RoPE slice by default).
        if isinstance(self.model_config, Decoder.Config):
            layer = self.model_config.layers[0]
            attn_config = layer.attention
        else:
            attn_config = None
        mask_type = getattr(attn_config, "mask_type", "causal")

        positions = extra_inputs.pop("positions", None)
        if mask_type == "block_causal" or "suffix_ids" in extra_inputs:
            extra_kwargs["positions"] = positions
        elif self.parallel_dims.cp_enabled:
            # Sequential positions needed for correct RoPE after CP sharding
            extra_kwargs["positions"] = torch.arange(
                0, inputs.shape[1], dtype=torch.int32, device=self.device
            ).expand(inputs.shape)

        inner_attention = getattr(attn_config, "inner_attention", None)
        if inner_attention is not None:
            from torchtitan.models.common.attention import (
                FA4Attention,
                FlexAttention,
                VarlenAttention,
            )

            if isinstance(
                inner_attention,
                (FlexAttention.Config, VarlenAttention.Config, FA4Attention.Config),
            ):
                assert (
                    self.tokenizer is not None
                ), "tokenizer is required for flex/varlen/fa4 attention"
                # FA4+CP builds its mask post-sharding (needs CP indices),
                # so skip here to avoid a wasted CuTe JIT compilation.
                fa4_cp = self.parallel_dims.cp_enabled and isinstance(
                    inner_attention, FA4Attention.Config
                )
                if not fa4_cp:
                    model = cast(Decoder, self.model_parts[0])
                    extra_kwargs["attention_masks"] = model.get_attention_masks(
                        input_batch=inputs,
                        tokenizer=self.tokenizer,
                        extra_inputs=extra_inputs,
                        positions=positions,
                    )

        if self.parallel_dims.cp_enabled:
            if inner_attention is not None and isinstance(
                inner_attention, FA4Attention.Config
            ):
                # FA4+CP: construct LB once, shard inputs directly (no mask
                # sharding needed), then build FA4Mask from same LB instance.
                from torch.distributed.tensor.experimental._attention import (
                    _context_parallel_shard,
                    _HeadTailLoadBalancer,
                    _PTRRLoadBalancer,
                )

                from torchtitan.models.common.attention import build_fa4_mask

                lb_type = self.config.parallelism.context_parallel_load_balancer
                cp_mesh = self.parallel_dims.get_mesh("cp")
                cp_world_size = cp_mesh.size(0)

                if lb_type == "ptrr":
                    from torch.nn.attention.flex_attention import and_masks

                    from torchtitan.models.common.attention import (
                        create_attention_mask,
                        get_causal_mask_mod,
                        get_document_mask_mod_from_positions,
                    )

                    mask_mods = [get_causal_mask_mod()]
                    if mask_type == "block_causal" and positions is not None:
                        mask_mods.append(
                            get_document_mask_mod_from_positions(positions)
                        )
                    ptrr_block_mask = create_attention_mask(
                        and_masks(*mask_mods),
                        inputs.shape[0],
                        None,
                        inputs.shape[1],
                        inputs.shape[1],
                    )
                    lb = _PTRRLoadBalancer(ptrr_block_mask, cp_world_size)
                elif lb_type == "headtail" or lb_type is None:
                    lb = _HeadTailLoadBalancer(
                        inputs.shape[1], cp_world_size, self.device.type
                    )
                else:
                    raise ValueError(
                        f"FA4 CP: unknown load_balancer_type '{lb_type}'. "
                        f"Must be 'headtail', 'ptrr', or None."
                    )

                if positions is None:
                    raise ValueError(
                        "FA4+CP requires per-document positions; ensure the "
                        "dataloader provides a 'positions' tensor."
                    )
                orig_positions = positions

                inputs, labels, positions = _context_parallel_shard(
                    mesh=cp_mesh,
                    buffers=(inputs, labels, extra_kwargs["positions"]),
                    seq_dims=(1, 1, 1),
                    load_balancer=lb,
                )
                extra_kwargs["positions"] = positions

                shard_indices = lb._generate_indices(restore=False)
                local_seq_len = inputs.shape[1]

                document_ids = None
                if mask_type == "block_causal":
                    from torchtitan.models.common.decoder import Decoder as _Dec

                    document_ids = _Dec._document_ids_from_positions(
                        orig_positions
                    )

                extra_kwargs["attention_masks"] = build_fa4_mask(
                    shard_indices=shard_indices,
                    cp_rank=cp_mesh.get_local_rank(),
                    local_seq_len=local_seq_len,
                    document_ids=document_ids,
                )
            else:
                inputs, labels, extra_kwargs = prepare_context_parallel_input(
                    inputs,
                    labels,
                    extra_kwargs,
                    self.parallel_dims.get_mesh("cp"),
                    self.device,
                    self.config.parallelism.context_parallel_load_balancer,
                )

        # Accumulate after CP sharding so counts reflect the actual unique
        # tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()
        self._rank_local_valid_tokens_per_step += int((labels != IGNORE_INDEX).sum())

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # Rescale PP loss to be "local loss sum / global valid tokens"
                # because each microbatch could have different number of valid tokens
                loss = (torch.sum(torch.stack(losses)) / global_valid_tokens).to(
                    self.device
                )
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                # Compute loss sum (reduction='sum')
                loss_sum = self.loss_fn(pred, labels)

                # Scale the loss by the inverse of the total weight denominator before backward
                # This ensures gradients are properly normalized across all microbatches
                loss = loss_sum / global_valid_tokens

                # need to free pred before bwd to avoid peaking memory
                del pred
                loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        self._rank_local_valid_tokens_per_step = 0
        accumulated_losses = []
        for i, (input_dict, labels) in enumerate(microbatches):
            if parallel_dims.dp_replicate_enabled:
                is_last = (i == len(microbatches) - 1)
                for model in self.model_parts:
                    model.set_requires_all_reduce(is_last)

            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of all the rank-local average losses.
            # local_valid_tokens is the pre-CP-sharding count, identical on all ranks in a CP group,
            # whereas max-loss requires the post-CP-sharding count. This is provided by
            # _rank_local_valid_tokens_per_step, which is accumulated after CP sharding in
            # post_dataloading_process:
            # local_avg_loss = local_loss_sum / rank_local_valid_tokens
            #                = (loss * global_valid_tokens) / rank_local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            rank_local_valid_tokens = torch.tensor(
                self._rank_local_valid_tokens_per_step,
                dtype=torch.int64,
                device=self.device,
            )
            # clamp avoids 0/0 when, e.g.,  a CP shard contains only masked tokens (loss=0)
            local_avg_loss = (
                loss * global_valid_tokens / rank_local_valid_tokens.clamp(min=1)
            )
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(local_avg_loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        stats_fn = getattr(self.dataloader, "get_data_stats", None)
        if stats_fn is None:
            dataset = getattr(self.dataloader, "dataset", None)
            if dataset is not None:
                stats_fn = getattr(dataset, "get_data_stats", None)
        if stats_fn is not None:
            raw = stats_fn()
            n_total = raw["n_total_tokens"]
            n_trained = raw["n_trained_tokens"]
            n_examples = raw["n_examples_packed"]
            # CP ranks all load the same pre-sharding batch, so only sum
            # across DP ranks (batch mesh) to avoid multiplying by CP_DEGREE.
            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                dp_degree = batch_mesh.size()
                n_total = int(
                    dist_utils.dist_sum(
                        torch.tensor(
                            n_total, dtype=torch.int64, device=self.device
                        ),
                        batch_mesh,
                    )
                )
                n_trained = int(
                    dist_utils.dist_sum(
                        torch.tensor(
                            n_trained, dtype=torch.int64, device=self.device
                        ),
                        batch_mesh,
                    )
                )
                n_examples = int(
                    dist_utils.dist_sum(
                        torch.tensor(
                            n_examples, dtype=torch.int64, device=self.device
                        ),
                        batch_mesh,
                    )
                )
                epochs_logged = (
                    dist_utils.dist_min(
                        torch.tensor(
                            raw["epochs"], dtype=torch.float64, device=self.device
                        ),
                        batch_mesh,
                    )
                    if raw["epochs"] is not None
                    else None
                )
            else:
                dp_degree = 1
                epochs_logged = raw["epochs"]
            s = max(self.step, 1)
            extra_metrics.update(
                {
                    "data/total_train_toks": n_trained,
                    "data/total_examples": n_examples,
                    "data/avg_total_toks_per_step": n_total / s,
                    "data/avg_train_toks_per_step": n_trained / s,
                    "data/avg_examples_per_step": n_examples / s,
                    "data/avg_toks_per_example": n_total / max(n_examples, 1),
                    "data/avg_train_toks_per_example": n_trained / max(n_examples, 1),
                    "data/avg_train_token_fraction": n_trained / max(n_total, 1),
                    "data/avg_padding_fraction": 1
                    - n_total
                    / (
                        self.config.training.seq_len
                        * self.config.training.local_batch_size
                        * dp_degree
                        * self.gradient_accumulation_steps
                        * s
                    ),
                }
            )
            self._cached_epochs = epochs_logged
            if epochs_logged is not None:
                extra_metrics["data/epochs"] = epochs_logged
            dataset_mean_length = raw.get("dataset_mean_length")
            if dataset_mean_length and n_examples > 0:
                extra_metrics["dataset/bias_length_ratio"] = (
                    n_total / n_examples / dataset_mean_length
                )
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            total_steps=self.config.training.steps,
            steps_to_next_ckpt=self.checkpointer.interval
            - self.step % self.checkpointer.interval
            if self.checkpointer.enable
            else None,
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)

        if config.training.max_epochs is not None:
            epoch_stats_fn = getattr(self.dataloader, "get_data_stats", None)
            if epoch_stats_fn is None:
                dataset = getattr(self.dataloader, "dataset", None)
                if dataset is not None:
                    epoch_stats_fn = getattr(dataset, "get_data_stats", None)
            if epoch_stats_fn is None:
                raise ValueError(
                    "training.max_epochs requires a dataloader (or dataset) "
                    "that implements get_data_stats()."
                )
            if epoch_stats_fn()["epochs"] is None:
                raise ValueError(
                    "training.max_epochs requires a dataset that tracks epoch boundaries. "
                    "Use a dataset whose get_data_stats() returns a non-None 'epochs' value "
                    "(e.g. ChatDataset over a map-style HuggingFace Dataset, or "
                    "StandardPackingDataset)."
                )

        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                epoch_done = self._epoch_limit_reached()
                self.checkpointer.save(
                    self.step,
                    last_step=(self.step == config.training.steps) or epoch_done,
                )
                if epoch_done:
                    logger.info(
                        f"Stopping: reached max_epochs={config.training.max_epochs} "
                        f"at step {self.step}."
                    )
                    break

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def _epoch_limit_reached(self) -> bool:
        max_epochs = self.config.training.max_epochs
        if max_epochs is None or self._cached_epochs is None:
            return False
        return self._cached_epochs >= max_epochs

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
