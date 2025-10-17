# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.distributed import utils as dist_utils

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class CustomCheckpointManager(CheckpointManager):
    def __init__(
        self,
        hf_storage_reader: type[HuggingFaceStorageReader] = HuggingFaceStorageReader,
        hf_storage_reader_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hf_storage_reader = hf_storage_reader
        self.hf_storage_reader_kwargs = hf_storage_reader_kwargs or {}

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.

        # NOTE: @goon - this is a direct copy of CustomCheckpointManager.dcp_load, just with the HF
        # reader modified.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)

            # TODO: @goon - DELETE
            dist_utils.rank_zero_print(
                f"About to dcp.load with {list(hf_state_dict)=}\n{list(state_dict)=}\n{checkpoint_id=}"
            )
            dcp.load(
                hf_state_dict,
                storage_reader=self.hf_storage_reader(
                    path=checkpoint_id, **self.hf_storage_reader_kwargs
                ),
            )
            # TODO: @goon - DELETE
            dist_utils.rank_zero_print("Done dcp.load")

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            # TODO: @goon - DELETE
            dist_utils.rank_zero_print(
                f"Loading {list(state_dict)=} into {self.states[MODEL]=}"
            )
            # NOTE: @goon - question: is this not erroring out if all keys don't match? Apparently
            # strict = False
            # https://github.com/garrett361/torchtitan/blob/a1c0715c8ef33862d6ec9bdcb302ceedc56a1069/torchtitan/components/checkpoint.py?plain=1#L80
            self.states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)
