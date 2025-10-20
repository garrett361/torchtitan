# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    SUFFIX,
    _HFStorageInfo,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlanner, ReadItem

from torchtitan.models.llama3_moe.model.args import Llama3MoEModelArgs
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


# NOTE: @goon - subclass based on the specific version of the HuggingFaceStorageReader for
# torch==2.10.0.dev20251006+cu126 which corresponds to commit
# 39cdb9bef4be0c181989a777a7b68ef04002d491
# https://github.com/pytorch/pytorch/blob/39cdb9bef4be0c181989a777a7b68ef04002d491/torch/distributed/checkpoint/hf_storage.py?plain=1#L202
# This is also the version of torch which exists in the image
# us.icr.io/cil15-shared-registry/platform/goon-torch-nightly-dev:20251006
class TransformingHuggingFaceStorageReader(HuggingFaceStorageReader):
    def __init__(
        self,
        state_dict: dict[str, Any],
        transform_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        sd_adapter: StateDictAdapter | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state_dict = state_dict
        if sd_adapter is not None:
            self.state_dict = sd_adapter.to_hf(self.state_dict)
        self.tranform_fn = transform_fn

    def _process_read_request(self, f, req: ReadItem, planner: LoadPlanner) -> None:
        """Helper function to process a single read request."""
        # Create slices for each dimension based on offsets and lengths
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        if self.tranform_fn is None:
            tensor = f.get_slice(req.storage_index.fqn)[slices]
        else:
            tensor = f.get_tensor(req.storage_index.fqn)
            tensor = self.tranform_fn(req.storage_index.fqn, tensor)
            tensor = tensor[slices]

        target_tensor = planner.resolve_tensor(req).detach()

        assert target_tensor.size() == tensor.size(), (
            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            f" Shapes: {target_tensor.shape=} vs {tensor.shape=} ({slices=}). "
            f" fqn: {req.storage_index.fqn=}."
        )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def read_metadata(self) -> Metadata:
        from safetensors import safe_open  # type: ignore[import]
        from safetensors.torch import _getdtype  # type: ignore[import]

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with safe_open(safetensor_file, framework="pt") as f:
                extra_metadata = f.metadata()

                dcp_sharding_info = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(
                        extra_metadata.get(CUSTOM_METADATA_KEY)
                    )
                for key in set(self.state_dict) & set(f.keys()):
                    storage_shape = f.get_slice(key).get_shape()
                    storage_dtype = f.get_slice(key).get_dtype()

                    if key in self.state_dict:
                        target_tensor = self.state_dict[key]
                        target_shape = list(target_tensor.shape)
                        target_dtype = target_tensor.dtype
                    else:
                        raise ValueError(f"{key=} not in {list(self.state_dict)=}")

                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(target_shape)

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=target_dtype),
                            size=torch.Size(
                                [
                                    saved + offset
                                    for saved, offset in zip(target_shape, offset)
                                ]
                            ),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(target_shape),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(
                                torch.Size(offset), sizes=torch.Size(target_shape)
                            )
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], target_shape[i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                        )
                    else:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=[0] * len(target_shape)
                        )

                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        shape=torch.Size(storage_shape),
                        dtype=_getdtype(storage_dtype),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        return metadata


class _HFWeightTransform(ABC):
    """
    Base class for implementing transforms of loaded HF safetensor weights prior to loading into
    a torchtitan model.

    Subclasses should be given a name in order to be returnable from get_hf_weight_transform_cls.
    """

    name: str | None = None

    def __init__(
        self, model_args: Llama3MoEModelArgs, hf_to_titan_fqn_map: dict[str, str]
    ) -> None:
        self.hf_to_titan_fqn_map = hf_to_titan_fqn_map
        self.model_args = model_args

    def __call__(self, hf_fqn: str, t: torch.Tensor) -> torch.Tensor:
        titan_fqn = self.hf_to_titan_fqn_map[hf_fqn]
        return self.transform(titan_fqn, t)

    @abstractmethod
    def transform(self, titan_fqn: str, t: torch.Tensor) -> torch.Tensor: ...


class ReplicateMoETransform(_HFWeightTransform):
    """
    Basic replication of FFN weights into MoE experts.  Handles any case where `num_experts *
    moe_inter_dim` is divisible by the hf FFN weight's hidden_dim by stacking the FFN weights and
    reshaping as appropriate.

    Cases:
    1.  moe_inter_dim = hf_hidden_dim: basic replication with each expert weight matching its HF FFN
        counterpart as in 2212.05055.
    2.  moe_inter_dim < hf_hidden_dim: each HF FFN is broken up across experts, and can be re-formed
        by concatenating together hf_hidden_dim // moe_inter_dim consecutive expert weights
        appropriately, as in 2410.07524.
    """

    name = "replicate"

    def transform(self, titan_fqn: str, t: torch.Tensor) -> torch.Tensor:
        if "moe.experts.w" in titan_fqn:
            num_experts = self.model_args.moe_args.num_experts
            moe_inter_dim = self.model_args.moe_inter_dim
            dim = self.model_args.dim
            if titan_fqn.endswith("w1") or titan_fqn.endswith("w3"):
                hidden_dim_hf, dim_hf = t.shape
            elif titan_fqn.endswith("w2"):
                dim_hf, hidden_dim_hf = t.shape
            else:
                raise ValueError(f"{titan_fqn=} does not end with any of (w1, w2, w3)")
            n_groups, remainder = divmod(hidden_dim_hf, moe_inter_dim)
            if remainder:
                raise ValueError(
                    f"{hidden_dim_hf=} must be divisible by {moe_inter_dim=}"
                )

            n_replicas, remainder = divmod(num_experts, n_groups)
            if remainder:
                raise ValueError(f"{n_replicas=} must be divisible by {n_groups=}")
            if dim != dim_hf:
                raise ValueError(
                    f"The MoE and FFN input dims do not match: {dim=} != {dim_hf=}."
                )

            t = torch.stack([t for _ in range(n_replicas)], dim=0).contiguous()

            if n_groups != 1:
                if titan_fqn.endswith("w1") or titan_fqn.endswith("w3"):
                    t = rearrange(t, "r h d-> (r h) d")
                    t = rearrange(t, "(e m) d-> e m d", e=num_experts, m=moe_inter_dim)
                elif titan_fqn.endswith("w2"):
                    t = rearrange(t, "r d h-> (r h) d")
                    t = rearrange(t, "(e m) d-> e d m", e=num_experts, m=moe_inter_dim)
                else:
                    raise ValueError(
                        f"{titan_fqn=} does not end with any of (w1, w2, w3)"
                    )
        return t


class ReplicateShuffleMoETransform(_HFWeightTransform):
    """
    Basic replication + hidden_dim shuffling of FFN weights into MoE experts.  Handles any case
    where `num_experts * moe_inter_dim` is divisible by the hf FFN weight's hidden_dim by stacking
    the FFN weights and reshaping as appropriate.
    """

    name = "replicate_shuffle"

    # TODO: @goon - implement

    def transform(self, titan_fqn: str, t: torch.Tensor) -> torch.Tensor:
        if "moe.experts.w" in titan_fqn:
            num_experts = self.model_args.moe_args.num_experts
            moe_inter_dim = self.model_args.moe_inter_dim
            dim = self.model_args.dim
            if titan_fqn.endswith("w1") or titan_fqn.endswith("w3"):
                hidden_dim_hf, dim_hf = t.shape
                shuffle_dim = 0
            elif titan_fqn.endswith("w2"):
                dim_hf, hidden_dim_hf = t.shape
                shuffle_dim = 1
            else:
                raise ValueError(f"{titan_fqn=} does not end with any of (w1, w2, w3)")
            n_groups, remainder = divmod(hidden_dim_hf, moe_inter_dim)
            if remainder:
                raise ValueError(
                    f"{hidden_dim_hf=} must be divisible by {moe_inter_dim=}"
                )

            n_replicas, remainder = divmod(num_experts, n_groups)
            if remainder:
                raise ValueError(f"{n_replicas=} must be divisible by {n_groups=}")
            if dim != dim_hf:
                raise ValueError(
                    f"The MoE and FFN input dims do not match: {dim=} != {dim_hf=}."
                )

            # Ensure that the same shuffles occur on every rank:
            # TODO: @goon - in principle this should already be handled by `set_determinism`.  Check
            # and also ensure this call does mess anything up.
            torch.manual_seed(42)
            replicated_shuffled_t_list = [t]
            for _ in range(n_replicas - 1):
                idx = torch.randperm(hidden_dim_hf)
                if shuffle_dim == 0:
                    replicated_shuffled_t_list.append(t[idx])
                elif shuffle_dim == 1:
                    replicated_shuffled_t_list.append(t[:, idx])
                else:
                    raise ValueError

            t = torch.stack(replicated_shuffled_t_list, dim=0).contiguous()

            if n_groups != 1:
                if titan_fqn.endswith("w1") or titan_fqn.endswith("w3"):
                    t = rearrange(t, "r h d-> (r h) d")
                    t = rearrange(t, "(e m) d-> e m d", e=num_experts, m=moe_inter_dim)
                elif titan_fqn.endswith("w2"):
                    t = rearrange(t, "r d h-> (r h) d")
                    t = rearrange(t, "(e m) d-> e d m", e=num_experts, m=moe_inter_dim)
                else:
                    raise ValueError(
                        f"{titan_fqn=} does not end with any of (w1, w2, w3)"
                    )
        return t


def get_hf_weight_transform_cls(name: str) -> type[_HFWeightTransform]:
    transform_map = {
        sc.name: sc for sc in _HFWeightTransform.__subclasses__() if hasattr(sc, "name")
    }
    return transform_map[name]
