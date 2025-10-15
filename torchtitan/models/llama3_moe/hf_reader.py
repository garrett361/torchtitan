from collections.abc import Callable

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.planner import LoadPlanner, ReadItem


# NOTE: @goon - subclass based on the specific version of the HuggingFaceStorageReader for
# torch==2.10.0.dev20251006+cu126 which corresponds to commit 39cdb9bef4be0c181989a777a7b68ef04002d491
# https://github.com/pytorch/pytorch/blob/39cdb9bef4be0c181989a777a7b68ef04002d491/torch/distributed/checkpoint/hf_storage.py?plain=1#L202
class TransformingHuggingFaceStorageReader(HuggingFaceStorageReader):
    def __init__(
        self,
        transform_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
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
            tensor = self.tranform_fn(req.storage_index.fqn, tensor)[slices]

        target_tensor = planner.resolve_tensor(req).detach()

        assert target_tensor.size() == tensor.size(), (
            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
        )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)
