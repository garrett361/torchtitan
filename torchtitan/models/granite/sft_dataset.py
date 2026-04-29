# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.hf_datasets.text_datasets import ChatDataLoader, ChatDataset


class GraniteSFTDataset(ChatDataset):
    """ChatDataset subclass used by GraniteSFTDataLoader.

    The base class now handles general multi-turn masking; this subclass
    exists solely so GraniteSFTDataLoader can instantiate the right class
    via its _dataset_class hook.
    """


class GraniteSFTDataLoader(ChatDataLoader):
    """ChatDataLoader that instantiates GraniteSFTDataset."""

    _dataset_class = GraniteSFTDataset

    @dataclass(kw_only=True, slots=True)
    class Config(ChatDataLoader.Config):
        pass  # inherits all fields; this definition sets _owner = GraniteSFTDataLoader
