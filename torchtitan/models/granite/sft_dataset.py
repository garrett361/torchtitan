# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Callable

import tyro

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
        sample_processor: Annotated[Callable, tyro.conf.Suppress] = lambda s: s["messages"]

    def __init__(self, config: Config, **kwargs):
        if config.dataset_path and not config.load_dataset_kwargs:
            path = Path(config.dataset_path)
            if path.is_file():
                data_files = str(path)
            else:
                data_files = str(path / "*.jsonl")
            config = replace(
                config,
                dataset_path="json",
                load_dataset_kwargs={"data_files": data_files, "split": "train"},
            )
        super().__init__(config, **kwargs)
