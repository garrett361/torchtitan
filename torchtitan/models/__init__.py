# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Import the built-in models here so that the corresponding register_model_spec()
# will be called.
import torchtitan.models.llama  # noqa: F401


model_name_to_tokenizer = {"llama3": "tiktoken"}
from torchtitan.models.llama import llama3_configs, Transformer
from torchtitan.models.bamba import bamba_configs, Bamba

models_config = {
    "llama3": llama3_configs,
    "bamba": bamba_configs,
}

model_name_to_cls = {"llama3": Transformer, "bamba": Bamba}

model_name_to_tokenizer = {
    "llama3": "tiktoken",
    "bamba": "tiktoken",
}
