# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama3.parallelize import parallelize_llama


def parallelize_granite(model, **kwargs):
    # GraniteModel has identical attribute structure to Llama3Model (both inherit
    # Decoder) so parallelize_llama applies without modification.
    return parallelize_llama(model, **kwargs)
