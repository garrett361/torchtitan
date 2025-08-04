# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.models.deepseek_v3.model.moe import (  # noqa: F401
    FeedForward,
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)


# TODO: @goon - rewrite MoE/FeedForward classes to consolidate w1, w3 into single weight.
# Also see https://github.com/pytorch/torchtitan/commit/ed288bc9f28700b992cb7e50465648cc21aced28
