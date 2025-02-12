# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.bamba.model import BambaModelArgs, Bamba

__all__ = ["Bamba"]

bamba_configs = {
    "debugmodel": BambaModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        max_seq_len=512,
        chunk_size=64,
    ),
    "debugmodel_mamba_kernels": BambaModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        max_seq_len=512,
        chunk_size=64,
        use_mamba_kernels=True,
    ),
}
