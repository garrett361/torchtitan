# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch.distributed as dist

from torchtitan.models.llama3_moe.model.args import Llama3MoEModelArgs
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.tools.logging import logger, warn_once


# Modified from Llama3StateDictAdapter
class Llama3MoEStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: Llama3MoEModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # Map from the actual hf names to the actual model weights used for this model.
        self.from_hf_map: dict[str, str | None] = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        is_moe_list = model_args.is_moe_list or [
            False for _ in range(model_args.n_layers)
        ]
        attn_name_weight_map = {
            "q_proj": "wq",
            "k_proj": "wk",
            "v_proj": "wv",
            "o_proj": "wo",
        }
        moe_name_weight_map = {
            "mlp.gate_proj": "moe.experts.w1",
            "mlp.up_proj": "moe.experts.w3",
            "mlp.down_proj": "moe.experts.w2",
        }
        ffn_name_weight_map = {
            "mlp.gate_proj": "feed_forward.w1",
            "mlp.up_proj": "feed_forward.w3",
            "mlp.down_proj": "feed_forward.w2",
        }
        for layer_idx, is_moe in enumerate(is_moe_list):
            self.from_hf_map[
                f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
            ] = None
            self.from_hf_map[f"model.layers.{layer_idx}.input_layernorm.weight"] = (
                f"layers.{layer_idx}.attention_norm.weight"
            )
            self.from_hf_map[
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ] = f"layers.{layer_idx}.ffn_norm.weight"
            for hf_name, titan_name in attn_name_weight_map.items():
                self.from_hf_map[
                    f"model.layers.{layer_idx}.self_attn.{hf_name}.weight"
                ] = f"layers.{layer_idx}.attention.{titan_name}.weight"
            if is_moe:
                for hf_name, titan_name in moe_name_weight_map.items():
                    self.from_hf_map[f"model.layers.{layer_idx}.{hf_name}.weight"] = (
                        f"layers.{layer_idx}.{titan_name}"
                    )
            else:
                for hf_name, titan_name in ffn_name_weight_map.items():
                    self.from_hf_map[f"model.layers.{layer_idx}.{hf_name}.weight"] = (
                        f"layers.{layer_idx}.{titan_name}.weight"
                    )

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Take the (probably sharded) torchtitan model's state dict and re-key it to match HF
        conventions, possibly also changing tensor layouts, if necessary.

        Only used when loading from or saving to an HF ckpt (dcp_load,{save}) and in a conversion
        utility script. NOTE: @goon - we will need separate versions of this function for weight
        loading and HF ckpt saving at some point.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                # NOTE: @goon - added the ability here to only load a portion of the serialized HF
                # model's weights, e.g. if testing out with fewer layers than actually exist in the
                # real model.
                if key not in to_hf_map:
                    warn_once(
                        logger,
                        f"{key=} not found in {list(to_hf_map)=}. Skipping.",
                    )
                    continue
                else:
                    new_key = to_hf_map[key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if "attention.wq.weight" in key:
                    value = self._permute(value, n_heads)
                if "attention.wk.weight" in key:
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Take the (probably sharded) HF model's state dict and re-key it to match torchtitan
        conventions, possibly also changing tensor layouts, if necessary.

        Only used when loading from an HF ckpt (dcp_load) and in a conversion utility script.
        """
        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                new_key = self.from_hf_map[key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if "self_attn.q_proj.weight" in key:
                    value = self._reverse_permute(value, n_heads)
                if "self_attn.k_proj.weight" in key:
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict
