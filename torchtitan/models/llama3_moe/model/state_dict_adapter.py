# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any
from warnings import warn

from torchtitan.distributed import utils as dist_utils
from torchtitan.models.llama3_moe.model.args import TransformerModelArgs
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

logger = logging.getLogger()


# Modified from Llama3StateDictAdapter
class Llama3MoEStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: TransformerModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        # Map from the all serialized HF llama keys to
        self.from_hf_map_vanilla_llama = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Map from the actual hf names to the actual model weights used for this model.
        self.from_hf_map: dict[str, str | None] = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        is_moe_list = model_args.is_moe_list or [
            False for _ in range(model_args.n_layers)
        ]
        for layer_idx, is_moe in enumerate(is_moe_list):
            self.from_hf_map[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (
                f"layers.{layer_idx}.attention.wq.weight"
            )
            self.from_hf_map[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (
                f"layers.{layer_idx}.attention.wk.weight"
            )
            self.from_hf_map[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (
                f"layers.{layer_idx}.attention.wv.weight"
            )
            self.from_hf_map[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (
                f"layers.{layer_idx}.attention.wo.weight"
            )
            self.from_hf_map[
                f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
            ] = None
            self.from_hf_map[f"model.layers.{layer_idx}.input_layernorm.weight"] = (
                f"layers.{layer_idx}.attention_norm.weight"
            )
            self.from_hf_map[
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ] = f"layers.{layer_idx}.ffn_norm.weight"
            if is_moe:
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (
                    f"layers.{layer_idx}.moe.experts.w1.weight"
                )
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (
                    f"layers.{layer_idx}.moe.experts.w3.weight"
                )
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = (
                    f"layers.{layer_idx}.moe.experts.w2.weight"
                )
            else:
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (
                    f"layers.{layer_idx}.feed_forward.w1.weight"
                )
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (
                    f"layers.{layer_idx}.feed_forward.w3.weight"
                )
                self.from_hf_map[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = (
                    f"layers.{layer_idx}.feed_forward.w2.weight"
                )
            dist_utils.rank_zero_print(f"{self.from_hf_map=}")

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
        utility script.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map_vanilla_llama.items()}

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
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                # NOTE: @goon - added the ability here to only load a portion of the serialized HF
                # model's weights, e.g. if testing out with fewer layers than actually exist in the
                # real model.
                if abstract_key not in to_hf_map:
                    warn(f"{abstract_key=} not found in {list(to_hf_map)=}. Skipping.")
                    continue
                else:
                    new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def _to_hf_orig(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Take the (probably sharded) torchtitan model's state dict and re-key it to match HF
        conventions, possibly also changing tensor layouts, if necessary.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map_vanilla_llama.items()}

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
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                # NOTE: @goon - added the ability here to only load a portion of the serialized HF
                # model's weights, e.g. if testing out with fewer layers than actually exist in the
                # real model.
                if abstract_key not in to_hf_map:
                    warn(f"{abstract_key=} not found in {list(to_hf_map)=}. Skipping.")
                    continue
                else:
                    new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
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
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map_vanilla_llama[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map_vanilla_llama[key]

            state_dict[new_key] = value
        return state_dict
