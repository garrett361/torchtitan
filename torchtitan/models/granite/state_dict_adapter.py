# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import GraniteModel


class GraniteStateDictAdapter(StateDictAdapter):
    """Converts between torchtitan and HuggingFace Granite weight layouts.

    Identical to Llama3StateDictAdapter except:
    - ``lm_head.weight`` is absent from the HF checkpoint (weight tying);
      ``from_hf`` synthesizes it from ``model.embed_tokens.weight``.
    - HF Granite uses the same interleaved Q/K RoPE layout as LLaMA3 HF, so
      ``_permute`` (forward and reverse) is correct and must be kept.
      (Validated empirically: logits match ``transformers.GraniteForCausalLM``
      within bfloat16 numerical noise, max absolute diff ≈ 0.48 over 40 layers.)
    """

    def __init__(
        self,
        model_config: GraniteModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention.qkv_linear, FusedQKVLinear.Config
        )

        if self.fuse_qkv:
            self.from_hf_map = {
                "model.embed_tokens.weight": "tok_embeddings.weight",
                "model.layers.{}.self_attn.q_proj.weight": None,
                "model.layers.{}.self_attn.k_proj.weight": None,
                "model.layers.{}.self_attn.v_proj.weight": None,
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
        else:
            self.from_hf_map = {
                "model.embed_tokens.weight": "tok_embeddings.weight",
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
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

    def _permute(self, w, n_heads_arg, dim1=None, dim2=None, *, reverse=False):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        # When w is an FSDP DTensor, the intermediate view(n_heads_arg, ...) may have a leading
        # dim smaller than the mesh size. All-gather to Replicate so the intermediate reshapes
        # are unconstrained, then re-shard the shape-preserving result back to the original
        # placement (final dim1 is always divisible by the fsdp mesh size).
        if isinstance(w, DTensor):
            mesh, placements = w.device_mesh, w.placements
            w = w.redistribute(device_mesh=mesh, placements=[Replicate()] * mesh.ndim)
        else:
            mesh = placements = None
        half = dim1 // n_heads_arg // 2
        if reverse:
            shape = (n_heads_arg, 2, half, dim2)
        else:
            shape = (n_heads_arg, half, 2, dim2)
        result = w.view(shape).transpose(1, 2).reshape(dim1, dim2).clone()
        if placements is not None:
            return result.redistribute(device_mesh=mesh, placements=placements)
        return result

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = (
            self.model_config.layers[0].attention.n_kv_heads
            if self.model_config.layers[0].attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if (
                    self.fuse_qkv
                    and abstract_key == "layers.{}.attention.qkv_linear.wqkv.weight"
                ):
                    wq, wk, wv = self.fused_to_separate_qkv(
                        value, n_heads, n_kv_heads, head_dim
                    )
                    wq = self._permute(wq, n_heads)
                    key_value_dim = head_dim * n_kv_heads
                    wk = self._permute(wk, n_kv_heads, key_value_dim, dim)
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.q_proj.weight"
                    ] = wq
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.k_proj.weight"
                    ] = wk
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.v_proj.weight"
                    ] = wv
                    continue

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue

                if not self.fuse_qkv:
                    if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                        value = self._permute(value, n_heads)
                    if abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                        key_value_dim = head_dim * n_kv_heads
                        value = self._permute(value, n_kv_heads, key_value_dim, dim)

                new_key = new_key.format(layer_num)
            else:
                if self.model_config.enable_weight_tying and key == "output.weight":
                    continue
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = hf_state_dict.copy()
        if (
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = (
            self.model_config.layers[0].attention.n_kv_heads
            if self.model_config.layers[0].attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        state_dict = {}

        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._permute(value, n_heads, reverse=True)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim, reverse=True)

                if self.fuse_qkv and abstract_key in (
                    "model.layers.{}.self_attn.q_proj.weight",
                    "model.layers.{}.self_attn.k_proj.weight",
                    "model.layers.{}.self_attn.v_proj.weight",
                ):
                    if layer_num not in pending_qkv:
                        pending_qkv[layer_num] = {}
                    proj = abstract_key.split(".")[-2]
                    pending_qkv[layer_num][proj] = value
                    if len(pending_qkv[layer_num]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[layer_num]["q_proj"],
                            pending_qkv[layer_num]["k_proj"],
                            pending_qkv[layer_num]["v_proj"],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                        )
                        state_dict[
                            f"layers.{layer_num}.attention.qkv_linear.wqkv.weight"
                        ] = fused
                        del pending_qkv[layer_num]
                    continue

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    raise ValueError(
                        f"Unexpected or unmapped top-level HF key {key!r}. "
                        "Add it to from_hf_map (use None only for layer-pattern keys)."
                    )

            # pyrefly: ignore [unsupported-operation]
            state_dict[new_key] = value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        return state_dict
