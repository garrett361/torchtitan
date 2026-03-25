"""
Minimal FSDP + sonic-moe race condition reproducer.

Usage:
  torchrun --nproc_per_node=8 debug_sonic_race_minimal.py --sonic          # should crash
  torchrun --nproc_per_node=8 debug_sonic_race_minimal.py --grouped-mm     # should pass
  torchrun --nproc_per_node=8 debug_sonic_race_minimal.py --sonic --sync-fwd  # sync before fwd
  torchrun --nproc_per_node=8 debug_sonic_race_minimal.py --sonic --sync-bwd  # sync before bwd
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--sonic", action="store_true")
group.add_argument("--grouped-mm", action="store_true")
parser.add_argument("--sync-fwd", action="store_true", help="torch.cuda.synchronize() before forward compute per layer")
parser.add_argument("--sync-bwd", action="store_true", help="torch.cuda.synchronize() before backward compute per layer")
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--num-experts", type=int, default=128)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--top-k", type=int, default=16)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--model-dim", type=int, default=2048)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

if __name__ == "__main__":
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
        rank = dist.get_rank()
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(device)

        def log(msg):
            if rank == 0:
                print(msg, flush=True)

        log(
            f"sonic={args.sonic}, sync_fwd={args.sync_fwd}, sync_bwd={args.sync_bwd}, experts={args.num_experts}, top_k={args.top_k}"
        )

        if args.sonic:
            from sonicmoe.enums import ActivationType
            from sonicmoe.functional import moe_general_routing_inputs

        class BaseMoELayer(nn.Module):
            def __init__(self, dim, hidden_dim, num_experts, top_k):
                super().__init__()
                self.dim = dim
                self.hidden_dim = hidden_dim
                self.num_experts = num_experts
                self.top_k = top_k
                self.gate = nn.Linear(dim, num_experts, bias=False)
                self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
                self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
                self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
                nn.init.trunc_normal_(self.gate.weight, std=0.02)
                nn.init.trunc_normal_(self.w1, std=0.02)
                nn.init.trunc_normal_(self.w2, std=0.02)
                nn.init.trunc_normal_(self.w3, std=0.02)

            def _route(self, x_flat):
                scores = F.softmax(self.gate(x_flat).float(), dim=-1)
                top_scores, selected = torch.topk(scores, k=self.top_k, dim=-1)
                return top_scores / (
                    top_scores.sum(dim=-1, keepdim=True) + 1e-20
                ), selected

        class SonicMoELayer(BaseMoELayer):
            def forward(self, x):
                bs, slen, dim = x.shape
                x_flat = x.view(-1, dim)
                T = x_flat.shape[0]

                top_scores, selected_experts = self._route(x_flat)
                token_indices = (
                    torch.arange(T, device=x.device)
                    .unsqueeze(1)
                    .expand(-1, self.top_k)
                    .reshape(-1)
                )
                w1_w3 = (
                    torch.stack([self.w1, self.w3], dim=2)
                    .reshape(self.num_experts, 2 * self.hidden_dim, dim)
                    .permute(1, 2, 0)
                )
                w2_sonic = self.w2.permute(1, 2, 0)

                out, _ = moe_general_routing_inputs(
                    x=x_flat,
                    router_scores=top_scores.view(-1),
                    token_indices=token_indices.int(),
                    expert_indices=selected_experts.view(-1).int(),
                    w1=w1_w3,
                    b1=None,
                    w2=w2_sonic,
                    b2=None,
                    E=self.num_experts,
                    stream_id=0,
                    activation_type=ActivationType.SWIGLU,
                    is_inference_mode_enabled=False,
                )
                return out.view(bs, slen, dim)

        class GroupedMMMoELayer(BaseMoELayer):
            def forward(self, x):
                bs, slen, dim = x.shape
                x_flat = x.view(-1, dim)

                top_scores, selected_experts = self._route(x_flat)
                flat_experts = selected_experts.view(-1)
                sort_indices = torch.argsort(flat_experts, stable=True)
                offsets = torch.cumsum(
                    torch.histc(
                        flat_experts.float(),
                        bins=self.num_experts,
                        min=0,
                        max=self.num_experts,
                    ),
                    dim=0,
                    dtype=torch.int32,
                )
                token_indices = sort_indices // self.top_k
                routed = x_flat[token_indices]
                routed = (
                    routed.float() * top_scores.view(-1)[sort_indices].unsqueeze(-1)
                ).to(x.dtype)

                h = F.silu(
                    torch._grouped_mm(
                        routed.bfloat16(),
                        self.w1.bfloat16().transpose(-2, -1),
                        offs=offsets,
                    )
                )
                h = h * torch._grouped_mm(
                    routed.bfloat16(),
                    self.w3.bfloat16().transpose(-2, -1),
                    offs=offsets,
                )
                out_routed = torch._grouped_mm(
                    h, self.w2.bfloat16().transpose(-2, -1), offs=offsets
                ).to(x.dtype)

                out = torch.zeros_like(x_flat)
                out.scatter_add_(
                    0, token_indices.unsqueeze(-1).expand_as(out_routed), out_routed
                )
                return out.view(bs, slen, dim)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                moe_cls = SonicMoELayer if args.sonic else GroupedMMMoELayer
                self.layers = nn.ModuleList(
                    [
                        moe_cls(
                            args.model_dim,
                            args.hidden_dim,
                            args.num_experts,
                            args.top_k,
                        )
                        for _ in range(args.num_layers)
                    ]
                )
                self.output = nn.Linear(args.model_dim, args.model_dim, bias=False)

            def forward(self, x):
                for layer in self.layers:
                    x = x + layer(x)
                return self.output(x)

        with device:
            model = Model().bfloat16()

        for layer in model.layers:
            # fully_shard(layer, reshard_after_forward=False)
            fully_shard(layer)
        fully_shard(model)

        def _sync_hook(*_args, **_kwargs):
            torch.cuda.synchronize()

        if args.sync_fwd:
            for layer in model.layers:
                layer.register_forward_pre_hook(_sync_hook)
            log("HOOKED: sync before forward compute per layer")

        if args.sync_bwd:
            for layer in model.layers:
                layer.register_full_backward_pre_hook(_sync_hook)
            log("HOOKED: sync before backward compute per layer")

        log(f"Running {args.steps} steps\n{model=}\n")
        x = torch.randn(
            args.batch_size,
            args.seq_len,
            args.model_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        for step in range(args.steps):
            model(x).sum().backward()
            model.zero_grad()
            if step % 100 == 0:
                log(f"  Step {step}/{args.steps}")

        log(f"Completed {args.steps} steps without error!")
    finally:
        dist.destroy_process_group()
