import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import and_masks

from dtest import DTest
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.models.common.attention import (
    build_fa4_mask,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel

SEQ_LEN = 512
B = 1
CONFIG = "debugmodel_fa4"

torch._dynamo.config.capture_scalar_outputs = True


class TestCPFA4(DTest):
    default_world_size = 2

    def _make_mesh(self):
        """Single mesh used for both CP and FSDP (dp=1, so they span the same ranks)."""
        world_mesh = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("dp", "cp")
        )
        return world_mesh["cp"]

    def _fsdp_wrap(self, model, mesh):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}
        fully_shard(
            [model.tok_embeddings, model.norm, model.output], **fsdp_config
        )
        for block in model.layers.values():
            fully_shard(block, **fsdp_config)
        fully_shard(model, **fsdp_config)

    def _build_ref_model(self, mesh) -> GraniteModel:
        torch.manual_seed(42)
        config = granite_configs[CONFIG]()
        model = GraniteModel(config)
        model.init_states()
        model.to(self.device)
        self._fsdp_wrap(model, mesh)
        return model

    def _build_cp_model(self, mesh) -> GraniteModel:
        torch.manual_seed(42)
        config = granite_configs[CONFIG]()
        model = GraniteModel(config)
        model.init_states()
        model.to(self.device)

        attn_modules = [
            layer.attention.inner_attention for layer in model.layers.values()
        ]
        apply_cp_to_attention_module(attn_modules, mesh)

        for layer in model.layers.values():
            layer.compile(fullgraph=True)

        self._fsdp_wrap(model, mesh)
        disable_fsdp_gradient_division(model)

        return model

    def _make_tokens(self, seq_len=SEQ_LEN):
        torch.manual_seed(123)
        vocab_size = granite_configs[CONFIG]().vocab_size
        return torch.randint(0, vocab_size, (B, seq_len), device=self.device)

    def _make_positions_multidoc(self, seq_len=SEQ_LEN, n_docs=4):
        doc_len = seq_len // n_docs
        return (
            torch.cat([torch.arange(doc_len) for _ in range(n_docs)])
            .unsqueeze(0)
            .to(self.device)
        )

    def _compare_fwd(
        self, ref_model, cp_model, tokens, positions, cp_mesh, lb,
        document_ids=None, atol=1e-2, rtol=1e-2,
    ):
        ref_mask = (
            build_fa4_mask(document_ids=document_ids)
            if document_ids is not None
            else None
        )
        with torch.no_grad():
            out_ref = ref_model(
                tokens, attention_masks=ref_mask, positions=positions
            )

        shard_indices = lb._generate_indices(restore=False)
        tokens_sh, positions_sh = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=(tokens, positions),
            seq_dims=(1, 1),
            load_balancer=lb,
        )
        local_seq_len = tokens_sh.shape[1]
        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=cp_mesh.get_local_rank(),
            local_seq_len=local_seq_len,
            document_ids=document_ids,
        )
        with torch.no_grad():
            out_cp_local = cp_model(
                tokens_sh, attention_masks=fa4_mask, positions=positions_sh
            )

        gathered = [
            torch.zeros_like(out_cp_local) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, out_cp_local.contiguous())
        out_cp_full = torch.cat(gathered, dim=1)
        restore_indices = lb._generate_indices(restore=True)
        out_cp_restored = out_cp_full[:, restore_indices.squeeze(0)]

        diff = (out_cp_restored.float() - out_ref.float()).abs()
        ref_abs = out_ref.float().abs()
        rel = diff / ref_abs.clamp(min=1e-10)
        if cp_mesh.get_local_rank() == 0:
            print(f"  FWD: max_abs={diff.max().item():.6e} "
                  f"max_rel={rel.max().item():.6e}")
        torch.testing.assert_close(
            out_cp_restored, out_ref, atol=atol, rtol=rtol
        )

    def _compare_bwd(
        self, ref_model, cp_model, tokens, positions, cp_mesh, lb,
        document_ids=None, atol=2e-2, rtol=2e-2,
    ):
        vocab_size = granite_configs[CONFIG]().vocab_size
        norm = SEQ_LEN * B * vocab_size

        shard_indices = lb._generate_indices(restore=False)
        tokens_sh, positions_sh = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=(tokens, positions),
            seq_dims=(1, 1),
            load_balancer=lb,
        )
        local_seq_len = tokens_sh.shape[1]
        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=cp_mesh.get_local_rank(),
            local_seq_len=local_seq_len,
            document_ids=document_ids,
        )
        out_cp = cp_model(
            tokens_sh, attention_masks=fa4_mask, positions=positions_sh
        )
        loss_cp = out_cp.sum() / norm
        loss_cp.backward()

        ref_mask = (
            build_fa4_mask(document_ids=document_ids)
            if document_ids is not None
            else None
        )
        out_ref = ref_model(
            tokens, attention_masks=ref_mask, positions=positions
        )
        loss_ref = out_ref.sum() / norm
        loss_ref.backward()

        seen_ids = set()
        compared = 0
        worst_abs = 0.0
        worst_rel = 0.0
        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), cp_model.named_parameters()
        ):
            if id(p2) in seen_ids:
                continue
            seen_ids.add(id(p2))
            if p1.grad is None:
                continue
            assert p2.grad is not None, f"CP model missing grad for {n2}"
            full_grad_ref = p1.grad.full_tensor()
            full_grad_cp = p2.grad.full_tensor()
            diff = (full_grad_cp.float() - full_grad_ref.float()).abs()
            ref_abs = full_grad_ref.float().abs()
            abs_err = diff.max().item()
            rel_err = (diff / ref_abs.clamp(min=1e-10)).max().item()
            worst_abs = max(worst_abs, abs_err)
            worst_rel = max(worst_rel, rel_err)
            torch.testing.assert_close(
                full_grad_cp, full_grad_ref, atol=atol, rtol=rtol,
                msg=f"grad mismatch: {n1}",
            )
            compared += 1
        if cp_mesh.get_local_rank() == 0:
            print(f"  BWD: worst_abs={worst_abs:.6e} worst_rel={worst_rel:.6e} "
                  f"(compared {compared} params)")
        assert compared > 0, "No gradients compared"

    def _make_ptrr_lb(self, positions):
        mask_mods = [
            get_causal_mask_mod(),
            get_document_mask_mod_from_positions(positions),
        ]
        block_mask = create_attention_mask(
            and_masks(*mask_mods), B, None, SEQ_LEN, SEQ_LEN
        )
        return _PTRRLoadBalancer(block_mask, self.world_size)

    # --- Test methods ---

    def test_cp_fa4_causal_fwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = torch.arange(SEQ_LEN, device=self.device).unsqueeze(0)
        lb = _HeadTailLoadBalancer(SEQ_LEN, self.world_size, self.device.type)
        self._compare_fwd(ref_model, cp_model, tokens, positions, mesh, lb)

    def test_cp_fa4_causal_bwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = torch.arange(SEQ_LEN, device=self.device).unsqueeze(0)
        lb = _HeadTailLoadBalancer(SEQ_LEN, self.world_size, self.device.type)
        self._compare_bwd(ref_model, cp_model, tokens, positions, mesh, lb)

    def test_cp_fa4_doc_causal_fwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = self._make_positions_multidoc()
        document_ids = Decoder._document_ids_from_positions(positions)
        lb = _HeadTailLoadBalancer(SEQ_LEN, self.world_size, self.device.type)
        self._compare_fwd(
            ref_model, cp_model, tokens, positions, mesh, lb,
            document_ids=document_ids,
        )

    def test_cp_fa4_doc_causal_bwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = self._make_positions_multidoc()
        document_ids = Decoder._document_ids_from_positions(positions)
        lb = _HeadTailLoadBalancer(SEQ_LEN, self.world_size, self.device.type)
        self._compare_bwd(
            ref_model, cp_model, tokens, positions, mesh, lb,
            document_ids=document_ids,
        )

    def test_cp_fa4_ptrr_doc_causal_fwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = self._make_positions_multidoc()
        document_ids = Decoder._document_ids_from_positions(positions)
        lb = self._make_ptrr_lb(positions)
        self._compare_fwd(
            ref_model, cp_model, tokens, positions, mesh, lb,
            document_ids=document_ids,
        )

    def test_cp_fa4_ptrr_doc_causal_bwd(self):
        mesh = self._make_mesh()
        ref_model = self._build_ref_model(mesh)
        cp_model = self._build_cp_model(mesh)
        tokens = self._make_tokens()
        positions = self._make_positions_multidoc()
        document_ids = Decoder._document_ids_from_positions(positions)
        lb = self._make_ptrr_lb(positions)
        self._compare_bwd(
            ref_model, cp_model, tokens, positions, mesh, lb,
            document_ids=document_ids,
        )
