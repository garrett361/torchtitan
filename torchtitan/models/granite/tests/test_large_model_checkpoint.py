import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dtest import DTest
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel
from torchtitan.models.granite.state_dict_adapter import GraniteStateDictAdapter


def _get_ckpt_path():
    from dotenv import load_dotenv

    load_dotenv()
    path = os.getenv("HF_ASSETS_PATH_30B")
    if path is None:
        pytest.skip("HF_ASSETS_PATH_30B not set")
    return path


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="requires 8 GPUs"
)
class Test30BCheckpoint(DTest):
    default_world_size = 8

    def _fsdp_wrap_tt(self, model, mesh):
        for block in model.layers.values():
            fully_shard(block, mesh=mesh)
        fully_shard([model.tok_embeddings, model.norm, model.output], mesh=mesh)
        fully_shard(model, mesh=mesh)

    def _fsdp_wrap_hf_layerwise(self, hf_model, mesh):
        for block in hf_model.model.layers:
            block.to(self.device)
            fully_shard(block, mesh=mesh)
        hf_model.model.embed_tokens.to(self.device)
        hf_model.model.norm.to(self.device)
        hf_model.lm_head.to(self.device)
        fully_shard(
            [hf_model.model.embed_tokens, hf_model.model.norm, hf_model.lm_head],
            mesh=mesh,
        )
        fully_shard(hf_model, mesh=mesh)

    def test_logits_match_hf(self):
        from transformers import AutoModelForCausalLM

        ckpt_path = _get_ckpt_path()
        mesh = init_device_mesh("cuda", (self.world_size,))
        torch.manual_seed(self.rank)
        tokens = torch.randint(1, 1000, (1, 8), dtype=torch.long, device=self.device)

        # Phase 1: HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        hf_model.eval()
        self._fsdp_wrap_hf_layerwise(hf_model, mesh)
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits.cpu()
        del hf_model
        torch.cuda.empty_cache()

        # Phase 2: TT model — meta init + DCP load (trainer pattern)
        config = granite_configs["30B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=ckpt_path)

        with torch.device("meta"):
            model = GraniteModel(config)
        self._fsdp_wrap_tt(model, mesh)
        model.to_empty(device=self.device)
        model.init_states(buffer_device=self.device)

        storage_reader = adapter.get_hf_storage_reader(ckpt_path)
        hf_state_dict = adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        tt_state_dict = adapter.from_hf(hf_state_dict)
        del hf_state_dict
        set_model_state_dict(
            model,
            model_state_dict=tt_state_dict,
            options=StateDictOptions(strict=True),
        )
        del tt_state_dict
        model.eval()

        with torch.no_grad():
            tt_logits = model(tokens).cpu()

        torch.testing.assert_close(tt_logits, hf_logits, atol=1e-4, rtol=0.0)
