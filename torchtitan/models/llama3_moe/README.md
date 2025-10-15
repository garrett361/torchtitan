# Weight Loading, Checkpointing, etc.

### Understanding of Initial HF Loading

In `CheckpointManager.dcp_load` with `from_hf=True`

1. Create the `torchtitan` state dict, where the weights are probably already sharded.
2. Re-key the `torchtitan` state dict, and maybe perform some weight surgery, so that we convert it
   into a format which matches the serialized HF state dict. Uses `StateDictAdapter.to_hf`.
3. Stream serialized weights into HF state dict via `dcp.load`.
4. Convert back to a `torchtitan` state dict with `StateDictAdapter.from_hf`
5. Load the `torchtitan` state dict into the sharded model.


With the `DefaultPlanner`, the `ReadItem` instances carry the state dict fqn corresponding to the
given tensor via `read_item.storage_index.fqn`.
