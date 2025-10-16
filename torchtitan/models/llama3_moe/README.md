# Weight Loading, Checkpointing, etc.

## Understanding of Initial HF Loading

In `CheckpointManager.dcp_load` with `from_hf=True`

1. Create the `torchtitan` state dict, where the weights are probably already sharded.
2. Re-key the `torchtitan` state dict, and maybe perform some weight surgery, so that we convert it
   into a format which matches the serialized HF state dict. Uses `StateDictAdapter.to_hf`.
3. Stream serialized weights into HF state dict via `dcp.load`.
4. Convert back to a `torchtitan` state dict with `StateDictAdapter.from_hf`
5. Load the `torchtitan` state dict into the sharded model.


The `HuggingFaceStorageReader` class plays a central role in the loading, and can be hacked for our
intended purpose of modifying the HF weights so they can be loaded into a LLama3MoE model, e.g. by
replicating a FFN weight `num_expert` times to insert into an MoE weight.

The base class reads in the safetensors files that the user points to and generates metadata
associated with both the serialized tensors defined in the files and the model tensors that this
data will be loaded into. Metadata about the safetensors goes into `_HFStorageInfo`, while
torchtitan model weight metadata goes into `TensorStorageMetadata` and `ChunkStorageMetadata`, which
each capture things like the `shape`, `dtype`, offset info, etc.

The base class assumes that the torchtitan model weights have the same layout as the safetensors
ones, so all the metadata is derived from the safetensor metadata header alone. We break this
assumption, and in our `TransformingHuggingFaceStorageReader` subclass we instead derive the model
metadata from the actual model state dict.  Finally, to be able to actually load the safetensor
weights into the model, we need to perform whatever transformations are necessary on the safetensor
weights prior to loading. This is accomplished by initializing the subclass with a
`transform_fn: Callable[[str, torch.Tensor], torch.Tensor]` arg which gets the weight and its fqn
and is applied to the safetensor weight through a tiny modification of
`HuggingFaceStorageReader._process_read_request`. For this all to work out cleanly, the torchtitan
state dict and the safetensors keys need to match, so the `StateDictAdapter` is also passed to the
subclass to make this step seamless.


##
Other notes:

- With the `DefaultPlanner`, the `ReadItem` instances carry the state dict fqn corresponding to the
  given tensor via `read_item.storage_index.fqn`.
