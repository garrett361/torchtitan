# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.hf_datasets.text_datasets import ChatDataLoader, ChatDataset


class GraniteSFTDataset(ChatDataset):
    """ChatDataset variant that accepts an optional leading system message.

    Extends ChatDataset to support [user, assistant] and
    [system, user, assistant] message formats. The prompt length used for
    label masking is computed from all messages except the final assistant
    turn, ensuring the system turn (if present) is masked along with the
    user turn.
    """

    @staticmethod
    def _validate_messages(messages: list[dict]) -> None:
        if len(messages) not in (2, 3):
            raise ValueError(
                f"Expected [user, assistant] or [system, user, assistant], "
                f"got {len(messages)} messages"
            )
        if len(messages) == 3 and messages[0]["role"] != "system":
            raise ValueError(
                f"First of 3 messages must be 'system', got '{messages[0]['role']}'"
            )
        user_idx = 1 if len(messages) == 3 else 0
        if messages[user_idx]["role"] != "user":
            raise ValueError(
                f"Expected 'user' at index {user_idx}, got '{messages[user_idx]['role']}'"
            )
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                f"Last message must be 'assistant', got '{messages[-1]['role']}'"
            )

    def _prompt_messages(self, messages: list[dict]) -> list[dict]:
        """Everything except the final assistant turn.

        For [user, assistant] returns [user] (same as parent).
        For [system, user, assistant] returns [system, user], ensuring the
        system block is included in the masked prompt prefix.
        """
        return messages[:-1]


class GraniteSFTDataLoader(ChatDataLoader):
    """ChatDataLoader that instantiates GraniteSFTDataset."""

    _dataset_class = GraniteSFTDataset

    @dataclass(kw_only=True, slots=True)
    class Config(ChatDataLoader.Config):
        pass  # inherits all fields; this definition sets _owner = GraniteSFTDataLoader
