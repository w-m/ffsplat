from hashlib import sha256
from typing import Any

import torch
from torch import Tensor


class Field:
    data: Tensor
    # TODO: :ops are operations, but importing Operation to type it is circular. How should I do this?
    ops: list

    def __init__(self, data: Tensor):
        self.data = data
        # TODO: this should never be empty, how to initalize it?
        self.ops = []

    def __hash__(self) -> int:
        """Return the hash of the field."""
        return sha256(self.to_json())

    def to(self, device: str | torch.device) -> "Field":
        return Field(self.data.to(device))

    def to_json(self) -> list[dict[str, Any]]:
        """Convert the field to a JSON-serializable format."""
        # TODO: as list? or should this be a json string?
        return [op.to_json() for op in self.ops]
