import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..models.operations import Operation


class Field:
    data: Tensor
    op: "Operation"

    def __init__(self, data: Tensor, op: "Operation") -> None:
        self.data = data
        self.op = op

    def __hash__(self) -> int:
        """Return the hash of the field."""
        json_str = json.dumps(self.to_json())
        return int(sha256(json_str.encode()).hexdigest(), 16)

    def __eq__(self, value: object, /) -> bool:
        return self.to_json() == value.to_json() if isinstance(value, Field) else False

    @classmethod
    def from_file(cls, data: Tensor, file_path: Path) -> "Field":
        """Create a field from a file instead of a operation"""

        from ..models.operations import Operation

        op = Operation(
            input_fields={},
            params={"from": {"file": str(file_path), "last modified": file_path.stat().st_mtime}},
        )
        return cls(data, op)

    def to(self, device: str | torch.device) -> "Field":
        return Field(self.data.to(device), self.op)

    def to_json(self) -> dict[str, Any]:
        """Convert the field to a JSON-serializable format."""
        return self.op.to_json()
