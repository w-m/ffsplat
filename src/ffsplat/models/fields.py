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

    def copy(self) -> "Field":
        """Return a copy of the field."""
        return Field(self.data.clone(), self.op)

    @classmethod
    def from_file(cls, data: Tensor, file_path: Path, field_name: str) -> "Field":
        """Create a field from a file instead of a operation"""

        from ..models.operations import Operation

        op = Operation(
            input_fields={},
            params={
                "from": {"file": str(file_path), "last modified": Path(file_path).stat().st_mtime},
                "field_name": field_name,
            },
        )
        return cls(data, op)

    def to(self, device: str | torch.device) -> "Field":
        return Field(self.data.to(device, copy=True), self.op)

    def to_json(self) -> dict[str, Any]:
        """Convert the field to a JSON-serializable format."""
        return self.op.to_json()


class FieldDict(dict[str, Field]):
    def print_field_stats(self) -> None:
        print("Encoded field statistics:")
        for field_name, field_obj in sorted(self.items()):
            stats = f"{field_name}: \t{tuple(field_obj.data.shape)} | {field_obj.data.dtype}"
            if field_obj.data.numel() > 0:
                stats += f" | Min: {field_obj.data.min().item():.4f} | Max: {field_obj.data.max().item():.4f}"
                stats += f" | Median: {field_obj.data.median().item():.4f}"
                # stats += f" | Unique Count: {field_obj.data.unique().numel()}"
            print(stats)
