from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ..models.fields import Field


@dataclass
class Operation:
    input_fields: dict[str, Field]
    params: dict[str, Any]

    def __hash__(self) -> int:
        return sha256(self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the operation to a JSON-serializable format."""
        return {
            "input_fields": {name: field.to_json() for name, field in self.input_fields.items()},
            "params": self.params,
        }
