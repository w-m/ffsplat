import json
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.fields import Field


@dataclass
class Operation:
    input_fields: dict[str, "Field"]
    params: dict[str, Any]
    out_field: str | list[str]

    def __hash__(self) -> int:
        json_str = json.dumps(self.to_json())
        return int(sha256(json_str.encode()).hexdigest(), 16)

    def __eq__(self, value: object, /) -> bool:
        return self.to_json() == value.to_json() if isinstance(value, Operation) else False

    def to_json(self) -> dict[str, Any]:
        """Convert the operation to a JSON-serializable format."""
        return {
            "input_fields": {name: field.to_json() for name, field in self.input_fields.items()},
            "params": self.params,
            "out_field": self.out_field,
        }
