import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models.transformations import Transformation, apply_transform

if TYPE_CHECKING:
    from ..models.fields import Field


class Operation:
    input_fields: dict[str, "Field"]
    params: dict[str, Any]
    transform_type: str
    transform: Transformation

    def __init__(self, input_fields: dict[str, "Field"], params: dict[str, Any], verbose: bool = False) -> None:
        self.input_fields = input_fields
        self.params = params
        # get trasformation from params
        self.transform_type = next(iter(params))

    @classmethod
    def from_json(
        cls,
        input_field_param: list[str] | dict[str, str],
        transform_param: dict[str, Any],
        field_data: dict[str, "Field"],
        output_path: Path | None = None,
    ) -> "Operation":
        params = copy.deepcopy(transform_param)
        transform_type = next(iter(params))

        if transform_type == "write_file":
            params[transform_type]["base_path"] = str(output_path)

        if isinstance(input_field_param, dict):
            prefix = input_field_param.get("from_fields_with_prefix", None)
            if prefix is None:
                raise ValueError("Expected a prefix in the input field parameters")
            if transform_type == "write_file" and params[transform_type]["type"] == "ply":
                params[transform_type]["field_prefix"] = prefix
            input_fields = {name: field_data[name] for name in field_data if name.startswith(prefix)}

        elif isinstance(input_field_param, list):
            input_fields = {name: field_data[name] for name in input_field_param}

        return cls(input_fields, params)

    def __hash__(self) -> int:
        json_str = json.dumps(self.to_json())
        return int(sha256(json_str.encode()).hexdigest(), 16)

    def __eq__(self, value: object, /) -> bool:
        return self.to_json() == value.to_json() if isinstance(value, Operation) else False

    def __str__(self) -> str:
        """Return a compact string representation of the operation."""
        return f"Operation(type={self.transform_type}, inputs={list(self.input_fields.keys())})"

    def to_json(self) -> dict[str, Any]:
        """Convert the operation to a JSON-serializable format."""
        return {
            "input_fields": {name: field.to_json() for name, field in self.input_fields.items()},
            "params": self.params,
        }

    def apply(self, verbose: bool) -> tuple[dict[str, "Field"], list[dict[str, Any]]]:
        return apply_transform(self, verbose=verbose)
