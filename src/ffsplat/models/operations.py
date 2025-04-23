import json
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import ffsplat.models.transformations as transformations

from ..models.transformations import Transformation

if TYPE_CHECKING:
    from ..models.fields import Field


class Operation:
    input_fields: dict[str, "Field"]
    params: dict[str, Any]
    transform: Transformation

    def __init__(self, input_fields: dict[str, "Field"], params: dict[str, Any], verbose: bool = False) -> None:
        self.input_fields = input_fields
        self.params = params
        # TODO: create transformation based on params
        self.transform_type = params[next(iter(params))]
        match self.transform_type:
            case "cluster":
                self.transform = transformations.cluster(self.params["cluster"], self)

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
        }

    def apply(self):
        return self.transform.apply()
