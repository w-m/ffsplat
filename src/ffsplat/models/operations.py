import json
from collections import defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import ffsplat.models.transformations as transformations

from ..models.transformations import Transformation

if TYPE_CHECKING:
    from ..models.fields import Field

transformation_map = {
    "cluster": transformations.Cluster,
    "split": transformations.Split,
    "flatten": transformations.Flatten,
    "reshape": transformations.Reshape,
    "remapping": transformations.Remapping,
    "to_field": transformations.ToField,
    "permute": transformations.Permute,
    "to_dtype": transformations.ToDType,
    "split_bytes": transformations.SplitBytes,
    "reindex": transformations.Reindex,
    "plas": transformations.PLAS,
}


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

    def apply(self, verbose: bool) -> tuple[dict[str, "Field"], defaultdict[str, list]]:
        transformation = transformation_map.get(self.transform_type)
        if transformation is None:
            raise ValueError(f"Unknown transformation: {self.transform_type}")
        return transformation.apply(self.params[self.transform_type], self, verbose)
