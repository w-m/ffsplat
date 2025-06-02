import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models.fields import Field, FieldDict
from ..models.gaussians import Gaussians
from ..models.operations import Operation


class SerializableDumper(yaml.SafeDumper):
    """Custom YAML Dumper with enhanced indentation and type handling.

    Handles special Python types for serialization:
    - defaultdicts
    - torch.Size
    - tuples
    - dataclass instances

    Removes Python-specific type tags from the output.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)

    def represent_defaultdict(self, data: defaultdict) -> yaml.nodes.MappingNode:
        return self.represent_mapping("tag:yaml.org,2002:map", dict(data).items())

    def represent_torch_size(self, data: torch.Size) -> yaml.nodes.SequenceNode:
        # Always use flow style for torch.Size
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", list(data))
        sequence.flow_style = True
        return sequence

    def represent_tuple(self, data: tuple) -> yaml.nodes.SequenceNode:
        # Always use flow style for tuples
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", list(data))
        sequence.flow_style = True
        return sequence

    def represent_list(self, data: Iterable[Any]) -> yaml.nodes.SequenceNode:
        """Special representation for lists based on content."""
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", data)
        # Use flow style for lists that contain only numbers
        if all(isinstance(item, (int, float)) for item in data):
            sequence.flow_style = True
        # Use flow style for lists that contain only strings
        if all(isinstance(item, str) for item in data):
            sequence.flow_style = True
        return sequence

    def represent_general(self, data: Any) -> Any:
        """General representer that handles dataclasses specially."""
        if is_dataclass(data) and not isinstance(data, type):
            return self.represent_mapping("tag:yaml.org,2002:map", asdict(data).items())
        return self.represent_data(data)


@dataclass
class DecodingParams:
    # TODO duplicated definition from scene_decoder
    # in decoding, the fields shouldn't be nullable
    """Parameters for decoding 3D scene formats."""

    container_identifier: str
    container_version: str

    packer: str

    profile: str
    profile_version: str

    meta: dict[str, Any] = field(default_factory=dict)

    ops: list[dict[str, Any]] = field(default_factory=list)
    scene: dict[str, Any] = field(default_factory=dict)

    def reverse_ops(self) -> None:
        """The operations we're accumulation during encoding need to be reversed for decoding."""
        prev_ops = self.ops.copy()
        self.ops = []
        for op in reversed(prev_ops):
            self.ops.append(op)
            # don't reverse the file reading operations
            if op["input_fields"] == []:
                continue
            self.ops[-1]["transforms"].reverse()


@dataclass
class EncodingParams:
    profile: str
    profile_version: str

    scene: dict[str, Any]

    ops: list[dict[str, Any]]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "EncodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(
            profile=data["profile"],
            profile_version=data["profile_version"],
            scene=data["scene"],
            ops=data["ops"],
        )

    def to_yaml_file(self, yaml_path: Path) -> None:
        # Write the YAML directly using our custom dumper
        with open(yaml_path / f"custom_{self.profile}.yaml", "w") as f:
            yaml.dump(self, f, Dumper=SerializableDumper, default_flow_style=False, sort_keys=False)


# Register representers for types outside the class definition
SerializableDumper.add_representer(defaultdict, SerializableDumper.represent_defaultdict)
SerializableDumper.add_representer(torch.Size, SerializableDumper.represent_torch_size)
SerializableDumper.add_representer(tuple, SerializableDumper.represent_tuple)
SerializableDumper.add_representer(list, SerializableDumper.represent_list)
SerializableDumper.add_multi_representer(object, SerializableDumper.represent_general)


@lru_cache
def process_operation(
    op: Operation,
    verbose: bool,
) -> tuple[dict[str, Field], list[dict[str, Any]]]:
    """Process the operation and return the new fields and decoding updates."""
    if verbose:
        print(f"Encoding {op}...")
    return op.apply(verbose=verbose)


@dataclass
class SceneEncoder:
    encoding_params: EncodingParams
    output_path: Path
    decoding_params: DecodingParams

    fields: FieldDict

    def _encode_fields(self, verbose: bool) -> None:
        """Process the fields according to the operations defined in the encoding parameters."""
        # this loops over every block in the encoding config
        for op_params in self.encoding_params.ops:
            # build each operation and process it
            input_fields_params = op_params["input_fields"]
            for transform_param in op_params["transforms"]:
                op = Operation.from_json(input_fields_params, transform_param, self.fields, self.output_path)
                new_fields, decoding_updates = process_operation(op, verbose=verbose)
                #  if the coding_updates are not a copy the cache will be wrong
                for decoding_update in copy.deepcopy(decoding_updates):
                    # if the last decoding update has the same input fields we can combine the transforms into one list
                    if (
                        self.decoding_params.ops
                        and self.decoding_params.ops[-1]["input_fields"] == decoding_update["input_fields"]
                    ):
                        self.decoding_params.ops[-1]["transforms"] += decoding_update["transforms"]
                    else:
                        self.decoding_params.ops.append(decoding_update)
                self.fields.update(new_fields)

    def encode(self, verbose: bool) -> None:
        # container as folder for now
        self.output_path.mkdir(parents=True, exist_ok=True)

        self._encode_fields(verbose=verbose)

        if verbose:
            self.fields.print_field_stats()

        # revert the order of the operations in self.decoding_params to enable straightforward decoding
        self.decoding_params.reverse_ops()

        # Write the YAML directly using our custom dumper
        with open(self.output_path / "container_meta.yaml", "w") as f:
            yaml.dump(self.decoding_params, f, Dumper=SerializableDumper, default_flow_style=False, sort_keys=False)


def encode_gaussians(gaussians: Gaussians, output_path: Path, output_format: str, verbose: bool) -> None:
    match output_format:
        case "3DGS-INRIA.ply":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/3DGS_INRIA_ply.yaml"))
        case "3DGS-INRIA-nosh.ply":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/3DGS_INRIA_nosh_ply.yaml"))
        case "SOG-web":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/SOG-web.yaml"))
        case "SOG-web-nosh":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/SOG-web-nosh.yaml"))
        case "SOG-web-sh-split":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/SOG-web-sh-split.yaml"))
        case "SOG-canvas":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/SOG-canvas.yaml"))
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    encoder = SceneEncoder(
        encoding_params=encoding_params,
        output_path=output_path,
        fields=gaussians.to_field_dict(),
        decoding_params=DecodingParams(
            container_identifier="smurfx",
            container_version="0.1",
            packer="ffsplat-v0.1",
            profile=encoding_params.profile,
            profile_version=encoding_params.profile_version,
            scene=encoding_params.scene,
        ),
    )
    encoder.encode(verbose=verbose)
