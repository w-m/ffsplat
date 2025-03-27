from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from ..io.ply import encode_ply
from ..models.gaussians import Gaussians


class SerializableDumper(yaml.SafeDumper):
    """Custom YAML Dumper with enhanced indentation and type handling.

    Handles special Python types for serialization:
    - defaultdicts
    - torch.Size
    - tuples
    - dataclass instances

    Removes Python-specific type tags from the output.
    """

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

    def represent_defaultdict(self, data):
        return self.represent_mapping("tag:yaml.org,2002:map", dict(data).items())

    def represent_torch_size(self, data):
        # Always use flow style for torch.Size
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", list(data))
        sequence.flow_style = True
        return sequence

    def represent_tuple(self, data):
        # Always use flow style for tuples
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", list(data))
        sequence.flow_style = True
        return sequence

    def represent_list(self, data):
        """Special representation for lists based on content."""
        sequence = self.represent_sequence("tag:yaml.org,2002:seq", data)
        # Use flow style for lists that contain only numbers
        if all(isinstance(item, (int, float)) for item in data):
            sequence.flow_style = True
        return sequence

    def represent_general(self, data):
        """General representer that handles dataclasses specially."""
        if is_dataclass(data) and not isinstance(data, type):
            return self.represent_mapping("tag:yaml.org,2002:map", asdict(data).items())
        return self.represent_data(data)


@dataclass
class DecodingParams:
    # TODO duplicated definition from scene_decoder
    # in decoding, the fields shouldn't be nullable
    """Parameters for decoding 3D scene formats."""

    container_identifier: str = "smurfx"
    container_version: str = "0.1"

    packer: str = "ffsplat-v0.1"

    meta: dict[str, Any] = field(default_factory=dict)

    files: list[dict[str, str]] = field(default_factory=list)
    fields: defaultdict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    scene: dict[str, Any] = field(default_factory=dict)

    def reverse_fields(self):
        """The operations we're accumulation during encoding need to be reversed for decoding."""
        prev_fields = self.fields.copy()
        self.fields = defaultdict(list)
        for field_name, field_ops in reversed(prev_fields.items()):
            self.fields[field_name] = list(reversed(field_ops))


@dataclass
class EncodingParams:
    profile: str
    profile_version: str

    scene: dict[str, Any]

    fields: dict[str, dict[str, Any]]
    files: list[dict[str, str]]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "EncodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(
            profile=data["profile"],
            profile_version=data["profile_version"],
            scene=data["scene"],
            fields=data["fields"],
            files=data.get("files", []),
        )


# Register representers for types outside the class definition
SerializableDumper.add_representer(defaultdict, SerializableDumper.represent_defaultdict)
SerializableDumper.add_representer(torch.Size, SerializableDumper.represent_torch_size)
SerializableDumper.add_representer(tuple, SerializableDumper.represent_tuple)
SerializableDumper.add_representer(list, SerializableDumper.represent_list)
SerializableDumper.add_multi_representer(object, SerializableDumper.represent_general)


@dataclass
class SceneEncoder:
    encoding_params: EncodingParams
    output_path: Path
    fields: dict[str, Tensor] = field(default_factory=dict)

    decoding_params: DecodingParams = field(default_factory=DecodingParams)

    # encoding YAML

    # fields:
    #   sh:
    #     - split:
    #         to_field_list: [f_dc, f_rest]
    #         split_size_or_sections: [1, 15]
    #         dim: 1

    #   f_dc:
    #     - reshape:
    #         shape: [-1, 3]
    #     - split:
    #         to_fields_with_prefix: point_cloud.ply@f_dc_
    #         split_size_or_sections: 1
    #         dim: 1

    #   f_rest:
    #     # TODO what is the correct order of f_rest?
    #     # requires reshape / permute?
    #     - reshape:
    #         shape: [-1, 45]
    #     - split:
    #         to_fields_with_prefix: point_cloud.ply@f_rest_
    #         split_size_or_sections: 1
    #         dim: 1

    #   quaternions:
    #     - split:
    #         to_fields_with_prefix: point_cloud.ply@rot_
    #         split_size_or_sections: 1
    #         dim: 1

    #   means:
    #     - split:
    #         to_field_list: [point_cloud.ply@x, point_cloud.ply@y, point_cloud.ply@z]
    #         split_size_or_sections: 1
    #         dim: 1

    #   scales:
    #     - remapping:
    #         method: exp
    #         inverse: True
    #     - split:
    #         to_fields_with_prefix: point_cloud.ply@scale_
    #         split_size_or_sections: 1
    #         dim: 1

    #   opacities:
    #     - remapping:
    #         method: sigmoid
    #         inverse: True
    #     - to_field: point_cloud.ply@opacity

    # files:
    #   - file_path: "point_cloud.ply"
    #     type: ply
    #     fields_with_prefix: point_cloud.ply@

    def _encode_fields(self) -> None:
        # go through the fields of encoding params
        for field_name, field_config in self.encoding_params.fields.items():
            field_data = self.fields.get(field_name)
            if field_data is None:
                raise ValueError(f"Field data is None: {field_name}")

            for field_op in field_config:
                match field_op:
                    case {
                        "split": {
                            "to_fields_with_prefix": to_fields_with_prefix,
                            "split_size_or_sections": split_size_or_sections,
                            "dim": dim,
                        }
                    }:
                        field_data = field_data.split(split_size_or_sections, dim)
                        for i, chunk in enumerate(field_data):
                            self.fields[f"{to_fields_with_prefix}{i}"] = chunk.squeeze(dim)

                        self.decoding_params.fields[field_name].append({
                            "combine": {
                                "from_fields_with_prefix": to_fields_with_prefix,
                                "method": "stack" if split_size_or_sections == 1 else "concat",
                                "dim": dim,
                            }
                        })

                    case {
                        "split": {
                            "to_field_list": to_field_list,
                            "split_size_or_sections": split_size_or_sections,
                            "dim": dim,
                        }
                    }:
                        field_data = field_data.split(split_size_or_sections, dim)
                        for target_field_name, target_field_data in zip(to_field_list, field_data):
                            self.fields[target_field_name] = target_field_data.squeeze(dim)

                        self.decoding_params.fields[field_name].append({
                            "combine": {
                                "from_field_list": to_field_list,
                                "method": "stack" if split_size_or_sections == 1 else "concat",
                                "dim": dim,
                            }
                        })

                    case {"reshape": {"shape": shape}}:
                        self.decoding_params.fields[field_name].append({"reshape": {"shape": field_data.shape}})
                        field_data = field_data.reshape(*shape)

                    case {"remapping": {"method": "log"}}:
                        self.decoding_params.fields[field_name].append({"remapping": {"method": "exp"}})
                        field_data = field_data.log()

                    case {"remapping": {"method": "inverse-sigmoid"}}:
                        self.decoding_params.fields[field_name].append({"remapping": {"method": "sigmoid"}})
                        field_data = torch.log(field_data / (1 - field_data))

                    case {"to_field": name}:
                        self.decoding_params.fields[field_name].append({"from_field": name})
                        self.fields[name] = field_data

    def _write_files(self) -> None:
        for file in self.encoding_params.files:
            file_path = file["file_path"]
            file_type = file["type"]
            field_prefix = file["fields_with_prefix"]

            fields_to_write = {
                field_name[len(field_prefix) :]: field_data
                for field_name, field_data in self.fields.items()
                if field_name.startswith(field_prefix)
            }

            self.decoding_params.files.append({
                "file_path": file_path,
                "type": file_type,
                "field_prefix": field_prefix,
            })

            output_file_path = self.output_path / file_path

            match file_type:
                case "ply":
                    encode_ply(fields=fields_to_write, path=output_file_path)
                case _:
                    raise NotImplementedError(f"Encoding for {file_type} is not supported")

    def encode(self) -> None:

        # container as folder for now
        self.output_path.mkdir(parents=True, exist_ok=True)

        self._encode_fields()
        self._write_files()

        # revert the order of the fields in self.decoding_params to enable straightforward decoding
        self.decoding_params.reverse_fields()
        self.decoding_params.scene = self.encoding_params.scene

        # Write the YAML directly using our custom dumper
        with open(self.output_path / "container_meta.yaml", "w") as f:
            yaml.dump(self.decoding_params, f, Dumper=SerializableDumper, default_flow_style=False, sort_keys=False)


def encode_gaussians(gaussians: Gaussians, output_path: Path, output_format: str) -> None:
    match output_format:
        case "3DGS-INRIA.ply":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/3DGS_INRIA_ply.yaml"))
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    encoder = SceneEncoder(encoding_params=encoding_params, output_path=output_path, fields=gaussians.to_dict())
    encoder.encode()
