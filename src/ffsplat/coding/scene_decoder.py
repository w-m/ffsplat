from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from ..io.ply import decode_ply
from ..models.gaussians import Gaussians


@dataclass
class DecodingParams:
    """Parameters for decoding 3D scene formats."""

    files: list[dict[str, str]]
    fields: dict[str, dict[str, Any]]
    scene: dict[str, Any]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "DecodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(files=data.get("files", []), fields=data.get("fields", {}), scene=data.get("scene", {}))

    def with_input_path(self, input_path: Path) -> "DecodingParams":
        # check that we have a single file only, replace its path with the input path
        if len(self.files) != 1:
            raise ValueError("Expected a single file in the YAML template")

        self.files[0]["file_path"] = str(input_path)
        return self


@dataclass
class SceneDecoder:
    decoding_params: DecodingParams
    fields: dict[str, Tensor] = field(default_factory=dict)
    scene: Any = None

    def _decode_files(self) -> None:
        for file in self.decoding_params.files:
            file_path = Path(file["file_path"])
            file_type = file["type"]
            field_prefix = file["field_prefix"]

            match file_type:
                case "ply":
                    ply_fields = decode_ply(file_path=file_path, field_prefix=field_prefix)
                    self.fields.update(ply_fields)
                case _:
                    raise ValueError(f"Unsupported file type: {type}")

    def _process_fields(self) -> None:
        for field_name, field_ops in self.decoding_params.fields.items():
            for field_op in field_ops:
                match field_op:
                    case {"combine": {"from_fields_with_prefix": from_prefix, "method": method, "dim": dim}}:
                        prefix_tensors: list[Tensor] = [
                            field_data
                            for source_field_name, field_data in self.fields.items()
                            if source_field_name.startswith(from_prefix)
                        ]
                        if method == "stack":
                            field_data = torch.stack(prefix_tensors, dim=dim)
                        elif method == "concat":
                            field_data = torch.cat(prefix_tensors, dim=dim)
                        else:
                            raise ValueError(f"Unsupported combine method: {method}")

                    case {"combine": {"from_field_list": from_list, "method": method, "dim": dim}}:
                        source_tensors: list[Tensor] = [
                            self.fields[source_field_name]
                            for source_field_name in from_list
                            if source_field_name in self.fields
                        ]
                        if method == "stack":
                            field_data = torch.stack(source_tensors, dim=dim)
                        elif method == "concat":
                            field_data = torch.cat(source_tensors, dim=dim)
                        else:
                            raise ValueError(f"Unsupported combine method: {method}")

                    case {"from_field": name}:
                        if name in self.fields:
                            field_data = self.fields[name]
                        else:
                            raise ValueError(f"Field not found: {name}")

                    case {"reshape": {"shape": shape}}:
                        field_data = field_data.reshape(*shape)

                    case {"permute": {"dims": dims}}:
                        field_data = field_data.permute(*dims)

                    case {"remapping": {"method": method}}:
                        match method:
                            case "exp":
                                field_data = torch.exp(field_data)
                            case "sigmoid":
                                field_data = torch.sigmoid(field_data)
                            case _:
                                raise ValueError(f"Unsupported remapping method: {method}")
                    case _:
                        raise ValueError(f"Unsupported field operation: {field_op}")

            self.fields[field_name] = field_data

    def _create_scene(self) -> None:
        match self.decoding_params.scene.get("primitives"):
            case "3DGS-INRIA":
                self.scene = Gaussians(
                    means=self.fields["means"],
                    quaternions=self.fields["quaternions"],
                    scales=self.fields["scales"],
                    opacities=self.fields["opacities"],
                    sh=self.fields["sh"],
                )
            case _:
                raise ValueError("Unsupported scene format")

    def decode(self) -> None:
        self._decode_files()
        self._process_fields()
        self._create_scene()


def decode_gaussians(input_path: Path, input_format: str) -> Gaussians:
    input_file_extension = input_path.suffix

    if input_format == "3DGS-INRIA" and input_file_extension == ".ply":
        decoding_params = DecodingParams.from_yaml_file(Path("3DGS_INRIA_ply_template.yaml")).with_input_path(
            input_path
        )

    decoder = SceneDecoder(decoding_params)
    decoder.decode()
    gaussians = decoder.scene

    return gaussians
