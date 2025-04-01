from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from torch import Tensor

from ..io.ply import decode_ply
from ..models.gaussians import Gaussians


# TODO duplicated code, scene_encoder
def minmax(tensor: Tensor) -> Tensor:
    """Scale a tensor to the range [0, 1]."""
    tensor = tensor - tensor.min()
    if tensor.max() - tensor.min() > 0:
        tensor = tensor / (tensor.max() - tensor.min())
    return tensor


@dataclass
class DecodingParams:
    """Parameters for decoding 3D scene formats."""

    files: list[dict[str, str]]
    fields: dict[str, list[dict[str, Any]]]
    scene: dict[str, Any]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "DecodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(files=data.get("files", []), fields=data.get("fields", {}), scene=data.get("scene", {}))

    @classmethod
    def from_container_folder(cls, folder_path: Path) -> "DecodingParams":
        dec_params = cls.from_yaml_file(folder_path / "container_meta.yaml")
        for file in dec_params.files:
            file_path = folder_path / file["file_path"]
            file["file_path"] = str(file_path)
        return dec_params

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
    scene: Gaussians = field(init=False)

    def _decode_files(self) -> None:
        for file in self.decoding_params.files:
            match file:
                case {"file_path": file_path, "type": "ply", "field_prefix": field_prefix}:
                    ply_fields = decode_ply(file_path=Path(file_path), field_prefix=field_prefix)
                    self.fields.update(ply_fields)
                case {"file_path": file_path, "type": "png", "field_name": field_name}:
                    png_field = torch.tensor(
                        cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                    )
                    self.fields[field_name] = png_field
                case _:
                    raise ValueError("Unsupported file format")

    def _process_field(self, field_op: dict[str, Any], field_data: Tensor | None) -> Tensor:  # noqa: C901
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

            case {"flatten": {"start_dim": start_dim, "end_dim": end_dim}}:
                if field_data is None:
                    raise ValueError("Field data is None before flatten")
                field_data = field_data.flatten(start_dim=start_dim, end_dim=end_dim)

            case {"from_field": name}:
                if name in self.fields:
                    field_data = self.fields[name]
                else:
                    raise ValueError(f"Field not found: {name}")

            case {"reshape": {"shape": shape}}:
                if field_data is None:
                    raise ValueError("Field data is None before reshape")
                field_data = field_data.reshape(*shape)

            case {"permute": {"dims": dims}}:
                if field_data is None:
                    raise ValueError("Field data is None before permute")
                field_data = field_data.permute(*dims)

            case {"remapping": {"method": method, "min": min_val, "max": max_val}}:
                if field_data is None:
                    raise ValueError("Field data is None before remapping")
                field_data_norm = minmax(field_data)
                field_data = (field_data_norm * (max_val - min_val)) + min_val

            case {"remapping": {"method": method}}:
                if field_data is None:
                    raise ValueError("Field data is None before remapping")

                match method:
                    case "exp":
                        field_data = torch.exp(field_data)
                    case "sigmoid":
                        field_data = torch.sigmoid(field_data)
                    case _:
                        raise ValueError(f"Unsupported remapping method: {method}")
            case {"assign": {"field_name": field_name}}:
                if field_name in self.fields:
                    field_data = self.fields[field_name]
                else:
                    raise ValueError(f"Field not found for assignment: {field_name}")

            case {"to_dtype": {"dtype": dtype_str}}:
                if field_data is None:
                    raise ValueError("Field data is None before dtype conversion")
                match dtype_str:
                    case "uint8":
                        if field_data.min() < 0 or field_data.max() > 255:
                            raise ValueError(
                                f"Field data out of range for uint8 conversion: {field_data.min().item()} - {field_data.max().item()}"
                            )
                        field_data = field_data.to(torch.uint8)
                    case "uint16":
                        if field_data.min() < 0 or field_data.max() > 65535:
                            raise ValueError(
                                f"Field data out of range for uint16 conversion: {field_data.min().item()} - {field_data.max().item()}"
                            )
                        field_data = field_data.to(torch.uint16)
                    case "float32":
                        field_data = field_data.to(torch.float32)
                    case _:
                        raise ValueError(f"Unsupported dtype for conversion: {dtype_str}")

            case _:
                raise ValueError(f"Unsupported field operation: {field_op}")

        if field_data is None:
            raise ValueError("Field data is None after processing field operation")
        return field_data

    def _process_fields(self) -> None:
        for field_name, field_ops in self.decoding_params.fields.items():
            field_data = self.fields.get(field_name, None)

            for field_op in field_ops:
                field_data = self._process_field(field_op, field_data)

            if field_data is not None:
                self.fields[field_name] = field_data
            else:
                raise ValueError(f"Empty field after processing field operations: {field_name}")

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

    if input_format == "3DGS-INRIA.ply":
        if input_file_extension == ".ply":
            decoding_params = DecodingParams.from_yaml_file(
                Path("3DGS_INRIA_ply_decoding_template.yaml")
            ).with_input_path(input_path)
        else:
            raise ValueError("Input file must be a .ply file for 3DGS-INRIA format")
    elif input_format == "smurfx":
        if not input_path.is_dir():
            raise ValueError("Input path must be a directory for smurfx format")
        decoding_params = DecodingParams.from_container_folder(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    decoder = SceneDecoder(decoding_params)
    decoder.decode()
    gaussians = decoder.scene

    return gaussians
