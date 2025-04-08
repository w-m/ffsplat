from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]
from torch import Tensor

from ..io.ply import decode_ply
from ..models.gaussians import Gaussians

# Register AVIF support
register_avif_opener()


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
                case {"file_path": file_path, "type": file_type, "field_name": field_name}:
                    match file_type:
                        case "png":
                            img_field = torch.tensor(
                                cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                            )
                        case "avif":
                            img_field = torch.tensor(np.array(Image.open(file_path)))

                    self.fields[field_name] = img_field

                case _:
                    raise ValueError("Unsupported file format")

    def _process_field(self, field_op: dict[str, Any], field_data: Tensor | None) -> Tensor:  # noqa: C901
        match field_op:
            case {"combine": {"from_field_list": from_list, "method": "bytes"}}:
                num_bytes = len(from_list)

                if num_bytes < 2 or num_bytes > 8:
                    raise ValueError("num_bytes must be between 2 and 8")

                byte_tensors: list[Tensor] = [
                    self.fields[source_field_name]
                    for source_field_name in from_list
                    if source_field_name in self.fields
                ]

                target_dtype = torch.int32 if num_bytes <= 4 else torch.int64

                field_data = byte_tensors[0].to(target_dtype)

                for i, byte_tensor in enumerate(byte_tensors):
                    if byte_tensor.dtype != torch.uint8:
                        raise ValueError(f"Source tensor {i} must be of type uint8")
                    field_data = field_data | (byte_tensor.to(target_dtype) << (i * 8))

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

            case {"lookup": {"from_field": from_field, "to_field": to_field}}:
                values = self.fields[to_field][self.fields[from_field].to(torch.int32)]
                field_data = values

            case {"reshape_from_dim": {"start_dim": start_dim, "shape": shape}}:
                if field_data is None:
                    raise ValueError("Field data is None before reshape")
                target_shape = list(field_data.shape[:start_dim]) + list(shape)
                field_data = field_data.reshape(*target_shape)

            case {"permute": {"dims": dims}}:
                if field_data is None:
                    raise ValueError("Field data is None before permute")
                field_data = field_data.permute(*dims)

            case {"remapping": {"method": method, "min": min_val, "max": max_val}}:
                if field_data is None:
                    raise ValueError("Field data is None before remapping")
                field_data_norm = minmax(field_data)
                field_data = (field_data_norm * (max_val - min_val)) + min_val

            case {
                "remapping": {
                    "method": "channelwise-minmax",
                    "min_values": min_values,
                    "max_values": max_values,
                    "dim": dim,
                }
            }:
                if field_data is None:
                    raise ValueError("Field data is None before channelwise remapping")

                min_tensor = torch.tensor(min_values, device=field_data.device, dtype=torch.float32)
                max_tensor = torch.tensor(max_values, device=field_data.device, dtype=torch.float32)

                # create a shape that's 1 everywhere but in dim, where it has the size of min_values
                # e.g. [1, 1, 3] for dim=2 with 3 min_values
                view_shape = [1 if d != dim else -1 for d in range(field_data.dim())]

                min_tensor = min_tensor.view(view_shape)
                max_tensor = max_tensor.view(view_shape)

                field_range = max_tensor - min_tensor
                field_range[field_range == 0] = 1.0

                field_data = minmax(field_data)
                field_data = field_data * field_range + min_tensor

            case {"remapping": {"method": method}}:
                if field_data is None:
                    raise ValueError("Field data is None before remapping")

                match method:
                    case "exp":
                        field_data = torch.exp(field_data)
                    case "sigmoid":
                        field_data = torch.sigmoid(field_data)
                    case "signed-exp":
                        field_data = torch.sign(field_data) * (torch.expm1(torch.abs(field_data)))
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
    elif input_format == "3DGS-INRIA-nosh.ply":
        if input_file_extension == ".ply":
            decoding_params = DecodingParams.from_yaml_file(
                Path("3DGS_INRIA_ply_nosh_decoding_template.yaml")
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
