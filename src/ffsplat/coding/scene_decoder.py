from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]

from ..io.ply import decode_ply
from ..models.fields import Field
from ..models.gaussians import Gaussians
from ..models.operations import Operation

# Register AVIF support
register_avif_opener()


@dataclass
class DecodingParams:
    """Parameters for decoding 3D scene formats."""

    files: list[dict[str, str]]
    ops: dict[str, list[dict[str, Any]]]
    scene: dict[str, Any]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "DecodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(files=data.get("files", []), ops=data.get("ops", {}), scene=data.get("scene", {}))

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


@lru_cache
def process_operation(
    op: Operation,
    verbose: bool = False,
) -> tuple[dict[str, Field]]:
    """Process the operation and return the new fields and decoding updates."""
    print("Cache miss")
    return op.apply(verbose=verbose)[0]


@dataclass
class SceneDecoder:
    decoding_params: DecodingParams
    fields: dict[str, Field] = field(default_factory=dict)
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
                            img_field_data = torch.tensor(
                                cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                            )
                        case "avif":
                            img_field_data = torch.tensor(np.array(Image.open(file_path)))

                    self.fields[field_name] = Field.from_file(img_field_data, file_path)

                case _:
                    raise ValueError("Unsupported file format")

    def _process_fields(self) -> None:
        for op_params in self.decoding_params.ops:
            # build each operation and process it
            input_fields_params = op_params["input_fields"]
            for transform_param in op_params["transforms"]:
                op = Operation.from_json(input_fields_params, transform_param, self.fields)
                new_fields = process_operation(op)
                self.fields.update(new_fields)

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

    def _print_field_stats(self) -> None:
        print("Decoded field statistics:")
        for field_name, field_data in sorted(self.fields.items()):
            stats = f"{field_name}: \t{tuple(field_data.shape)} | {field_data.dtype}"
            if field_data.numel() > 0:
                stats += f" | Min: {field_data.min().item():.4f} | Max: {field_data.max().item():.4f}"
                stats += f" | Median: {field_data.median().item():.4f}"
                # stats += f" | Unique Count: {field_data.unique().numel()}"
            print(stats)

    def decode(self, verbose: bool) -> None:
        self._decode_files()
        self._process_fields()
        if verbose:
            self._print_field_stats()
        self._create_scene()


def decode_gaussians(input_path: Path, input_format: str, verbose: bool) -> Gaussians:
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
    decoder.decode(verbose=verbose)
    gaussians = decoder.scene

    return gaussians
