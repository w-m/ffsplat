from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]

from ..models.fields import Field
from ..models.gaussians import Gaussians
from ..models.operations import Operation

# Register AVIF support
register_avif_opener()


@dataclass
class DecodingParams:
    """Parameters for decoding 3D scene formats."""

    files: list[dict[str, str]]
    ops: list[dict[str, Any]]
    scene: dict[str, Any]

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "DecodingParams":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(files=data.get("files", []), ops=data.get("ops", {}), scene=data.get("scene", {}))

    @classmethod
    def from_container_folder(cls, folder_path: Path) -> "DecodingParams":
        dec_params = cls.from_yaml_file(folder_path / "container_meta.yaml")
        for op_params in dec_params.ops:
            for transform_params in op_params["transforms"]:
                if "read_file" in transform_params:
                    file = transform_params["read_file"]
                    file["file_path"] = str(folder_path / file["file_path"])
        return dec_params

    def with_input_path(self, input_path: Path) -> "DecodingParams":
        read_file_params = self.ops[0]["transforms"][0]["read_file"]
        file_type = read_file_params.get("type", None)
        if file_type is None or file_type != "ply":
            raise ValueError("Expected a ply file read as first operation")

        self.ops[0]["transforms"][0]["read_file"]["file_path"] = str(input_path)
        return self


@lru_cache
def process_operation(
    op: Operation,
    verbose: bool = False,
) -> dict[str, Field]:
    """Process the operation and return the new fields and decoding updates."""
    print("Cache miss")
    return op.apply(verbose=verbose)[0]


@dataclass
class SceneDecoder:
    decoding_params: DecodingParams
    fields: dict[str, Field] = field(default_factory=dict)
    scene: Gaussians = field(init=False)

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
        for field_name, field_obj in sorted(self.fields.items()):
            stats = f"{field_name}: \t{tuple(field_obj.data.shape)} | {field_obj.data.dtype}"
            if field_obj.data.numel() > 0:
                stats += f" | Min: {field_obj.data.min().item():.4f} | Max: {field_obj.data.max().item():.4f}"
                stats += f" | Median: {field_obj.data.median().item():.4f}"
                # stats += f" | Unique Count: {field_obj.data.unique().numel()}"
            print(stats)

    def decode(self, verbose: bool) -> None:
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
