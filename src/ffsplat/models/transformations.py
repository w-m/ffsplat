import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]
from plas import sort_with_plas  # type: ignore[import-untyped]
from torch import Tensor
from torchpq.clustering import KMeans  # type: ignore[import-untyped]

from ..io.ply import decode_ply, encode_ply
from ..models.fields import Field

if TYPE_CHECKING:
    from ..models.fields import Operation


def convert_to_dtype(field_data: Tensor, dtype_str: str) -> Tensor:
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
        case "int32":
            if field_data.min() < -2147483648 or field_data.max() > 2147483647:
                raise ValueError(
                    f"Field data out of range for int32 conversion: {field_data.min().item()} - {field_data.max().item()}"
                )
            field_data = field_data.to(torch.int32)
        case "uint32":
            if field_data.min() < 0 or field_data.max() > 4294967295:
                raise ValueError(
                    f"Field data out of range for uint32 conversion: {field_data.min().item()} - {field_data.max().item()}"
                )
            field_data = field_data.to(torch.uint32)
        case "float32":
            field_data = field_data.to(torch.float32)
        case _:
            raise ValueError(f"Unsupported dtype for quantization: {dtype_str}")

    return field_data


def write_image(output_file_path: Path, field_data: Tensor, file_type: str, coding_params: dict[str, Any]) -> None:
    match file_type:
        case "png":
            cv2.imwrite(
                str(output_file_path),
                field_data.cpu().numpy(),
                [cv2.IMWRITE_PNG_COMPRESSION, coding_params.get("compression_level", 3)],
            )
        case "avif":
            # TODO: only do this once?
            register_avif_opener()
            Image.fromarray(field_data.cpu().numpy()).save(
                output_file_path,
                format="AVIF",
                quality=coding_params.get("quality", -1),
                chroma=coding_params.get("chroma", 444),
                matrix_coefficients=coding_params.get("matrix_coefficients", 0),
            )
        case "webp":
            Image.fromarray(field_data.cpu().numpy()).save(
                output_file_path,
                format="WEBP",
                quality=coding_params.get("quality", 100),
                lossless=coding_params.get("lossless", True),
                method=coding_params.get("method", 6),
                exact=coding_params.get("exact", True),
            )
        case _:
            raise ValueError(f"Unsupported file type: {file_type}")


def write_json(output_file_path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=2)


@dataclass
class PLASConfig:
    """Configuration for PLAS sorting."""

    prune_by: str
    scaling_fn: str
    # activated: bool
    shuffle: bool
    improvement_break: float
    # this is not needed for plas, just to know where to store the result
    to_field: str
    weights: dict[str, float] = field(default_factory=dict)


def standardize(tensor: Tensor) -> Tensor:
    """Standardize a tensor by removing the mean and scaling to unit variance."""
    tensor = tensor - tensor.mean()
    std = tensor.std(unbiased=False)
    if std > 0:
        tensor = tensor / std
    return tensor


def minmax(tensor: Tensor) -> Tensor:
    """Scale a tensor to the range [0, 1]."""
    tensor = tensor - tensor.min()
    if tensor.max() - tensor.min() > 0:
        tensor = tensor / (tensor.max() - tensor.min())
    return tensor


class Transformation(ABC):
    @staticmethod
    @abstractmethod
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        """Apply the transformation to the input fields. returns new/updated fields and decoding updates.
        Transformations that are only available for decoding do return empty decoding updates"""
        pass

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""
        return []


class Cluster(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        # Implement the clustering logic here
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {
                "method": "kmeans",
                "num_clusters": num_clusters,
                "distance": distance,
                "to_fields_with_prefix": to_fields_with_prefix,
            }:
                kmeans = KMeans(n_clusters=num_clusters, distance=distance, verbose=True)
                labels = kmeans.fit(field_data.permute(1, 0).contiguous())
                centroids = kmeans.centroids.permute(1, 0)
                new_fields[f"{to_fields_with_prefix}labels"] = Field(labels, parentOp)
                new_fields[f"{to_fields_with_prefix}centroids"] = Field(centroids, parentOp)
                decoding_update.append({
                    "input_fields": [f"{to_fields_with_prefix}labels", f"{to_fields_with_prefix}centroids"],
                    "transforms": [
                        {
                            "lookup": {
                                "labels": f"{to_fields_with_prefix}labels",
                                "table": f"{to_fields_with_prefix}centroids",
                                "to_field": field_name,
                            }
                        }
                    ],
                })
            case _:
                raise ValueError(f"Unknown clustering method: {params["method"]}")
        return new_fields, decoding_update

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""
        dynamic_params_config: list[dict[str, Any]] = []
        dynamic_params_config.append({
            "label": "#clusters",
            "type": "number",
            "min": 1,
            "max": 16,
            "step": 1,
            "dtype": int,
            "set": "num_clusters",
            "mapping": lambda x: 2**x,
            "inverse_mapping": lambda x: np.log2(x),
        })
        dynamic_params_config.append({
            "label": "distance",
            "type": "dropdown",
            "values": ["euclidean", "cosine", "manhattan"],
        })

        return dynamic_params_config


class Split(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {
                "split_size_or_sections": split_size_or_sections,
                "dim": dim,
                "squeeze": squeeze,
                "to_field_list": to_field_list,
            }:
                chunks = field_data.split(split_size_or_sections, dim)
                for target_field_name, chunk in zip(to_field_list, chunks, strict=False):
                    if target_field_name == "_":
                        continue
                    if squeeze:
                        chunk = chunk.squeeze(dim)
                    new_fields[target_field_name] = Field(chunk, parentOp)
                to_field_list = [name for name in to_field_list if name != "_"]

                decoding_update.append({
                    "input_fields": to_field_list,
                    "transforms": [
                        {
                            "combine": {
                                "method": "stack" if split_size_or_sections == 1 else "concat",
                                "dim": dim,
                                "to_field": field_name,
                            }
                        }
                    ],
                })

            # TODO: should squeeze be optional?
            case {
                "split_size_or_sections": split_size_or_sections,
                "dim": dim,
                "to_fields_with_prefix": to_fields_with_prefix,
            }:
                chunks = field_data.split(split_size_or_sections, dim)
                for i, chunk in enumerate(chunks):
                    new_fields[f"{to_fields_with_prefix}{i}"] = Field(chunk.squeeze(dim), parentOp)

                decoding_update.append({
                    "input_fields": {"from_fields_with_prefix": to_fields_with_prefix},
                    "transforms": [
                        {
                            "combine": {
                                "method": "stack" if split_size_or_sections == 1 else "concat",
                                "dim": dim,
                                "to_field": field_name,
                            }
                        }
                    ],
                })

            case _:
                raise ValueError(f"Unknown split parameters: {params}")
        return new_fields, decoding_update


class Remapping(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"method": "exp"}:
                field_data = torch.exp(field_data)
            case {"method": "sigmoid"}:
                field_data = torch.sigmoid(field_data)
            case {"method": "signed-exp"}:
                field_data = torch.sign(field_data) * (torch.expm1(torch.abs(field_data)))
            case {"method": "log"}:
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"remapping": {"method": "exp"}}],
                })
                field_data = field_data.log()
                if torch.isnan(field_data).any() or torch.isinf(field_data).any():
                    raise ValueError("Can't apply log to values <= 0. You might want to try signed-log")

            case {"method": "signed-log"}:
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"remapping": {"method": "signed-exp"}}],
                })
                field_data = torch.sign(field_data) * torch.log1p(torch.abs(field_data))

            case {"method": "inverse-sigmoid"}:
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"remapping": {"method": "sigmoid"}}],
                })
                # ensure that we can encode opacity values in the full range [0, 1]
                # by clamping the values to (eps, 1 - eps) -> no infinite values from 0.0, 1.0
                eps = 1e-6
                field_data = field_data.clamp(eps, 1 - eps)
                field_data = torch.log(field_data / (1 - field_data))

            case {"method": "minmax", "min": min_val, "max": max_val}:
                field_min = field_data.min().item()
                field_max = field_data.max().item()

                min_val_f = float(min_val)
                max_val_f = float(max_val)

                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [
                        {
                            "remapping": {
                                "method": "minmax",
                                "min": field_min,
                                "max": field_max,
                            }
                        }
                    ],
                })

                field_data = minmax(field_data)
                field_data = field_data * (max_val_f - min_val_f) + min_val_f

            case {"method": "channelwise-minmax", "min": min_val, "max": max_val, "dim": dim}:
                min_val_f = float(min_val)
                max_val_f = float(max_val)

                min_vals = torch.amin(field_data, dim=[d for d in range(field_data.ndim) if d != dim])
                max_vals = torch.amax(field_data, dim=[d for d in range(field_data.ndim) if d != dim])

                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [
                        {
                            "remapping": {
                                "method": "channelwise-minmax",
                                "min_values": min_vals.tolist(),
                                "max_values": max_vals.tolist(),
                                "dim": dim,
                            }
                        }
                    ],
                })

                field_range = max_vals - min_vals
                field_range[field_range == 0] = 1.0

                normalized = (field_data - min_vals) / field_range
                normalized = normalized * (max_val_f - min_val_f) + min_val_f

                field_data = normalized
            case _:
                raise ValueError(f"Unknown remapping parameters: {params}")
        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""

        dynamic_params_config: list[dict[str, Any]] = []
        dynamic_params_config.append({
            "label": "method",
            "type": "dropdown",
            "values": ["log", "signed-log", "inverse-sigmoid", "minmax"],
        })
        if params.get("method") == "minmax":
            dynamic_params_config.append({
                "label": "min",
                "type": "number",
                "min": 0,
                "max": 4294967295,  # unit32 max
                "step": 1,
                "dtype": int,
                "set": "min",
            })
            dynamic_params_config.append({
                "label": "max",
                "type": "number",
                "min": 0,
                "max": 4294967295,  # unit32 max
                "step": 1,
                "dtype": int,
                "set": "max",
            })
        return dynamic_params_config


class Reparametrize(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        match params:
            case {"method": "unit_sphere", "dim": dim}:
                field_data = field_data / torch.linalg.norm(field_data, dim=dim, keepdim=True)

                sign_mask = field_data.select(dim, 3) < 0
                field_data[sign_mask] *= -1
                new_fields[field_name] = Field(field_data, parentOp)

            case {"method": "pack_quaternions", "to_fields_with_prefix": to_fields_with_prefix, "dim": dim}:
                # Ensure the first component is positive
                sign = field_data.select(dim, 0).sign()
                sign[sign == 0] = 1
                q_signed = field_data * sign.unsqueeze(-1)

                # Drop the first component (is always the largest)
                values = q_signed.narrow(dim, 1, 3)
                max_idx = torch.zeros_like(sign, dtype=values.dtype).unsqueeze(-1)

                new_fields[f"{to_fields_with_prefix}indices"] = Field(max_idx, parentOp)
                new_fields[f"{to_fields_with_prefix}values"] = Field(values, parentOp)
                decoding_update.append({
                    "input_fields": [f"{to_fields_with_prefix}indices", f"{to_fields_with_prefix}values"],
                    "transforms": [
                        {
                            "reparametize": {
                                "method": "unpack_quaternions",
                                "from_fields_with_prefix": to_fields_with_prefix,
                                "dim": -1,
                                "to_field_name": field_name,
                            }
                        }
                    ],
                })

            case {
                "method": "unpack_quaternions",
                "from_fields_with_prefix": from_fields_with_prefix,
                "dim": dim,
                "to_field_name": to_field_name,
            }:
                values_field = input_fields[f"{from_fields_with_prefix}values"].data

                if values_field is None:
                    raise ValueError("values field data is None before unit sphere recovery")
                if values_field.shape[dim] != 3:
                    raise ValueError(f"Field data shape mismatch for unit sphere recovery: {field_data.shape}")

                partial_norm_sq = (values_field**2).sum(dim=dim, keepdim=True)
                # Recover the missing component (always non-negative)
                w = torch.sqrt(torch.clamp(1.0 - partial_norm_sq, min=0.0))

                num_components = values_field.shape[-1] + 1  # Original data had one more component
                reconstructed = torch.zeros(
                    *values_field.shape[:-1], num_components, dtype=values_field.dtype, device=values_field.device
                )

                # wxyz convention
                reconstructed[..., 0] = w.squeeze(-1)
                reconstructed[..., 1:] = values_field

                # Optional re-normalization to mitigate numerical drift
                field_data = reconstructed / torch.linalg.norm(reconstructed, dim=dim, keepdim=True)
                new_fields[to_field_name] = Field(field_data, parentOp)

            case _:
                raise ValueError(f"Unknown ToField parameters: {params}")

        return new_fields, decoding_update


class ToField(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"to_field_name": to_field_name}:
                decoding_update.append({
                    "input_fields": [to_field_name],
                    "transforms": [{"to_field": {"to_field_name": field_name}}],
                })
                new_fields[to_field_name] = Field(field_data, parentOp)
            case _:
                raise ValueError(f"Unknown ToField parameters: {params}")

        return new_fields, decoding_update


class Flatten(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data
        if field_data is None:
            raise ValueError("Field data is None before flattening")

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"start_dim": start_dim, "end_dim": end_dim}:
                target_shape = field_data.shape
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"reshape": {"shape": target_shape}}],
                })
                field_data = field_data.flatten(start_dim=start_dim, end_dim=end_dim)
            case {"start_dim": start_dim}:
                target_shape = field_data.shape[start_dim:]
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"reshape": {"start_dim": start_dim, "shape": target_shape}}],
                })
                field_data = field_data.flatten(start_dim=start_dim)
            case _:
                raise ValueError(f"Unknown Flatten parameters: {params}")

        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class Reshape(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"start_dim": start_dim, "shape": shape}:
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"reshape": {"shape": field_data.shape}}],
                })
                target_shape = list(field_data.shape[:start_dim]) + list(shape)
                field_data = field_data.reshape(*target_shape)
            case {"shape": shape}:
                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [{"reshape": {"shape": field_data.shape}}],
                })
                field_data = field_data.reshape(*shape)
            case _:
                raise ValueError(f"Unknown Reshape parameters: {params}")

        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class Permute(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        dims = params.get("dims")
        if dims is None:
            raise ValueError(f"Unknown Permute parameters: {params}")
        decoding_update.append({
            "input_fields": [field_name],
            "transforms": [{"permute": {"dims": dims}}],
        })
        field_data = field_data.permute(*dims)

        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class ToDType(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        dtype_str = params.get("dtype")
        round_to_int = params.get("round_to_int", False)

        if dtype_str is None:
            raise ValueError(f"Unknown ToDType parameters: {params}")

        torch_dtype_to_str = {
            torch.float32: "float32",
            torch.uint8: "uint8",
            torch.uint16: "uint16",
            torch.int32: "int32",
            torch.uint32: "uint32",
        }

        if round_to_int:
            field_data = torch.round(field_data)

        decoding_update.append({
            "input_fields": [field_name],
            "transforms": [{"to_dtype": {"dtype": torch_dtype_to_str[field_data.dtype]}}],
        })
        field_data = convert_to_dtype(field_data, dtype_str)

        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class SplitBytes(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        match params:
            case {"num_bytes": num_bytes, "to_fields_with_prefix": to_fields_with_prefix}:
                num_bytes = int(num_bytes)

                if num_bytes < 2 or num_bytes > 8:
                    raise ValueError("num_bytes must be between 2 and 8")

                if torch.is_floating_point(field_data):
                    raise ValueError(f"Field data must be an integer data type, got {field_data.dtype}")

                field_data = field_data.to(torch.int32) if num_bytes <= 4 else field_data.to(torch.int64)

                res_field_names = []

                for i in range(num_bytes):
                    mask = (field_data >> (i * 8)) & 0xFF
                    res_field_name = f"{to_fields_with_prefix}{i}"
                    res_field_names.append(res_field_name)
                    new_fields[res_field_name] = Field(mask.to(torch.uint8), parentOp)

                # TODO: why from_field_list and not from_fields_with_prefix?
                decoding_update.append({
                    "input_fields": res_field_names,
                    "transforms": [
                        {
                            "combine": {
                                "method": "bytes",
                                "to_field": field_name,
                            }
                        }
                    ],
                })
            case _:
                raise ValueError(f"Unknown SplitBytes parameters: {params}")

        return new_fields, decoding_update


class Reindex(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        # TODO: make this compatible with 1D-tensors
        match params:
            case {
                "src_field": src_field_name,
                "index_field": index_field_name,
            }:
                index_field_obj = input_fields[index_field_name]
                original_data = input_fields[src_field_name].data
                new_fields[src_field_name] = Field(original_data[index_field_obj.data], parentOp)

                if index_field_obj.data.ndim > 1:
                    decoding_update.append({
                        "input_fields": [src_field_name],
                        "transforms": [{"flatten": {"start_dim": 0, "end_dim": 1}}],
                    })
            case _:
                raise ValueError(f"Unknown Reindex parameters: {params}")

        return new_fields, decoding_update


class Sort(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        sorted_indices = None
        if field_data is None:
            raise ValueError("Field data is None before sorting")
        prefix = params["to_fields_with_prefix"]
        match params:
            case {"method": "lexicographic", "labels": labels_name, "target": target_name}:
                target = input_fields[target_name].data
                sorted_indices = np.lexsort(target.permute(dims=(1, 0)).cpu().numpy())
                sorted_indices = torch.tensor(sorted_indices, device=field_data.device)
                inverse_sorted_indices = torch.argsort(sorted_indices)

                if labels_name != "_":
                    original_labels = input_fields[labels_name].data
                    # orignal_labels_grid = PLAS.primitive_filter_pruning_to_square_shape(original_labels,verbose=verbose)
                    original_labels_grid = PLAS.as_grid_img(original_labels)
                    updated_labels = inverse_sorted_indices[original_labels_grid]
                    new_fields[labels_name] = Field(updated_labels, parentOp)
                    decoding_update.append({
                        "input_fields": [labels_name],
                        "transforms": [{"flatten": {"start_dim": 0, "end_dim": 1}}],
                    })
            case {"method": "plas"}:
                plas_cfg = {k: v for k, v in params.items() if k not in ["method", "to_fields_with_prefix"]}
                plas_cfg["to_field"] = f"{prefix}indices"
                sorted_indices = PLAS.plas_preprocess(
                    plas_cfg=PLASConfig(**plas_cfg),
                    fields=parentOp.input_fields,
                    verbose=verbose,
                )
                inverse_sorted_indices = torch.argsort(sorted_indices)

            case _:
                raise ValueError(f"Unknown Sort parameters: {params}")
        new_fields[f"{prefix}indices"] = Field(sorted_indices, parentOp)
        new_fields[f"{prefix}indices_inverse"] = Field(inverse_sorted_indices, parentOp)

        return new_fields, decoding_update

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""
        dynamic_params_config: list[dict[str, Any]] = []
        dynamic_params_config.append({
            "label": "method",
            "type": "dropdown",
            "values": ["lexicographic", "plas"],
        })
        if params.get("method") == "plas":
            dynamic_params_config.extend(PLAS.get_dynamic_params(params))

        return dynamic_params_config


class PLAS:
    @staticmethod
    def as_grid_img(tensor: Tensor) -> Tensor:
        num_primitives = tensor.shape[0]
        grid_sidelen = int(math.sqrt(num_primitives))
        if grid_sidelen * grid_sidelen != num_primitives:
            raise ValueError(
                f"Number of primitives {num_primitives} is not a perfect square. Cannot reshape to grid image."
            )
        tensor = tensor.reshape(grid_sidelen, grid_sidelen, *tensor.shape[1:])
        return tensor

    @staticmethod
    def primitive_filter_pruning_to_square_shape(data: Tensor, verbose: bool) -> Tensor | None:
        """Returning None indicates that no primitives need to be pruned"""
        num_primitives = data.shape[0]

        grid_sidelen = int(math.sqrt(num_primitives))
        num_to_remove = num_primitives - grid_sidelen * grid_sidelen

        if num_to_remove == 0:
            return None

        if verbose:
            print(
                f"Removing {num_to_remove}/{num_primitives} primitives to fit the grid. ({100 * num_to_remove / num_primitives:.4f}%)"
            )

        _, keep_indices = torch.topk(data.squeeze(), k=grid_sidelen * grid_sidelen)
        sorted_keep_indices = torch.sort(keep_indices)[0]
        return sorted_keep_indices

    @staticmethod
    def plas_preprocess(plas_cfg: PLASConfig, fields: dict[str, Field], verbose: bool) -> Tensor:
        primitive_filter = PLAS.primitive_filter_pruning_to_square_shape(fields[plas_cfg.prune_by].data, verbose)

        # TODO untested
        match plas_cfg.scaling_fn:
            case "standardize":
                normalization_fn: Callable[[Tensor], Tensor] = standardize
            case "minmax":
                normalization_fn = minmax
            case "none":
                normalization_fn = lambda x: x
            case _:
                raise ValueError(f"Unsupported scaling function: {plas_cfg.scaling_fn}")

        # attr_getter_fn = get_activated_attr_flat if sorting_cfg.activated else get_attr_flat

        attr_getter_fn = (
            lambda attr_name: fields[attr_name].data[primitive_filter]
            if primitive_filter is not None
            else fields[attr_name].data
        )

        params_to_sort: list[Tensor] = []

        for attr_name, attr_weight in plas_cfg.weights.items():
            if attr_weight > 0:
                params_to_sort.append(
                    normalization_fn(attr_getter_fn(attr_name).to(torch.float32)).flatten(start_dim=1) * attr_weight
                )

        params_tensor = torch.cat(params_to_sort, dim=1)

        if plas_cfg.shuffle:
            # TODO shuffling should be an option of sort_with_plas
            torch.manual_seed(42)
            shuffled_indices = torch.randperm(params_tensor.shape[0], device=params_tensor.device)
            params_tensor = params_tensor[shuffled_indices]

        grid_to_sort = PLAS.as_grid_img(params_tensor).permute(2, 0, 1)
        _, sorted_indices_ret = sort_with_plas(
            grid_to_sort, improvement_break=float(plas_cfg.improvement_break), verbose=verbose
        )

        sorted_indices: Tensor = sorted_indices_ret.squeeze(0).to(params_tensor.device)

        if plas_cfg.shuffle:
            flat_indices = sorted_indices.flatten()
            unshuffled_flat_indices = shuffled_indices[flat_indices]
            sorted_indices = unshuffled_flat_indices.reshape(sorted_indices.shape)

        if primitive_filter is not None:
            return primitive_filter[sorted_indices]

        return sorted_indices

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""

        # if params.get("weights") is None:
        # raise ValueError(f"PLAS parameters is missing weights: {params}")
        field_names = list(params["weights"].keys())
        scaling_functions = ["standardize", "minmax", "none"]

        dynamic_params_config: list[dict[str, Any]] = []

        params.setdefault("scaling_fn", "standardize")
        params.setdefault("shuffle", True)
        params.setdefault("improvement_break", 1e-4)
        # coding_params.setdefault("quality", -1)
        # coding_params.setdefault("chroma", 444)
        # coding_params.setdefault("matrix_coefficients", 0)
        # TODO: initial values for scaling function, improvement break!!!!
        dynamic_params_config.append({
            "label": "scaling_fn",
            "type": "dropdown",
            "values": scaling_functions,
        })
        dynamic_params_config.append({
            "label": "shuffle",
            "type": "bool",
        })
        dynamic_params_config.append({
            "label": "improvement_break",
            "type": "number",
            "min": -7,
            "max": -1,
            "step": 1,
            "dtype": int,
            "mapping": lambda x: 10**x,
            "inverse_mapping": lambda x: np.log10(x),
        })

        weight_config: list[dict[str, Any]] = []
        # if params.get("weights") is None:
        # params.
        for field_name in field_names:
            weight_config.append({
                "label": field_name,
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "dtype": float,
            })
        dynamic_params_config.append({
            "label": "weights",
            "type": "heading",
            "params": weight_config,
        })

        return dynamic_params_config


class Combine(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        new_fields: dict[str, Field] = {}
        field_data: Tensor = torch.empty(0)
        decoding_update: list[dict[str, Any]] = []

        match params:
            case {"method": "bytes", "to_field": to_field_name}:
                num_bytes = len(parentOp.input_fields)

                if num_bytes < 2 or num_bytes > 8:
                    raise ValueError("num_bytes must be between 2 and 8")

                byte_tensors: list[Tensor] = [
                    parentOp.input_fields[source_field_name].data for source_field_name in parentOp.input_fields
                ]

                target_dtype = torch.int32 if num_bytes <= 4 else torch.int64

                field_data = byte_tensors[0].to(target_dtype)

                for i, byte_tensor in enumerate(byte_tensors):
                    if byte_tensor.dtype != torch.uint8:
                        raise ValueError(f"Source tensor {i} must be of type uint8")
                    field_data = field_data | (byte_tensor.to(target_dtype) << (i * 8))

                new_fields[to_field_name] = Field(field_data, parentOp)

            case {"method": method, "dim": dim, "to_field": to_field_name}:
                tensors: list[Tensor] = [
                    parentOp.input_fields[source_field_name].data for source_field_name in parentOp.input_fields
                ]
                decoding_update_dict = {
                    "input_fields": [to_field_name],
                    "transforms": [
                        {
                            "split": {
                                "dim": dim,
                                "to_field_list": list(parentOp.input_fields),
                            }
                        }
                    ],
                }
                if method == "stack":
                    field_data = torch.stack(tensors, dim=dim)
                    decoding_update_dict["transforms"][0]["split"]["squeeze"] = (True,)
                elif method == "concat":
                    field_data = torch.cat(tensors, dim=dim)
                    decoding_update_dict["transforms"][0]["split"]["squeeze"] = False
                elif method == "stack-zeros":
                    zeros = torch.zeros(tensors[0].shape, dtype=tensors[0].dtype, device=tensors[0].device)
                    tensors.append(zeros)
                    field_data = torch.stack(tensors, dim=dim)
                    decoding_update_dict["transforms"][0]["split"]["squeeze"] = True
                    decoding_update_dict["transforms"][0]["split"]["to_field_list"] = [
                        *list(parentOp.input_fields),
                        "_",
                    ]

                else:
                    raise ValueError(f"Unsupported combine method: {method}")
                decoding_update_dict["transforms"][0]["split"]["split_size_or_sections"] = [
                    t.shape[dim] if dim < len(t.shape) else 1 for t in tensors
                ]
                decoding_update.append(decoding_update_dict)
                new_fields[to_field_name] = Field(field_data, parentOp)
            case _:
                raise ValueError(f"Unknown Combine parameters: {params}")

        return new_fields, decoding_update


class Lookup(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        new_fields: dict[str, Field] = {}
        match params:
            case {"labels": from_field, "table": to_field, "to_field": to_field_name}:
                values = input_fields[to_field].data[input_fields[from_field].data.to(torch.int32)]
                new_fields[to_field_name] = Field(values, parentOp)
            case _:
                raise ValueError(f"Unknown lookup parameters: {params}")

        return new_fields, []


class WriteFile(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any],
        parentOp: "Operation",
        verbose: bool = False,
        **kwargs: Any,
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        decoding_ops: list[dict[str, Any]] = kwargs.get("decoding_ops", [])
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"type": "ply", "file_path": file_path, "base_path": base_path, "field_prefix": field_prefix}:
                fields_to_write = {name[len(field_prefix) :]: field for name, field in parentOp.input_fields.items()}
                decoding_update.append({
                    "input_fields": [],
                    "transforms": [
                        {"read_file": {"file_path": file_path, "type": "ply", "field_prefix": field_prefix}}
                    ],
                })

                output_file_path = Path(base_path) / Path(file_path)

                encode_ply(fields=fields_to_write, path=output_file_path)
            case {"type": "image", "image_codec": codec, "coding_params": coding_params, "base_path": base_path}:
                for field_name, field_obj in parentOp.input_fields.items():
                    field_data = field_obj.data
                    if field_data.shape[-1] == 1:
                        field_data = field_data.squeeze(-1)
                    file_path = f"{field_name}.{codec}"
                    output_file_path = Path(base_path) / Path(file_path)
                    write_image(
                        output_file_path,
                        field_data,
                        codec,
                        coding_params if isinstance(coding_params, dict) else {},
                    )

                    decoding_update.append({
                        "input_fields": [],
                        "transforms": [
                            {
                                "read_file": {
                                    "file_path": file_path,
                                    "type": "image",
                                    "image_codec": codec,
                                    "field_name": field_name,
                                }
                            }
                        ],
                    })
            case {"type": "canvas-metadata", "base_path": base_path}:
                file_path = "meta.json"
                output_file_path = Path(base_path) / Path(file_path)
                field_names = list(parentOp.input_fields.keys())
                meta: dict[str, Any] = {}
                # Readfile no input_fields
                for field_name in field_names:
                    field = parentOp.input_fields[field_name]
                    shape = field.data.shape
                    shape_meta = [int(np.array(shape[:-1]).prod()), shape[-1]]
                    if field_name == "sh0":
                        shape_meta = [shape_meta[0], 1, shape_meta[1]]
                    # to keep the order semi-consistent with original sogs
                    meta[field_name] = {"shape": shape_meta, "dtype": None, "mins": [], "maxs": [], "files": []}

                    for op in decoding_ops:
                        transforms_str = [next(iter(t_str_braced.keys())) for t_str_braced in op["transforms"]]
                        transform_types = [transformation_map[transform] for transform in transforms_str]
                        input_field = op["input_fields"][0] if len(op["input_fields"]) > 0 else ""
                        for idx, (t_str, t) in enumerate(zip(transforms_str, transform_types, strict=False)):
                            if t is ReadFile:
                                field_name_read = op["transforms"][idx][t_str].get("field_name")
                                if field_name_read.startswith(field_name):
                                    codec = op["transforms"][idx][t_str].get("image_codec")
                                    meta[field_name]["files"].append(f"{field_name_read}.{codec}")
                            elif (
                                (t is SimpleQuantize)
                                and input_field.startswith(field_name)
                                and not input_field.endswith("labels")
                            ):
                                meta[field_name]["dtype"] = op["transforms"][idx][t_str]["dtype"]
                                meta[field_name]["mins"] = op["transforms"][idx][t_str]["min_values"]
                                meta[field_name]["maxs"] = op["transforms"][idx][t_str]["max_values"]
                            elif t is Reparametrize:
                                if op["transforms"][idx][t_str].get("method") == "unpack_quaternions":
                                    to_field_name = op["transforms"][idx][t_str].get("to_field_name")
                                    if to_field_name == field_name:
                                        meta[to_field_name]["encoding"] = "quaternion_packed"

                write_json(output_file_path, meta)
            case _:
                raise ValueError(f"Unknown WriteFile parameters: {params}")

        return {}, decoding_update

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""
        file_type = params.get("type")

        dynamic_params_config: list[dict[str, Any]] = []

        if file_type == "image":
            dynamic_params_config.append({
                "label": "image_codec",
                "type": "dropdown",
                "values": ["avif", "png", "webp"],
                "rebuild": True,
            })
            # coding_params for image file
            dynamic_coding_params: list[dict[str, Any]] = []

            dynamic_coding_params: list[dict[str, Any]] = []
            coding_params: dict[str, Any] = params.get("coding_params", {})
            match params["image_codec"]:
                case "avif":
                    # check whether we need to update the coding params default:
                    coding_params: dict[str, Any] = params.get("coding_params", {})

                    if not all(key in coding_params for key in ["quality", "chroma", "matrix_coefficients"]):
                        coding_params.clear()
                        coding_params["quality"] = -1
                        coding_params["chroma"] = 444
                        coding_params["matrix_coefficients"] = 0

                    dynamic_coding_params.append({
                        "label": "chroma",
                        "type": "dropdown",
                        "values": ["0", "420", "444"],
                        "data_type": int,
                    })
                    dynamic_coding_params.append({
                        "label": "matrix_coefficients",
                        "type": "dropdown",
                        "values": ["0", "1"],
                        "data_type": int,
                    })
                    dynamic_coding_params.append({
                        "label": "lossless",
                        "type": "bool",
                        "set": "quality",
                        "to": [0, -1],
                        "rebuild": True,
                    })
                    if params["coding_params"]["quality"] != -1:
                        dynamic_coding_params.append({
                            "label": "quality",
                            "type": "number",
                            "min": 0,
                            "max": 100,
                            "step": 1,
                            "dtype": int,
                        })
                    dynamic_params_config.append({
                        "label": "coding_params",
                        "type": "heading",
                        "params": dynamic_coding_params,
                    })

                case "png":
                    coding_params = params.get("coding_params", {})
                    if "compression_level" not in coding_params:
                        coding_params.clear()
                        coding_params["compression_level"] = 3
                    dynamic_coding_params.append({
                        "label": "compression",
                        "set": "compression_level",
                        "type": "number",
                        "min": 0,
                        "max": 9,
                        "step": 1,
                        "dtype": int,
                    })
                    dynamic_params_config.append({
                        "label": "coding_params",
                        "type": "heading",
                        "params": dynamic_coding_params,
                    })
                case "webp":
                    if not all(key in coding_params for key in ["quality", "method", "exact", "lossless"]):
                        coding_params.clear()
                        coding_params["quality"] = 100
                        coding_params["method"] = 6
                        coding_params["exact"] = True
                        coding_params["lossless"] = True
                    dynamic_coding_params.append({
                        "label": "exact",
                        "type": "bool",
                        "values": ["True", "False"],
                    })
                    dynamic_coding_params.append({
                        "label": "lossless",
                        "type": "bool",
                        "values": ["True", "False"],
                    })
                    # filesize-speed tradeoff when lossless
                    dynamic_coding_params.append({
                        "label": "quality",
                        "type": "number",
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "dtype": int,
                    })
                    # also filesize-speed tradeoff, all compatible with lossless
                    dynamic_coding_params.append({
                        "label": "method",
                        "type": "number",
                        "dtype": int,
                        "min": 0,
                        "max": 6,
                        "step": 1,
                    })
                    dynamic_params_config.append({
                        "label": "coding_params",
                        "type": "heading",
                        "params": dynamic_coding_params,
                    })

                case _:
                    raise ValueError(f"unknown image codec: {params["image_codec"]}")

        return dynamic_params_config


class ReadFile(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"file_path": file_path, "type": "ply", "field_prefix": field_prefix}:
                ply_fields = decode_ply(file_path=Path(file_path), field_prefix=field_prefix)
                new_fields.update(ply_fields)
            case {"file_path": file_path, "type": "image", "image_codec": codec, "field_name": field_name}:
                match codec:
                    case "png":
                        img_field_data = torch.tensor(
                            cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                        )
                    case "avif":
                        # TODO: only do this once?
                        register_avif_opener()
                        img_field_data = torch.tensor(np.array(Image.open(file_path)))
                    case "webp":
                        img_field_data = torch.tensor(np.array(Image.open(file_path)))

                new_fields[field_name] = Field.from_file(img_field_data, file_path, field_name)
            case _:
                raise ValueError(f"Unknown ReadFile parameters: {params}")

        return new_fields, decoding_update


class SimpleQuantize(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        torch_dtype_to_str = {
            torch.float32: "float32",
            torch.uint8: "uint8",
            torch.uint16: "uint16",
            torch.uint32: "uint32",
            torch.int32: "int32",
        }
        match params:
            case {"min": min_val, "max": max_val, "dim": dim, "dtype": dtype_str, "round_to_int": round_to_int}:
                min_val_f = float(min_val)
                max_val_f = float(max_val)

                min_vals = torch.amin(field_data, dim=[d for d in range(field_data.ndim) if d != dim])
                max_vals = torch.amax(field_data, dim=[d for d in range(field_data.ndim) if d != dim])

                field_range = max_vals - min_vals
                field_range[field_range == 0] = 1.0

                normalized = (field_data - min_vals) / field_range
                normalized = normalized * (max_val_f - min_val_f) + min_val_f

                field_data = normalized

                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [
                        {
                            "simple_quantize": {
                                "min_values": min_vals.tolist() if min_vals.ndim > 0 else [min_vals.item()],
                                "max_values": max_vals.tolist() if max_vals.ndim > 0 else [max_vals.item()],
                                "dim": dim,
                                "dtype": torch_dtype_to_str[field_data.dtype],
                                "round_to_int": False,  # in backmapping usually don't want rounding, right?
                            }
                        }
                    ],
                })

                if round_to_int:
                    field_data = torch.round(field_data)
                field_data = convert_to_dtype(field_data, dtype_str)

            case {
                "min_values": min_values,
                "max_values": max_values,
                "dim": dim,
                "dtype": dtype_str,
                "round_to_int": round_to_int,
            }:
                if field_data is None:
                    raise ValueError("Field data is None before channelwise remapping")

                min_vals = torch.amin(field_data, dim=[d for d in range(field_data.ndim) if d != dim])
                max_vals = torch.amax(field_data, dim=[d for d in range(field_data.ndim) if d != dim])

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

                decoding_update.append({
                    "input_fields": [field_name],
                    "transforms": [
                        {
                            "simple_quantize": {
                                "min_values": min_vals.tolist(),
                                "max_values": max_vals.tolist(),
                                "dim": dim,
                                "dtype": torch_dtype_to_str[field_data.dtype],
                                "round_to_int": False,  # in backmapping usually don't want rounding, right?
                            }
                        }
                    ],
                })

                if round_to_int:
                    field_data = torch.round(field_data)
                field_data = convert_to_dtype(field_data, dtype_str)

            case _:
                raise ValueError(f"Unknown simple_quantize parameters: {params}")
        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update

    @staticmethod
    def get_dynamic_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the dynamic parameters for a given transformation type. This might modify the values in params."""

        dynamic_params_config: list[dict[str, Any]] = []
        max_val = 0
        match params["dtype"]:
            case "uint8":
                max_val = 8
            case "uint16":
                max_val = 16
            case "uint32":
                max_val = 32
            case _:
                raise ValueError(f"Incompatible dtype {params["dtype"]}")

        dynamic_params_config.append({
            "label": "n bits",
            "type": "number",
            "min": 1,
            "max": max_val,
            "step": 1,
            "dtype": int,
            "set": "max",
            "mapping": lambda x: 2**x - 1,
            "inverse_mapping": lambda x: np.log2(x + 1),
        })
        return dynamic_params_config


transformation_map = {
    "cluster": Cluster,
    "split": Split,
    "flatten": Flatten,
    "reshape": Reshape,
    "remapping": Remapping,
    "reparametize": Reparametrize,
    "to_field": ToField,
    "permute": Permute,
    "to_dtype": ToDType,
    "split_bytes": SplitBytes,
    "reindex": Reindex,
    "sort": Sort,
    "lookup": Lookup,
    "combine": Combine,
    "write_file": WriteFile,
    "read_file": ReadFile,
    "simple_quantize": SimpleQuantize,
}


def apply_transform(
    parentOp: "Operation", verbose: bool, decoding_params_hashable: str
) -> tuple[dict[str, "Field"], list[dict[str, Any]]]:
    transformation = transformation_map.get(parentOp.transform_type)
    if transformation is None:
        raise ValueError(f"Unknown transformation: {parentOp.transform_type}")
    elif transformation is WriteFile:
        decoding_ops: list[dict[str, Any]] = yaml.load(decoding_params_hashable, Loader=yaml.SafeLoader)["ops"]
        return transformation.apply(
            parentOp.params[parentOp.transform_type], parentOp, verbose=verbose, decoding_ops=decoding_ops
        )
    else:
        return transformation.apply(parentOp.params[parentOp.transform_type], parentOp, verbose=verbose)


def get_dynamic_params(params: dict[str, dict[str, Any]], input_field: list | str) -> list[dict[str, Any]]:
    """Get the dynamic parameters for a given transformation type. This might modify the values in params."""
    transform_type = next(iter(params.keys()))
    transformation = transformation_map.get(transform_type)
    if transformation is None:
        raise ValueError(f"Unknown transformation: {transform_type}")
    elif (
        transformation is Sort
        and params[transform_type]["method"] == "plas"
        and "weights" not in params[transform_type]
    ):
        params[transform_type]["weights"] = {k: 1.0 for k in input_field}

    return transformation.get_dynamic_params(params[transform_type])
