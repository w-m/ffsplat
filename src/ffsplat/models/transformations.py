import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, override

import cv2
import numpy as np
import torch
from PIL import Image
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]
from plas import sort_with_plas  # type: ignore[import-untyped]
from torch import Tensor
from torchpq.clustering import KMeans  # type: ignore[import-untyped]

from ..io.ply import decode_ply, encode_ply
from ..models.fields import Field

if TYPE_CHECKING:
    from ..models.fields import Operation


def write_image(output_file_path: Path, field_data: Tensor, file_type: str, coding_params: dict[str, Any]) -> None:
    match file_type:
        case "png":
            cv2.imwrite(str(output_file_path), field_data.cpu().numpy())
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
        case _:
            raise ValueError(f"Unsupported file type: {file_type}")


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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        """Apply the transformation to the input fields. returns new/updated fields and decoding updates.
        Transformations that are only available for decoding do return empty decoding updates"""
        pass


class Cluster(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
                raise ValueError(f"Unknown clustering method: {params['method']}")
        return new_fields, decoding_update


class Split(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
                for target_field_name, chunk in zip(to_field_list, chunks):
                    if squeeze:
                        chunk = chunk.squeeze(dim)
                    new_fields[target_field_name] = Field(chunk, parentOp)

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
    def apply(  # noqa: C901
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
            case {
                "method": "channelwise-minmax",
                "min_values": min_values,
                "max_values": max_values,
                "dim": dim,
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
            case _:
                raise ValueError(f"Unknown remapping parameters: {params}")
        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class ToField(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"start_dim": start_dim, "shape": shape}:
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        }

        if round_to_int:
            field_data = torch.round(field_data)

        decoding_update.append({
            "input_fields": [field_name],
            "transforms": [{"to_dtype": {"dtype": torch_dtype_to_str[field_data.dtype]}}],
        })
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
            case "float32":
                field_data = field_data.to(torch.float32)
            case _:
                raise ValueError(f"Unsupported dtype for conversion: {dtype_str}")

        new_fields[field_name] = Field(field_data, parentOp)
        return new_fields, decoding_update


class SplitBytes(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        input_fields = parentOp.input_fields

        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []

        match params:
            case {"src_field": src_field_name, "index_field": index_field_name}:
                index_field_obj = input_fields[index_field_name]
                if len(index_field_obj.data.shape) != 2:
                    raise ValueError("Expecting grid for re-index operation")
                decoding_update.append({
                    "input_fields": [src_field_name],
                    "transforms": [{"flatten": {"start_dim": 0, "end_dim": 1}}],
                })
                original_data = input_fields[src_field_name].data
                new_fields[src_field_name] = Field(original_data[index_field_obj.data], parentOp)
            case _:
                raise ValueError(f"Unknown Reindex parameters: {params}")

        return new_fields, decoding_update


class PLAS(Transformation):
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

        _, keep_indices = torch.topk(data, k=grid_sidelen * grid_sidelen)
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
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        new_fields: dict[str, Field] = {}

        plas_cfg_dict = params
        if isinstance(plas_cfg_dict, dict):
            sorted_indices = PLAS.plas_preprocess(
                plas_cfg=PLASConfig(**plas_cfg_dict),
                fields=parentOp.input_fields,
                verbose=verbose,
            )
            new_fields[plas_cfg_dict["to_field"]] = Field(sorted_indices, parentOp)
        else:
            raise TypeError(f"Unknown PLAS parameters: {params}")

        return new_fields, []


class Combine(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        new_fields: dict[str, Field] = {}
        field_data: Tensor = torch.empty(0)

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
                if method == "stack":
                    field_data = torch.stack(tensors, dim=dim)
                elif method == "concat":
                    field_data = torch.cat(tensors, dim=dim)
                else:
                    raise ValueError(f"Unsupported combine method: {method}")
                new_fields[to_field_name] = Field(field_data, parentOp)
            case _:
                raise ValueError(f"Unknown Combine parameters: {params}")

        return new_fields, []


class Lookup(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
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
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
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
            case {"type": file_type, "coding_params": coding_params, "base_path": base_path}:
                for field_name, field_obj in parentOp.input_fields.items():
                    field_data = field_obj.data
                    file_path = f"{field_name}.{file_type}"
                    output_file_path = Path(base_path) / Path(file_path)
                    write_image(
                        output_file_path,
                        field_data,
                        file_type,
                        coding_params if isinstance(coding_params, dict) else {},
                    )

                    decoding_update.append({
                        "input_fields": [],
                        "transforms": [
                            {"read_file": {"file_path": file_path, "type": file_type, "field_name": field_name}}
                        ],
                    })
            case _:
                raise ValueError(f"Unknown WriteFile parameters: {params}")

        return {}, decoding_update


class ReadFile(Transformation):
    @staticmethod
    @override
    def apply(
        params: dict[str, Any], parentOp: "Operation", verbose: bool = False
    ) -> tuple[dict[str, Field], list[dict[str, Any]]]:
        new_fields: dict[str, Field] = {}
        decoding_update: list[dict[str, Any]] = []
        match params:
            case {"file_path": file_path, "type": "ply", "field_prefix": field_prefix}:
                ply_fields = decode_ply(file_path=Path(file_path), field_prefix=field_prefix)
                new_fields.update(ply_fields)
            case {"file_path": file_path, "type": file_type, "field_name": field_name}:
                match file_type:
                    case "png":
                        img_field_data = torch.tensor(
                            cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                        )
                    case "avif":
                        # TODO: only do this once?
                        register_avif_opener()
                        img_field_data = torch.tensor(np.array(Image.open(file_path)))

                new_fields[field_name] = Field.from_file(img_field_data, file_path, field_name)
            case _:
                raise ValueError(f"Unknown ReadFile parameters: {params}")

        return new_fields, decoding_update


transformation_map = {
    "cluster": Cluster,
    "split": Split,
    "flatten": Flatten,
    "reshape": Reshape,
    "remapping": Remapping,
    "to_field": ToField,
    "permute": Permute,
    "to_dtype": ToDType,
    "split_bytes": SplitBytes,
    "reindex": Reindex,
    "plas": PLAS,
    "lookup": Lookup,
    "combine": Combine,
    "write_file": WriteFile,
    "read_file": ReadFile,
}


def apply_transform(parentOp: "Operation", verbose: bool) -> tuple[dict[str, "Field"], list[dict[str, Any]]]:
    transformation = transformation_map.get(parentOp.transform_type)
    if transformation is None:
        raise ValueError(f"Unknown transformation: {parentOp.transform_type}")
    return transformation.apply(parentOp.params[parentOp.transform_type], parentOp, verbose)
