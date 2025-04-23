import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, override

import torch
from plas import sort_with_plas  # type: ignore[import-untyped]
from torch import Tensor
from torchpq.clustering import KMeans  # type: ignore[import-untyped]

from ..models.fields import Field

if TYPE_CHECKING:
    from ..models.fields import Operation


@dataclass
class PLASConfig:
    """Configuration for PLAS sorting."""

    prune_by: str
    scaling_fn: str
    # activated: bool
    shuffle: bool
    improvement_break: float
    # this is not needed for plas, just to now where to store the result
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
    def __init__(self, params: dict[str, Any], parentOp: Operation) -> None:
        self.params = params
        self.parentOp = parentOp

    @abstractmethod
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        pass


class Cluster(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        # Implement the clustering logic here
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)
        match self.params:
            case {
                "method": "kmeans",
                "num_clusters": num_clusters,
                "distance": distance,
                "to_fields_with_prefix": to_fields_with_prefix,
            }:
                kmeans = KMeans(n_clusters=num_clusters, distance=distance, verbose=True)
                labels = kmeans.fit(field_data.permute(1, 0).contiguous())
                centroids = kmeans.centroids.permute(1, 0)
                new_fields[f"{to_fields_with_prefix}labels"] = Field(labels, self.parentOp)
                new_fields[f"{to_fields_with_prefix}centroids"] = Field(centroids, self.parentOp)
                decoding_update[field_name].append({
                    "lookup": {
                        "from_field": f"{to_fields_with_prefix}labels",
                        "to_field": f"{to_fields_with_prefix}centroids",
                    }
                })
            case _:
                raise ValueError(f"Unknown clustering method: {self.params['method']}")
        return new_fields, decoding_update


class Split(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)
        match self.params:
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
                    new_fields[target_field_name] = Field(chunk, self.parentOp)

                decoding_update[field_name].append({
                    "combine": {
                        "from_field_list": to_field_list,
                        "method": "stack" if split_size_or_sections == 1 else "concat",
                        "dim": dim,
                    }
                })

            # TODO: should squeeze be optional?
            case {
                "split_size_or_sections": split_size_or_sections,
                "dim": dim,
                "to_fields_with_prefix": to_fields_with_prefix,
            }:
                chunks = field_data.split(split_size_or_sections, dim)
                for i, chunk in enumerate(chunks):
                    new_fields[f"{to_fields_with_prefix}{i}"] = Field(chunk.squeeze(dim), self.parentOp)

                decoding_update[field_name].append({
                    "combine": {
                        "from_fields_with_prefix": to_fields_with_prefix,
                        "method": "stack" if split_size_or_sections == 1 else "concat",
                        "dim": dim,
                    }
                })

            case _:
                raise ValueError(f"Unknown split parameters: {self.params}")
        return new_fields, decoding_update


class Remapping(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)
        match self.params:
            case {"method": "log"}:
                decoding_update[field_name].append({"remapping": {"method": "exp"}})
                field_data = field_data.log()

            case {"method": "signed-log"}:
                decoding_update[field_name].append({"remapping": {"method": "signed-exp"}})
                field_data = torch.sign(field_data) * torch.log1p(torch.abs(field_data))

            case {"method": "inverse-sigmoid"}:
                decoding_update[field_name].append({"remapping": {"method": "sigmoid"}})
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

                decoding_update[field_name].append({
                    "remapping": {
                        "method": "minmax",
                        "min": field_min,
                        "max": field_max,
                    }
                })

                field_data = minmax(field_data)
                field_data = field_data * (max_val_f - min_val_f) + min_val_f

            case {"method": "channelwise-minmax", "min": min_val, "max": max_val, "dim": dim}:
                min_val_f = float(min_val)
                max_val_f = float(max_val)

                min_vals = torch.amin(field_data, dim=[d for d in range(field_data.ndim) if d != dim])
                max_vals = torch.amax(field_data, dim=[d for d in range(field_data.ndim) if d != dim])

                decoding_update[field_name].append({
                    "remapping": {
                        "method": "channelwise-minmax",
                        "min_values": min_vals.tolist(),
                        "max_values": max_vals.tolist(),
                        "dim": dim,
                    }
                })

                field_range = max_vals - min_vals
                field_range[field_range == 0] = 1.0

                normalized = (field_data - min_vals) / field_range
                normalized = normalized * (max_val_f - min_val_f) + min_val_f

                field_data = normalized
            case _:
                raise ValueError(f"Unknown remapping parameters: {self.params}")
        new_fields[field_name] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class ToField(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        to_field = self.params.get("to_field")
        if to_field is None:
            raise ValueError(f"Unknown ToField parameters: {self.params}")
        decoding_update[field_name].append({"from_field": to_field})
        new_fields[to_field] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class Flatten(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        start_dim = self.params.get("start_dim")
        if start_dim is None:
            raise ValueError(f"Unknown Flatten parameters: {self.params}")
        target_shape = field_data.shape[start_dim:]
        decoding_update[field_name].append({"reshape_from_dim": {"start_dim": start_dim, "shape": target_shape}})
        field_data = field_data.flatten(start_dim=start_dim)

        new_fields[field_name] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class Reshape(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)
        shape = self.params.get("shape")
        if shape is None:
            raise ValueError(f"Unknown Reshape parameters: {self.params}")
        decoding_update[field_name].append({"reshape": {"shape": field_data.shape}})
        field_data = field_data.reshape(*shape)

        new_fields[field_name] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class Permute(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        dims = self.params.get("dims")
        if dims is None:
            raise ValueError(f"Unknown Permute parameters: {self.params}")
        decoding_update[field_name].append({"permute": {"dims": dims}})
        field_data = field_data.permute(*dims)

        new_fields[field_name] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class ToDType(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        match self.params:
            case {"to_dtype": {"dtype": dtype_str, "round_to_int": round_to_int}}:
                torch_dtype_to_str = {
                    torch.float32: "float32",
                }

                if round_to_int:
                    field_data = torch.round(field_data)

                decoding_update[field_name].append({"to_dtype": {"dtype": torch_dtype_to_str[field_data.dtype]}})
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
                    case _:
                        raise ValueError(f"Unsupported dtype for conversion: {dtype_str}")
            case _:
                raise ValueError(f"Unknown ToDType parameters: {self.params}")

        new_fields[field_name] = Field(field_data, self.parentOp)
        return new_fields, decoding_update


class SplitBytes(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        field_name = next(iter(input_fields.keys()))
        field_data = input_fields[field_name].data

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        match self.params:
            case {"split_bytes": {"num_bytes": num_bytes, "to_fields_with_prefix": to_fields_with_prefix}}:
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
                    new_fields[res_field_name] = Field(mask.to(torch.uint8), self.parentOp)

                # TODO: why from_field_list and not from_fields_with_prefix?
                decoding_update[field_name].append({
                    "combine": {
                        "from_field_list": res_field_names,
                        "method": "bytes",
                    }
                })
            case _:
                raise ValueError(f"Unknown SplitBytes parameters: {self.params}")

        return new_fields, decoding_update


class Reindex(Transformation):
    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        input_fields = self.parentOp.input_fields

        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        match self.params:
            case {"reindex": {"src_field": src_field_name, "index_field": index_field_name}}:
                index_field_obj = input_fields[index_field_name]
                if len(index_field_obj.data.shape) != 2:
                    raise ValueError("Expecting grid for re-index operation")
                decoding_update[src_field_name].append({"flatten": {"start_dim": 0, "end_dim": 1}})
                original_data = input_fields[src_field_name].data
                new_fields[src_field_name] = Field(original_data[index_field_obj.data], self.parentOp)
            case _:
                raise ValueError(f"Unknown Reindex parameters: {self.params}")

        return super().apply()


class PLAS(Transformation):
    def __init__(self, params: dict[str, Any], parentOp: Operation, verbose: bool) -> None:
        super().__init__(params, parentOp)
        self.verbose = verbose

    def as_grid_img(self, tensor: Tensor) -> Tensor:
        num_primitives = tensor.shape[0]
        grid_sidelen = int(math.sqrt(num_primitives))
        if grid_sidelen * grid_sidelen != num_primitives:
            raise ValueError(
                f"Number of primitives {num_primitives} is not a perfect square. Cannot reshape to grid image."
            )
        tensor = tensor.reshape(grid_sidelen, grid_sidelen, *tensor.shape[1:])
        return tensor

    def primitive_filter_pruning_to_square_shape(self, data: Tensor) -> Tensor | None:
        """Returning None indicates that no primitives need to be pruned"""
        num_primitives = data.shape[0]

        grid_sidelen = int(math.sqrt(num_primitives))
        num_to_remove = num_primitives - grid_sidelen * grid_sidelen

        if num_to_remove == 0:
            return None

        if self.verbose:
            print(
                f"Removing {num_to_remove}/{num_primitives} primitives to fit the grid. ({100 * num_to_remove / num_primitives:.4f}%)"
            )

        _, keep_indices = torch.topk(data, k=grid_sidelen * grid_sidelen)
        sorted_keep_indices = torch.sort(keep_indices)[0]
        return sorted_keep_indices

    def plas_preprocess(self, plas_cfg: PLASConfig) -> Tensor:
        fields = self.parentOp.input_fields
        primitive_filter = self.primitive_filter_pruning_to_square_shape(fields[plas_cfg.prune_by].data)

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

        # attr_getter_fn = self.get_activated_attr_flat if sorting_cfg.activated else self.get_attr_flat

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

        grid_to_sort = self.as_grid_img(params_tensor).permute(2, 0, 1)
        _, sorted_indices_ret = sort_with_plas(
            grid_to_sort, improvement_break=float(plas_cfg.improvement_break), verbose=self.verbose
        )

        sorted_indices: Tensor = sorted_indices_ret.squeeze(0).to(params_tensor.device)

        if plas_cfg.shuffle:
            flat_indices = sorted_indices.flatten()
            unshuffled_flat_indices = shuffled_indices[flat_indices]
            sorted_indices = unshuffled_flat_indices.reshape(sorted_indices.shape)

        if primitive_filter is not None:
            return primitive_filter[sorted_indices]

        return sorted_indices

    @override
    def apply(self) -> tuple[dict[str, Field], defaultdict[str, list]]:
        new_fields: dict[str, Field] = {}
        decoding_update = defaultdict(list)

        match self.params:
            case {"plas": plas_cfg_dict} if isinstance(plas_cfg_dict, dict):
                sorted_indices = self.plas_preprocess(
                    plas_cfg=PLASConfig(**plas_cfg_dict),
                )
                new_fields[plas_cfg_dict["to_field"]] = Field(sorted_indices, self.parentOp)
            case _:
                raise ValueError(f"Unknown PLAS parameters: {self.params}")

        return new_fields, decoding_update
