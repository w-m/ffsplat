import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
import yaml
from PIL import Image
from pillow_heif import register_avif_opener  # type: ignore[import-untyped]
from plas import sort_with_plas  # type: ignore[import-untyped]
from torch import Tensor
from torchpq.clustering import KMeans  # type: ignore[import-untyped]

from ..io.ply import encode_ply
from ..models.fields import Field
from ..models.gaussians import Gaussians
from ..models.operations import Operation

register_avif_opener()


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

    files: list[dict[str, str]] = field(default_factory=list)
    fields: defaultdict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    scene: dict[str, Any] = field(default_factory=dict)

    def reverse_fields(self) -> None:
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
class PLASConfig:
    """Configuration for PLAS sorting."""

    prune_by: str
    scaling_fn: str
    # activated: bool
    shuffle: bool
    improvement_break: float
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


def as_grid_img(tensor: Tensor) -> Tensor:
    num_primitives = tensor.shape[0]
    grid_sidelen = int(math.sqrt(num_primitives))
    if grid_sidelen * grid_sidelen != num_primitives:
        raise ValueError(
            f"Number of primitives {num_primitives} is not a perfect square. Cannot reshape to grid image."
        )
    tensor = tensor.reshape(grid_sidelen, grid_sidelen, *tensor.shape[1:])
    return tensor


def plas_preprocess(plas_cfg: PLASConfig, fields: dict[str, Field], verbose: bool) -> Tensor:
    primitive_filter = primitive_filter_pruning_to_square_shape(fields[plas_cfg.prune_by].data, verbose)

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

    grid_to_sort = as_grid_img(params_tensor).permute(2, 0, 1)
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


def write_image(output_file_path: Path, field_data: Tensor, file_type: str, coding_params: dict[str, Any]) -> None:
    match file_type:
        case "png":
            cv2.imwrite(str(output_file_path), field_data.cpu().numpy())
        case "avif":
            Image.fromarray(field_data.cpu().numpy()).save(
                output_file_path,
                format="AVIF",
                quality=coding_params.get("quality", -1),
                chroma=coding_params.get("chroma", 444),
                matrix_coefficients=coding_params.get("matrix_coefficients", 0),
            )
        case _:
            raise ValueError(f"Unsupported file type: {file_type}")


@lru_cache
def cached_process_field(op: Operation, verbose: bool) -> tuple[Field, dict[str, Field], defaultdict[str, list]]:  # noqa: C901
    print("processing with cached_process_field")
    field_op = op.params
    field_name = next(iter(op.input_fields.keys()))
    field_data = op.input_fields[field_name].data
    field_dict = {}
    decoding_update: defaultdict[str, list] = defaultdict(list)
    match field_op:
        case {
            "cluster": {
                "method": "kmeans",
                "num_clusters": num_clusters,
                "distance": distance,
                "to_fields_with_prefix": to_fields_with_prefix,
            }
        }:
            kmeans = KMeans(n_clusters=num_clusters, distance=distance, verbose=True)
            labels = kmeans.fit(field_data.permute(1, 0).contiguous())
            centroids = kmeans.centroids.permute(1, 0)
            field_dict[f"{to_fields_with_prefix}labels"] = Field(labels)
            field_dict[f"{to_fields_with_prefix}centroids"] = Field(centroids)
            decoding_update[field_name].append({
                "lookup": {
                    "from_field": f"{to_fields_with_prefix}labels",
                    "to_field": f"{to_fields_with_prefix}centroids",
                }
            })

        case {"flatten": {"start_dim": start_dim}}:
            target_shape = field_data.shape[start_dim:]
            decoding_update[field_name].append({"reshape_from_dim": {"start_dim": start_dim, "shape": target_shape}})
            field_data = field_data.flatten(start_dim=start_dim)

        case {
            "split": {
                "to_fields_with_prefix": to_fields_with_prefix,
                "split_size_or_sections": split_size_or_sections,
                "dim": dim,
            }
        }:
            chunks = field_data.split(split_size_or_sections, dim)
            for i, chunk in enumerate(chunks):
                field_dict[f"{to_fields_with_prefix}{i}"] = Field(chunk.squeeze(dim))

            decoding_update[field_name].append({
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
                "squeeze": squeeze,
            }
        }:
            chunks = field_data.split(split_size_or_sections, dim)
            for target_field_name, chunk in zip(to_field_list, chunks):
                if squeeze:
                    chunk = chunk.squeeze(dim)
                field_dict[target_field_name] = Field(chunk)

            decoding_update[field_name].append({
                "combine": {
                    "from_field_list": to_field_list,
                    "method": "stack" if split_size_or_sections == 1 else "concat",
                    "dim": dim,
                }
            })

        case {"reshape": {"shape": shape}}:
            decoding_update[field_name].append({"reshape": {"shape": field_data.shape}})
            field_data = field_data.reshape(*shape)

        case {"remapping": {"method": "log"}}:
            decoding_update[field_name].append({"remapping": {"method": "exp"}})
            field_data = field_data.log()

        case {"remapping": {"method": "signed-log"}}:
            decoding_update[field_name].append({"remapping": {"method": "signed-exp"}})
            field_data = torch.sign(field_data) * torch.log1p(torch.abs(field_data))

        case {"remapping": {"method": "inverse-sigmoid"}}:
            decoding_update[field_name].append({"remapping": {"method": "sigmoid"}})
            # ensure that we can encode opacity values in the full range [0, 1]
            # by clamping the values to (eps, 1 - eps) -> no infinite values from 0.0, 1.0
            eps = 1e-6
            field_data = field_data.clamp(eps, 1 - eps)
            field_data = torch.log(field_data / (1 - field_data))

        case {"remapping": {"method": "minmax", "min": min_val, "max": max_val}}:
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

        case {"remapping": {"method": "channelwise-minmax", "min": min_val, "max": max_val, "dim": dim}}:
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

        case {"to_field": name}:
            decoding_update[field_name].append({"from_field": name})
            field_dict[name] = Field(field_data)

        case {"to_tmp_field": name}:
            # a field that is used during the encoding process, but is not required for decoding
            # and is not stored in the output
            field_dict[name] = Field(field_data)

        case {"permute": {"dims": dims}}:
            decoding_update[field_name].append({"permute": {"dims": dims}})
            field_data = field_data.permute(*dims)

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

        case {"split_bytes": {"to_fields_with_prefix": to_fields_with_prefix, "num_bytes": num_bytes}}:
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
                field_dict[res_field_name] = Field(mask.to(torch.uint8))

            decoding_update[field_name].append({
                "combine": {
                    "from_field_list": res_field_names,
                    "method": "bytes",
                }
            })

        case _:
            raise ValueError(f"Unsupported field operation: {field_op}")

    new_field = Field(field_data)
    old_field = op.input_fields[field_name]
    new_field.ops = [*old_field.ops, op]
    return new_field, field_dict, decoding_update


@dataclass
class SceneEncoder:
    encoding_params: EncodingParams
    output_path: Path
    decoding_params: DecodingParams

    fields: dict[str, Field] = field(default_factory=dict)

    def _print_field_stats(self) -> None:
        # TODO duplicated code from scene_decoder
        print("Encoded field statistics:")
        for field_name, field_obj in sorted(self.fields.items()):
            stats = f"{field_name}: \t{tuple(field_obj.data.shape)} | {field_obj.data.dtype}"
            if field_obj.data.numel() > 0:
                stats += f" | Min: {field_obj.data.min().item():.4f} | Max: {field_obj.data.max().item():.4f}"
                stats += f" | Median: {field_obj.data.median().item():.4f}"
                # stats += f" | Unique Count: {field_obj.data.unique().numel()}"
            print(stats)

    def _encode_fields(self, verbose: bool) -> None:
        # go through the fields of encoding params
        for field_name, field_config in self.encoding_params.fields.items():
            field_obj = self.fields.get(field_name)

            for field_op in field_config:
                if isinstance(field_op, dict):
                    if field_obj is None:
                        raise ValueError(f"Field '{field_name}' not found in input fields.")
                    field_obj = self._process_field(Operation({field_name: field_obj}, field_op), verbose=verbose)
                    self.fields[field_name] = field_obj

    def _process_field(self, op: Operation, verbose: bool) -> Field:
        field_op = op.params
        field_name = next(iter(op.input_fields.keys()))
        field_data = op.input_fields[field_name].data
        match field_op:
            case {"reindex": {"index_field": index_field_name}}:
                index_field_obj = self.fields[index_field_name]
                if len(index_field_obj.data.shape) != 2:
                    raise ValueError("Expecting grid for re-index operation")
                self.decoding_params.fields[field_name].append({"flatten": {"start_dim": 0, "end_dim": 1}})
                field_data = field_data[index_field_obj.data]

            case {"plas": plas_cfg_dict} if isinstance(plas_cfg_dict, dict):
                sorted_indices = plas_preprocess(
                    plas_cfg=PLASConfig(**plas_cfg_dict),
                    fields=self.fields,
                    verbose=verbose,
                )
                field_data = sorted_indices

            case _:
                new_field, field_dict, decoding_update = cached_process_field(op, verbose=verbose)
                self.fields.update(field_dict)
                for key, op_list in decoding_update.items():
                    self.decoding_params.fields[key].extend(op_list)
                return new_field

        new_field = Field(field_data)
        old_field = op.input_fields[field_name]
        new_field.ops = [*old_field.ops, op]
        return new_field

    def _write_files(self) -> None:
        for file in self.encoding_params.files:
            match file:
                case {"from_fields_with_prefix": field_prefix, "type": "ply", "file_path": file_path}:
                    fields_to_write = {
                        field_name[len(field_prefix) :]: field_obj.data
                        for field_name, field_obj in self.fields.items()
                        if field_name.startswith(field_prefix)
                    }

                    self.decoding_params.files.append({
                        "file_path": file_path,
                        "type": "ply",
                        "field_prefix": field_prefix,
                    })

                    output_file_path = self.output_path / file_path

                    encode_ply(fields=fields_to_write, path=output_file_path)
                case {"from_field": field_name, "type": file_type, "coding_params": coding_params}:
                    field_data = self.fields[field_name].data
                    file_path = f"{field_name}.{file_type}"
                    output_file_path = self.output_path / file_path

                    self.decoding_params.files.append({
                        "file_path": file_path,
                        "type": file_type,
                        "field_name": field_name,
                    })

                    write_image(
                        output_file_path,
                        field_data,
                        file_type,
                        coding_params if isinstance(coding_params, dict) else {},
                    )

                case {
                    "from_fields_with_prefix": field_prefix,
                    "type": file_type,
                    "coding_params": coding_params,
                }:
                    for field_name, field_obj in self.fields.items():
                        field_data = field_obj.data
                        if field_name.startswith(field_prefix):
                            file_path = f"{field_name}.{file_type}"
                            output_file_path = self.output_path / file_path
                            write_image(
                                output_file_path,
                                field_data,
                                file_type,
                                coding_params if isinstance(coding_params, dict) else {},
                            )

                            self.decoding_params.files.append({
                                "file_path": file_path,
                                "type": file_type,
                                "field_name": field_name,
                            })

                case _:
                    raise ValueError(f"Unsupported file format: {file}")

    def encode(self, verbose: bool) -> None:
        # container as folder for now
        self.output_path.mkdir(parents=True, exist_ok=True)

        # empty base field for PLAS, which picks several input fields
        # TODO review design
        self.fields["_"] = Field(torch.empty(0))

        self._encode_fields(verbose=verbose)

        if verbose:
            self._print_field_stats()

        self._write_files()

        # revert the order of the fields in self.decoding_params to enable straightforward decoding
        self.decoding_params.reverse_fields()

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
