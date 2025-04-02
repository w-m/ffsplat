import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
import yaml
from plas import sort_with_plas  # type: ignore[import-untyped]
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


def indices_of_pruning_to_square_shape(data: Tensor, verbose: bool = False) -> Tensor | slice:
    num_primitives = data.shape[0]

    grid_sidelen = int(math.sqrt(num_primitives))
    num_to_remove = num_primitives - grid_sidelen * grid_sidelen

    if num_to_remove == 0:
        return slice(None)

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


def plas_preprocess(plas_cfg: PLASConfig, fields: dict[str, Tensor]) -> Tensor:
    pruned_indices = indices_of_pruning_to_square_shape(fields[plas_cfg.prune_by])

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

    attr_getter_fn = lambda attr_name: fields[attr_name][pruned_indices]

    params_to_sort: list[Tensor] = []

    for attr_name, attr_weight in plas_cfg.weights.items():
        if attr_weight > 0:
            params_to_sort.append(normalization_fn(attr_getter_fn(attr_name)).flatten(start_dim=1) * attr_weight)

    params_tensor = torch.cat(params_to_sort, dim=1)

    if plas_cfg.shuffle:
        # TODO shuffling should be an option of sort_with_plas
        torch.manual_seed(42)
        shuffled_indices = torch.randperm(params_tensor.shape[0], device=params_tensor.device)
        params_tensor = params_tensor[shuffled_indices]

    grid_to_sort = as_grid_img(params_tensor).permute(2, 0, 1).to("cuda")
    _, sorted_indices_ret = sort_with_plas(grid_to_sort, improvement_break=plas_cfg.improvement_break, verbose=True)

    sorted_indices: Tensor = sorted_indices_ret.squeeze(0).to(params_tensor.device)

    if plas_cfg.shuffle:
        flat_indices = sorted_indices.flatten()
        unshuffled_flat_indices = shuffled_indices[flat_indices]
        sorted_indices = unshuffled_flat_indices.reshape(sorted_indices.shape)

    return sorted_indices


@dataclass
class SceneEncoder:
    encoding_params: EncodingParams
    output_path: Path
    decoding_params: DecodingParams

    fields: dict[str, Tensor] = field(default_factory=dict)

    def _encode_fields(self) -> None:
        # go through the fields of encoding params
        for field_name, field_config in self.encoding_params.fields.items():
            field_data = self.fields.get(field_name)

            for field_op in field_config:
                if isinstance(field_op, dict):
                    if field_data is None:
                        raise ValueError(f"Field '{field_name}' not found in input fields.")
                    field_data = self._process_field(field_name, field_op, field_data)
                    self.fields[field_name] = field_data

    def _process_field(self, field_name: str, field_op: dict[str, Any], field_data: Tensor) -> Tensor:  # noqa: C901
        match field_op:
            case {
                "split": {
                    "to_fields_with_prefix": to_fields_with_prefix,
                    "split_size_or_sections": split_size_or_sections,
                    "dim": dim,
                }
            }:
                chunks = field_data.split(split_size_or_sections, dim)
                for i, chunk in enumerate(chunks):
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
                    "squeeze": squeeze,
                }
            }:
                chunks = field_data.split(split_size_or_sections, dim)
                for target_field_name, chunk in zip(to_field_list, chunks):
                    if squeeze:
                        chunk = chunk.squeeze(dim)
                    self.fields[target_field_name] = chunk

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

            case {"remapping": {"method": "signed-log"}}:
                self.decoding_params.fields[field_name].append({"remapping": {"method": "signed-exp"}})
                field_data = torch.sign(field_data) * torch.log1p(torch.abs(field_data))

            case {"remapping": {"method": "inverse-sigmoid"}}:
                self.decoding_params.fields[field_name].append({"remapping": {"method": "sigmoid"}})
                field_data = torch.log(field_data / (1 - field_data))

            case {"remapping": {"method": "minmax", "min": min_val, "max": max_val}}:
                field_min = field_data.min().item()
                field_max = field_data.max().item()

                min_val_f = float(min_val)
                max_val_f = float(max_val)

                self.decoding_params.fields[field_name].append({
                    "remapping": {
                        "method": "minmax",
                        "min": field_min,
                        "max": field_max,
                    }
                })

                field_data = minmax(field_data)
                field_data = field_data * (max_val_f - min_val_f) + min_val_f

            case {"reindex": {"index_field": index_field_name}}:
                index_field = self.fields[index_field_name]
                if len(index_field.shape) != 2:
                    raise ValueError("Expecting grid for re-index operation")
                self.decoding_params.fields[field_name].append({"flatten": {"start_dim": 0, "end_dim": 1}})
                field_data = field_data[index_field]

            case {"to_field": name}:
                self.decoding_params.fields[field_name].append({"from_field": name})
                self.fields[name] = field_data

            case {"to_tmp_field": name}:
                # a field that is used during the encoding process, but is not required for decoding
                # and is not stored in the output
                self.fields[name] = field_data

            case {"permute": {"dims": dims}}:
                self.decoding_params.fields[field_name].append({"permute": {"dims": dims}})
                field_data = field_data.permute(*dims)

            case {"plas": plas_cfg_dict} if isinstance(plas_cfg_dict, dict):
                sorted_indices = plas_preprocess(
                    plas_cfg=PLASConfig(**plas_cfg_dict),
                    fields=self.fields,
                )
                field_data = sorted_indices

            case {"to_dtype": {"dtype": dtype_str}}:
                torch_dtype_to_str = {
                    torch.float32: "float32",
                }

                self.decoding_params.fields[field_name].append({
                    "to_dtype": {"dtype": torch_dtype_to_str[field_data.dtype]}
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
                    case _:
                        raise ValueError(f"Unsupported dtype for conversion: {dtype_str}")

            case _:
                raise ValueError(f"Unsupported field operation: {field_op}")

        return field_data

    def _write_files(self) -> None:
        for file in self.encoding_params.files:
            match file:
                case {"from_fields_with_prefix": field_prefix, "type": "ply", "file_path": file_path}:
                    fields_to_write = {
                        field_name[len(field_prefix) :]: field_data
                        for field_name, field_data in self.fields.items()
                        if field_name.startswith(field_prefix)
                    }

                    self.decoding_params.files.append({
                        "file_path": file_path,
                        "type": "ply",
                        "field_prefix": field_prefix,
                    })

                    output_file_path = self.output_path / file_path

                    encode_ply(fields=fields_to_write, path=output_file_path)
                case {"from_field": field_name, "type": file_type}:
                    field_data = self.fields[field_name]
                    file_path = f"{field_name}.png"
                    output_file_path = self.output_path / file_path

                    self.decoding_params.files.append({
                        "file_path": file_path,
                        "type": file_type,
                        "field_name": field_name,
                    })

                    match file_type:
                        case "png":
                            cv2.imwrite(str(output_file_path), field_data.cpu().numpy())
                        case _:
                            raise ValueError(f"Unsupported file type: {file_type}")

                case {
                    "from_fields_with_prefix": field_prefix,
                    "type": "png",
                }:
                    for field_name, field_data in self.fields.items():
                        if field_name.startswith(field_prefix):
                            file_path = f"{field_name}.png"
                            output_file_path = self.output_path / file_path
                            cv2.imwrite(str(output_file_path), field_data.cpu().numpy())

                            self.decoding_params.files.append({
                                "file_path": file_path,
                                "type": "png",
                                "field_name": field_name,
                            })

                case _:
                    raise ValueError(f"Unsupported file format: {file}")

    def encode(self) -> None:
        # container as folder for now
        self.output_path.mkdir(parents=True, exist_ok=True)

        # empty base field for PLAS, which picks several input fields
        # TODO review design
        self.fields["_"] = torch.empty(0)

        self._encode_fields()
        self._write_files()

        # revert the order of the fields in self.decoding_params to enable straightforward decoding
        self.decoding_params.reverse_fields()

        # Write the YAML directly using our custom dumper
        with open(self.output_path / "container_meta.yaml", "w") as f:
            yaml.dump(self.decoding_params, f, Dumper=SerializableDumper, default_flow_style=False, sort_keys=False)


def encode_gaussians(gaussians: Gaussians, output_path: Path, output_format: str) -> None:
    match output_format:
        case "3DGS-INRIA.ply":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/3DGS_INRIA_ply.yaml"))
        case "SOG-web":
            encoding_params = EncodingParams.from_yaml_file(Path("src/ffsplat/conf/format/SOG-web.yaml"))
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    encoder = SceneEncoder(
        encoding_params=encoding_params,
        output_path=output_path,
        fields=gaussians.to_dict(),
        decoding_params=DecodingParams(
            container_identifier="smurfx",
            container_version="0.1",
            packer="ffsplat-v0.1",
            profile=encoding_params.profile,
            profile_version=encoding_params.profile_version,
            scene=encoding_params.scene,
        ),
    )
    encoder.encode()
