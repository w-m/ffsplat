from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import Tensor


def decode_ply(file_path: Path, field_prefix: str) -> dict[str, torch.Tensor]:
    vertices = PlyData.read(file_path)["vertex"]
    data = {}
    for prop in vertices.properties:
        data[f"{field_prefix}{prop.name}"] = torch.from_numpy(vertices[prop.name])

    return data


def encode_ply(fields: dict[str, Tensor], path: Path) -> None:
    dtype_list = [(field_name, "f4") for field_name in fields]

    num_primitives = len(next(iter(fields.values())))
    vertex_data = np.empty(num_primitives, dtype=dtype_list)

    for field_name, field_data in fields.items():
        vertex_data[field_name] = field_data.cpu().numpy()

    vertex_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex_element], text=False).write(str(path))
