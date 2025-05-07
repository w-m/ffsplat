from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from ..models.fields import Field


def decode_ply(file_path: Path, field_prefix: str) -> dict[str, Field]:
    vertices = PlyData.read(file_path)["vertex"]
    data = {}
    for prop in vertices.properties:
        data[f"{field_prefix}{prop.name}"] = Field.from_file(
            torch.from_numpy(vertices[prop.name]), file_path, f"{field_prefix}{prop.name}"
        )

    return data


def encode_ply(fields: dict[str, Field], path: Path) -> None:
    dtype_list = [(field_name, "f4") for field_name in fields]

    num_primitives = len(next(iter(fields.values())).data)
    vertex_data = np.empty(num_primitives, dtype=dtype_list)

    for field_name, field in fields.items():
        vertex_data[field_name] = field.data.cpu().numpy()

    vertex_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex_element], text=False).write(str(path))
