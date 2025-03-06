from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray


@dataclass
class DataParser(ABC):
    """Abstract data parser class"""

    data_dir: str
    normalize_data: bool
    datatype: str
    image_names: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    camera_ids: list[int] = field(default_factory=list)
    Ks_dict: dict[int, Float[NDArray, "3 3"]] = field(default_factory=dict)
    params_dict: dict[int, Float[NDArray, " 4"] | Float[NDArray, " 0"]] = field(default_factory=dict)
    imsize_dict: dict[int, tuple[int, int]] = field(default_factory=dict)
    mask_dict: dict[int, Float[NDArray, "N M"] | None] = field(default_factory=dict)
    camtoworlds: Float[NDArray, "N 4 4"] = field(default_factory=lambda: np.empty((0, 4, 4)))
    scene_scale: float = 0.0
    train_indices: Float[NDArray, " N"] = field(default_factory=lambda: np.empty(0, dtype=int))
    test_indices: Float[NDArray, " N"] = field(default_factory=lambda: np.empty(0, dtype=int))
    transform: Float[NDArray, "4 4"] = field(default_factory=lambda: np.eye(4, dtype=np.float32))
