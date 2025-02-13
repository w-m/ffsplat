import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image

from .dataparser import DataParser
from .normalize import normalize


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


class BlenderParser(DataParser):
    """synthetic parser."""

    def __init__(
        self,
        data_dir: str,
        normalize_data: bool = False,
    ):
        self.data_dir = data_dir
        self.normalize_data = normalize_data

        # Load camera-to-world matrices.
        self.image_names: list[str] = []  # (num_images,)
        self.image_paths: list[str] = []  # (num_images,)
        self.camtoworlds = []
        self.camera_ids: list[int] = []  # (num_images,)
        self.Ks_dict: Mapping[int, Float[NDArray, "3 3"]] = {}  # camera_id -> K
        self.imsize_dict: Mapping[int, tuple[int, int]] = {}  # camera_id -> (width, height)
        self.params_dict = {}
        self.mask_dict = {}

        self.load_synthetic(data_dir, "transforms_train.json", 0)

        self.train_indices = np.arange(len(self.image_names))
        train_camera_id_len = len(self.camera_ids)

        # load test data
        self.load_synthetic(data_dir, "transforms_train.json", train_camera_id_len)

        self.test_indices = np.arange(len(self.train_indices), len(self.image_names))

        self.camtoworlds = np.array(self.camtoworlds)

        print(f"[Parser] {len(self.image_names)} images, taken by {len(set(self.camera_ids))} cameras.")

        if len(self.image_names) == 0:
            raise ValueError("No images found.")

        # Normalize the world space.
        if normalize_data:
            self.camtoworlds, transform = normalize(self.camtoworlds)
        else:
            transform = np.eye(4)

        self.transform = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def load_synthetic(self, data_dir: str, file: str, id_offset: int):
        with open(os.path.join(data_dir, file)) as json_file:
            contents = json.load(json_file)
            FOVX = contents["camera_angle_x"]

            for idx, frame in enumerate(contents["frames"]):
                image_path = os.path.join(data_dir, frame["file_path"] + ".png")
                camera_id = idx + id_offset

                c2w = np.array(frame["transform_matrix"])
                c2w[:3, 1:3] *= -1

                image_name = Path(image_path).stem
                image = Image.open(image_path)

                fx = fov2focal(FOVX, image.width)
                fy = fov2focal(FOVX, image.height)
                cx, cy = 0.5 * image.width, 0.5 * image.height
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                self.Ks_dict[camera_id] = K

                self.camera_ids.append(camera_id)
                self.camtoworlds.append(c2w)
                self.imsize_dict[camera_id] = (
                    image.width,
                    image.height,
                )
                # assume no distortion
                self.params_dict[camera_id] = np.empty(0, dtype=np.float32)
                self.mask_dict[camera_id] = None
                self.image_names.append(image_name)
                self.image_paths.append(image_path)
        return
