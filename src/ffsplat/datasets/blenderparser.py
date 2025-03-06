import os
from pathlib import Path

import camorph.camorph as camorph
import camorph.lib.utils.math_utils as math_utils
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image

from .dataparser import DataParser
from .normalize import normalize


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2 * np.tan(fov / 2))


class BlenderParser(DataParser):
    """synthetic parser."""

    def __init__(
        self,
        data_dir: str,
        normalize_data: bool = False,
    ) -> None:
        self.data_dir: str = data_dir
        self.normalize_data: bool = normalize_data
        self.datatype: str = "blender"

        # Load camera-to-world matrices.
        self.image_names: list[str] = []  # (num_images,)
        self.image_paths: list[str] = []  # (num_images,)
        self.camtoworlds: Float[NDArray, "N 4 4"] = np.empty((0, 4, 4))  # (num_images, 4, 4)
        print(self.camtoworlds)
        self.camera_ids: list[int] = []  # (num_images,)
        self.Ks_dict: dict[int, Float[NDArray, "3 3"]] = {}  # camera_id -> K
        self.imsize_dict: dict[int, tuple[int, int]] = {}  # camera_id -> (width, height)
        self.params_dict: dict[int, Float[NDArray, " 4"] | Float[NDArray, " 0"]] = {}
        self.mask_dict: dict[int, None] = {}

        self.load_synthetic(data_dir, "transforms_train.json", 0)

        self.train_indices: Float[NDArray, " N"] = np.arange(len(self.image_names))
        train_camera_id_len: int = len(self.camera_ids)

        # load test data
        self.load_synthetic(data_dir, "transforms_test.json", train_camera_id_len)

        self.test_indices: Float[NDArray, " N"] = np.arange(len(self.train_indices), len(self.image_names))

        self.camtoworlds = np.stack(self.camtoworlds, axis=0)

        print(f"[Parser] {len(self.image_names)} images, taken by {len(set(self.camera_ids))} cameras.")

        if len(self.image_names) == 0:
            raise ValueError("No images found.")

        # Normalize the world space.
        if normalize_data:
            self.camtoworlds, transform = normalize(self.camtoworlds)
        else:
            transform = np.eye(4)

        self.transform: Float[NDArray, "4 4"] = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations: Float[NDArray, "N 3"] = self.camtoworlds[:, :3, 3]
        scene_center: Float[NDArray, " 3"] = np.mean(camera_locations, axis=0)
        dists: Float[NDArray, " N"] = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale: float = np.max(dists)

    def load_synthetic(self, data_dir: str, file: str, id_offset: int) -> None:
        cams = camorph.read_cameras("nerf", os.path.join(data_dir, file))
        bottom: Float[NDArray, "1 4"] = np.array([0, 0, 0, 1]).reshape(1, 4)
        camtoworlds_list: list[Float[NDArray, "4 4"]] = []

        for idx, cam in enumerate(cams):
            image_path: str = cam.source_image
            camera_id: int = idx + id_offset

            # although the coordinate system in cams is the same here as in the colmapparser
            # we can not convert it the same way. Instead we load the transform matrix and mirror y and z axes
            # to be compatible with other 3dgs implementations
            trans, rot = math_utils.convert_coordinate_systems(
                ["y", "-x", "z"], cam.t, cam.r, tdir=[0, 0, -1], tup=[0, 1, 0], transpose=True
            )
            trans = trans.reshape(3, 1)
            rot = rot.rotation_matrix

            c2w: Float[NDArray, " 4 4"] = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            c2w[:3, 1:3] *= -1

            image_name: str = Path(image_path).stem
            image = Image.open(image_path)

            fx, fy = cam.focal_length_px
            cx, cy = cam.principal_point
            K: Float[NDArray, "3 3"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            self.Ks_dict[camera_id] = K

            self.camera_ids.append(camera_id)
            camtoworlds_list.append(c2w)
            self.imsize_dict[camera_id] = (
                image.width,
                image.height,
            )
            # assume no distortion
            self.params_dict[camera_id] = np.empty(0, dtype=np.float32)
            self.mask_dict[camera_id] = None
            self.image_names.append(image_name)
            self.image_paths.append(image_path)
        self.camtoworlds = np.concatenate([self.camtoworlds, np.stack(camtoworlds_list, axis=0)], axis=0)
        return
