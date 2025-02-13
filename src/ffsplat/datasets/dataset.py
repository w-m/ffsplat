# This class builds on https://github.com/nerfstudio-project/gsplat/blob/0880d2b471e6650d458aa09fe2b2834531f6e93b/examples/datasets/colmap.py


from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .dataparser import DataParser


class Dataset:
    """A simple dataset class."""

    def __init__(self, parser: DataParser, split: str = "eval", load_depths: bool = False):
        self.parser = parser
        self.split = split
        self.load_depths = load_depths
        if parser.type == "blender":
            if split == "train":
                self.indices = self.parser.train_indices
            else:
                self.indices = self.parser.test_indices
        elif parser.type == "colmap":
            indices = np.arange(len(self.parser.image_names))
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Mapping[str, Any]:
        index = self.indices[item]
        camera_id = self.parser.camera_ids[index]
        image = Image.open(self.parser.image_paths[index])
        camtoworlds = self.parser.camtoworlds[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        mask = self.parser.mask_dict[camera_id]

        image = np.array(image) / 255.0
        bg = np.array([1.0, 1.0, 1.0]) if self.white_background else np.array([0.0, 0.0, 0.0])

        image = image[:, :, :3] * (image[:, :, 3:4]) + bg * (1 - (image[:, :, 3:4]))
        image = image[..., :3]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
