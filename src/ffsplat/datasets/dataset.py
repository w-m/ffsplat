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
        if parser.datatype == "blender":
            if split == "train":
                self.indices = self.parser.train_indices
            else:
                self.indices = self.parser.test_indices
            self.white_background = True
        elif parser.datatype == "colmap":
            indices = np.arange(len(self.parser.image_names))
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
            self.white_background = False

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

        if image.shape[2] == 4:
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

        return data
