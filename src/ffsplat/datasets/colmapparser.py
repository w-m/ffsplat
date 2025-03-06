import os
from pathlib import Path
from typing import Any

import camorph.camorph as camorph
import camorph.lib.utils.math_utils as math_utils
import cv2
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from .dataparser import DataParser
from .normalize import normalize


def _get_rel_paths(path_dir: str) -> list[str]:
    """Recursively get relative paths of files in a directory."""
    paths: list[str] = []
    for dp, _, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: float) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(resized_dir, os.path.splitext(image_file)[0] + ".png")
        if os.path.isfile(resized_path):
            continue
        img = Image.open(image_path)
        resized_size = (
            int(round(img.size[0] / factor)),
            int(round(img.size[1] / factor)),
        )
        resized_image = img.resize(resized_size, Image.BICUBIC)
        resized_image.save(resized_path)
    return resized_dir


class ColmapParser(DataParser):
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = -1,
        normalize_data: bool = False,
        test_every: int = 8,
    ) -> None:
        self.data_dir: str = data_dir
        self.normalize_data: bool = normalize_data
        self.test_every: int = test_every
        self.datatype: str = "colmap"

        colmap_dir: str = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")

        cams = camorph.read_cameras("COLMAP", colmap_dir)
        camera_ids: list[int] = []
        c2w_mats: list[Float[NDArray, "4 4"]] = []
        image_names: list[str] = []
        Ks_dict: dict[int, Float[NDArray, "3 3"]] = {}
        params_dict: dict[int, Float[NDArray, " 4"] | Float[NDArray, " 0"]] = {}
        imsize_dict: dict[int, tuple[int, int]] = {}
        mask_dict: dict[int, Float[NDArray, "N M"] | None] = {}
        bottom: Float[NDArray, "1 4"] = np.array([0, 0, 0, 1]).reshape(1, 4)
        for idx, cam in enumerate(cams):
            trans, rot = math_utils.convert_coordinate_systems(
                ["-y", "-z", "x"], cam.t, cam.r, tdir=[0, 0, 1], tup=[0, -1, 0], transpose=True
            )

            trans = trans.reshape(3, 1)
            rot = rot.rotation_matrix

            c2w: Float[NDArray, "4 4"] = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            c2w_mats.append(c2w)

            # support different camera intrinsics
            camera_id: int = idx
            camera_ids.append(camera_id)

            # camera intrinsics
            fx, fy, cx, cy = (
                cam.focal_length_px[0],
                cam.focal_length_px[1],
                cam.principal_point[0],
                cam.principal_point[1],
            )
            K: Float[NDArray, "3 3"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            model = cam.model
            params, camtype = self._get_distortion_params(model, cam)

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.resolution[0], cam.resolution[1])
            mask_dict[camera_id] = None
            image_names.append(Path(cam.source_image).name)
        print(f"[Parser] {len(image_names)} images, taken by {len(set(camera_ids))} cameras.")

        if len(image_names) == 0:
            raise ValueError("No images found in COLMAP.")

        camtoworlds: Float[NDArray, "N 4 4"] = np.stack(c2w_mats, axis=0)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        image_dir_suffix: str = f"_{factor}" if factor > 1 else ""
        colmap_image_dir: str = os.path.join(data_dir, "images")
        image_dir: str = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files: list[str] = sorted(_get_rel_paths(colmap_image_dir))
        image_files: list[str] = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_png", factor=factor)
            image_files = sorted(_get_rel_paths(image_dir))

        colmap_to_image: dict[str, str] = dict(zip(colmap_files, image_files))
        image_paths: list[str] = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # load one image to check the size.
        actual_image: Float[NDArray, "H W 3"] = np.array(Image.open(image_paths[0]))[..., :3]
        actual_height, actual_width = actual_image.shape[:2]

        # need to check image resolution, side length > 1600 should be downscaled
        # based on https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/camera_utils.py#L50
        max_side: int = max(actual_width, actual_height)
        global_down: float = max(max_side / 1600.0, 1.0)

        if factor == -1 and max_side > 1600:
            print("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n ")
            factor = global_down
            image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_1600px", factor=factor)

        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        self.factor: float = factor

        # Normalize the world space.
        self._normalize(camtoworlds)

        self.image_names: list[str] = image_names
        self.image_paths: list[str] = image_paths
        self.camera_ids: list[int] = camera_ids
        self.Ks_dict: dict[int, Float[NDArray, "3 3"]] = Ks_dict
        self.params_dict: dict[int, Float[NDArray, " 4"] | Float[NDArray, " 0"]] = params_dict
        self.imsize_dict: dict[int, tuple[int, int]] = imsize_dict
        self.mask_dict: dict[int, Float[NDArray, "N M"] | None] = mask_dict

        # load one image to check the size.
        actual_image: Float[NDArray, "H W 3"] = np.array(Image.open(image_paths[0]))[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width),
                int(height * s_height),
            )

        # undistortion
        self.mapx_dict: dict[int, Float[NDArray, " N M"]] = {}
        self.mapy_dict: dict[int, Float[NDArray, " N M"]] = {}
        self.roi_undist_dict: dict[int, list[int]] = {}
        self._undistort(camtype)

        # size of the scene measured by cameras
        camera_locations: Float[NDArray, "N 3"] = camtoworlds[:, :3, 3]
        scene_center: Float[NDArray, " 3"] = np.mean(camera_locations, axis=0)
        dists: Float[NDArray, " N"] = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale: float = np.max(dists)

    def _undistort(self, camtype: str) -> None:
        for camera_id in self.params_dict:
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            if camera_id not in self.Ks_dict:
                raise ValueError(f"Missing K for camera {camera_id}")
            if camera_id not in self.params_dict:
                raise ValueError(f"Missing params for camera {camera_id}")
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (width, height), cv2.CV_32FC1)
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = 1.0 + params[0] * theta**2 + params[1] * theta**4 + params[2] * theta**6 + params[3] * theta**8
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

    # WARN: this is untested for everything but pinhole
    def _get_distortion_params(self, type_: int | str, cam: Any) -> tuple[Float[NDArray, " 4"], str]:
        if type_ == "pinhole":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif type_ == "brown":
            params = np.zeros(4, dtype=np.float32)
            if cam.radial_distortion is not None:
                if len(cam.radial_distortion > 2):
                    raise ValueError(
                        "Only SIMPLE_RADIAL, RADIAL and OPENCV are supported. Camtype is probably FULL_OPENCV"
                    )
                params[: len(cam.radial_distortion)] = cam.radial_distortion
            if cam.tangential_distortion is not None:
                params[2 : 2 + len(cam.tangential_distortion)] = cam.tangential_distortion
            camtype = "perspective"
        elif type_ == "opencv_fisheye4":
            params = np.zeros(4, dtype=np.float32)
            if cam.radial_distortion is not None:
                params[: len(cam.radial_distortion)] = cam.radial_distortion
            camtype = "fisheye"
        if camtype != "perspective" and camtype != "fisheye":
            raise ValueError(f"Only perspective and fisheye cameras are supported, got {type_}")
        if type_ != "pinhole":
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")
        return params, camtype

    def _normalize(self, camtoworlds: Float[NDArray, "N 4 4"]) -> None:
        if self.normalize_data:
            camtoworlds, transform = normalize(camtoworlds)
        else:
            transform = np.eye(4)
        self.transform: Float[NDArray, "4 4"] = transform
        self.camtoworlds: Float[NDArray, "N 4 4"] = camtoworlds
