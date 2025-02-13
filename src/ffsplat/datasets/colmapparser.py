import os

import cv2
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm

from .dataparser import DataParser
from .normalize import normalize


def _get_rel_paths(path_dir: str) -> list[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
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
        img = Image.read(image_path)
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
    ):
        self.data_dir = data_dir
        self.normalize_data = normalize_data
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = {}
        params_dict = {}
        imsize_dict = {}  # width, height
        mask_dict = {}
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            params, camtype = self._get_distortion_params(type_, cam)

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (
                cam.width // abs(factor),
                cam.height // abs(factor),
            )
            mask_dict[camera_id] = None
        print(f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        image_dir_suffix = f"_{factor}" if factor > 1 else ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_png", factor=factor)
            image_files = sorted(_get_rel_paths(image_dir))

        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # load one image to check the size.
        actual_image = np.array(Image.open(image_paths[0]))[..., :3]
        actual_height, actual_width = actual_image.shape[:2]

        # need to check image resolution, side length > 1600 should be downscaled
        # based on https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/camera_utils.py#L50
        max_side = max(actual_width, actual_height)
        global_down = max_side / 1600.0

        if factor == -1 and max_side > 1600:
            print("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n ")
            factor = global_down
        image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_1600px", factor=factor)
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        self.factor = factor

        # 3D points and {image_name -> [point_idx]}
        points, points_err, points_rgb, point_indices = self._load_points(manager)

        # Normalize the world space.
        self._normalize(camtoworlds, points)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]

        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[:2, :] /= factor
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width / global_down),
                int(height * s_height / global_down),
            )

        # undistortion
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}
        self.undistort(camtype)

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def _undistort(self, camtype: str):
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

    def _get_distortion_params(type_: int | str, cam) -> tuple[Float[NDArray], str]:
        if type_ == 0 or type_ == "SIMPLE_PINHOLE" or type_ == 1 or type_ == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 3 or type_ == "RADIAL":
            params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 4 or type_ == "OPENCV":
            params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
            camtype = "fisheye"
        if camtype != "perspective" and camtype != "fisheye":
            raise ValueError(f"Only perspective and fisheye cameras are supported, got {type_}")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")
        return params, camtype

    def _load_points(self, manager):
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = {}

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}
        return points, points_err, points_rgb, point_indices

    def _normalize(self, camtoworlds, points):
        if self.normalize_data:
            camtoworlds, points, transform = normalize(camtoworlds, points)
        else:
            transform = np.eye(4)
        self.transform = transform  # np.ndarray, (4, 4)
        self.points = points  # np.ndarray, (num_points, 3)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
