# from https://github.com/nerfstudio-project/gsplat/blob/0880d2b471e6650d458aa09fe2b2834531f6e93b/examples/datasets/normalize.py

import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"transformation matrix has shape {matrix.shape} instead of (4, 4)")
    if len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4):
        raise ValueError(f"camtoworlds has shape {camtoworlds.shape} instead of (N, 4, 4)")
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize(camtoworlds):
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    return camtoworlds, T1
