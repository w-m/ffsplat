import numpy as np
from plyfile import PlyData

from ..models.gaussians import Gaussians


def load_ply(path: str) -> Gaussians:
    """Load a PLY file into a Gaussians instance"""
    data = PlyData.read(path)

    # Load 3D positions
    means = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T

    # Load rotations as quaternions
    quats = np.vstack([
        data["vertex"]["rot_0"],
        data["vertex"]["rot_1"],
        data["vertex"]["rot_2"],
        data["vertex"]["rot_3"],
    ]).T

    # Load scale factors
    scales = np.vstack([
        data["vertex"]["scale_0"],
        data["vertex"]["scale_1"],
        data["vertex"]["scale_2"],
    ]).T

    # Load opacity values
    opacities = data["vertex"]["opacity"]

    # Load base color (spherical harmonics DC term)
    sh0 = np.vstack([data["vertex"]["f_dc_0"], data["vertex"]["f_dc_1"], data["vertex"]["f_dc_2"]]).T[
        :, None, :
    ]  # shape: (N, 1, 3)

    # Load higher-order spherical harmonics coefficients
    f_rest_count = len([p.name for p in data.elements[0].properties if p.name.startswith("f_rest_")])
    shN = np.vstack([data["vertex"][f"f_rest_{i}"] for i in range(f_rest_count)]).T
    shN = shN.reshape(means.shape[0], 3, f_rest_count // 3).transpose(0, 2, 1)  # shape: (N, S, 3)

    return Gaussians.from_numpy(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
    )
