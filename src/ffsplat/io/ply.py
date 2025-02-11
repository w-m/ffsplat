import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from plyfile import PlyData


def load_ply(
    path: str,
) -> tuple[
    Float[NDArray, "num_gaussians 3"],  # means
    Float[NDArray, "num_gaussians 4"],  # quats
    Float[NDArray, "num_gaussians 3"],  # scales
    Float[NDArray, " num_gaussians"],  # opacities
    Float[NDArray, "num_gaussians 1 3"],  # sh0
    Float[NDArray, "num_gaussians S 3"],  # shN
]:
    data = PlyData.read(path)
    means = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    quats = np.vstack([
        data["vertex"]["rot_0"],
        data["vertex"]["rot_1"],
        data["vertex"]["rot_2"],
        data["vertex"]["rot_3"],
    ]).T
    scales = np.vstack([
        data["vertex"]["scale_0"],
        data["vertex"]["scale_1"],
        data["vertex"]["scale_2"],
    ]).T
    opacities = data["vertex"]["opacity"]

    # shape is (N, 1, 3)
    sh0 = np.vstack([data["vertex"]["f_dc_0"], data["vertex"]["f_dc_1"], data["vertex"]["f_dc_2"]]).T[:, None, :]

    f_rest_count = len([p.name for p in data.elements[0].properties if p.name.startswith("f_rest_")])

    # shape is (N, S, 3)
    shN = np.vstack([data["vertex"][f"f_rest_{i}"] for i in range(f_rest_count)]).T
    shN = shN.reshape(means.shape[0], 3, f_rest_count // 3).transpose(0, 2, 1)

    return means, quats, scales, opacities, sh0, shN
