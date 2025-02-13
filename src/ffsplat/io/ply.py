from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ..models.attribute import NamedAttribute
from ..models.gaussians import Gaussians


def load_ply(path: Path) -> Gaussians:
    """Load a PLY file into a Gaussians instance

    Args:
        path: Path to the PLY file

    Returns:
        Gaussians instance containing the loaded data
    """
    data = PlyData.read(str(path))

    # Load 3D positions
    means = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T

    # Load rotations as quaternions
    quats = np.vstack([
        data["vertex"]["rot_0"],
        data["vertex"]["rot_1"],
        data["vertex"]["rot_2"],
        data["vertex"]["rot_3"],
    ]).T

    # Load scale factors (already in log space)
    scales = np.vstack([
        data["vertex"]["scale_0"],
        data["vertex"]["scale_1"],
        data["vertex"]["scale_2"],
    ]).T

    # Load opacity values (already in logit space)
    opacities = data["vertex"]["opacity"]

    # Load base color (spherical harmonics DC term)
    sh0 = np.vstack([data["vertex"]["f_dc_0"], data["vertex"]["f_dc_1"], data["vertex"]["f_dc_2"]]).T[
        :, None, :
    ]  # shape: (N, 1, 3)

    # Load higher-order spherical harmonics coefficients
    f_rest_count = len([p.name for p in data.elements[0].properties if p.name.startswith("f_rest_")])
    shN = np.vstack([data["vertex"][f"f_rest_{i}"] for i in range(f_rest_count)]).T
    shN = shN.reshape(means.shape[0], 3, f_rest_count // 3).transpose(0, 2, 1)  # shape: (N, S, 3)

    # Combine sh0 and shN into a single array
    sh_combined = np.concatenate([sh0, shN], axis=1)

    return Gaussians(
        means_attr=NamedAttribute.from_packed_data(name="means", packed_data=means),
        quaternions_attr=NamedAttribute.from_packed_data(name="quaternions", packed_data=quats),
        scales_attr=NamedAttribute.from_packed_data(name="scales", packed_data=scales, remapping="exp"),
        opacities_attr=NamedAttribute.from_packed_data(name="opacities", packed_data=opacities, remapping="sigmoid"),
        sh_attr=NamedAttribute.from_packed_data(name="sh", packed_data=sh_combined),
    )


def save_ply(gaussians: Gaussians, path: Path) -> None:
    """Save a Gaussians instance to a PLY file.

    Args:
        gaussians: Gaussians instance to save
        path: Path where to save the PLY file
    """

    means = gaussians.means_attr.packed_data.cpu().numpy()
    quats = gaussians.quaternions_attr.packed_data.cpu().numpy()
    scales = gaussians.scales_attr.packed_data.cpu().numpy()
    opacities = gaussians.opacities_attr.packed_data.cpu().numpy()

    # Handle spherical harmonics
    sh_data = gaussians.sh_attr.packed_data.cpu().numpy()
    sh0 = sh_data[:, :1, :]  # (N, 1, 3)
    shN = sh_data[:, 1:, :]  # (N, S, 3)

    # Calculate number of spherical harmonics coefficients
    sh_rest_count = shN.shape[1] * 3  # S * 3 for RGB

    # Create dtype for vertex data
    dtype_list = [
        # Position
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        # Rotation (quaternion)
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        # Scale factors
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        # Opacity
        ("opacity", "f4"),
        # Base color (DC term)
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    # Add fields for higher-order spherical harmonics
    for i in range(sh_rest_count):
        dtype_list.append((f"f_rest_{i}", "f4"))

    # Create structured array for vertex data
    vertex_data = np.empty(len(means), dtype=dtype_list)

    # Fill position data
    vertex_data["x"] = means[:, 0]
    vertex_data["y"] = means[:, 1]
    vertex_data["z"] = means[:, 2]

    # Fill rotation data
    vertex_data["rot_0"] = quats[:, 0]
    vertex_data["rot_1"] = quats[:, 1]
    vertex_data["rot_2"] = quats[:, 2]
    vertex_data["rot_3"] = quats[:, 3]

    # Fill scale data
    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]

    # Fill opacity
    vertex_data["opacity"] = opacities

    # Fill base color (DC term)
    vertex_data["f_dc_0"] = sh0[:, 0, 0]
    vertex_data["f_dc_1"] = sh0[:, 0, 1]
    vertex_data["f_dc_2"] = sh0[:, 0, 2]

    # Fill higher-order spherical harmonics
    for s in range(shN.shape[1]):  # For each SH degree
        for c in range(3):  # For each color channel
            idx = s * 3 + c
            vertex_data[f"f_rest_{idx}"] = shN[:, s, c]

    # Create PLY element and save
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex_element], text=False).write(str(path))
