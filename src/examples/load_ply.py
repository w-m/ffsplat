import math
import time
from argparse import ArgumentParser

import numpy as np
import torch
import viser
from gsplat.rendering import rasterization
from plyfile import PlyData

from view import CameraState, Viewer


def load_ply(path):
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive compression tool parameters")
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()
    means, quats, scales, opacities, sh0, shN = load_ply(args.data_path)

    means = torch.from_numpy(means).to("cuda")
    quats = torch.from_numpy(quats).to("cuda")
    scales = torch.from_numpy(scales).to("cuda")
    opacities = torch.from_numpy(opacities).to("cuda")
    sh0 = torch.from_numpy(sh0).to("cuda")
    shN = torch.from_numpy(shN).to("cuda")
    colors = torch.cat([sh0, shN], dim=-2).to("cuda")
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    scales = torch.exp(scales)
    opacities = torch.sigmoid(opacities)

    @torch.no_grad()
    def render_fn(camera_state: CameraState, img_wh: tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")
        viewmat = c2w.inverse()

        render_colors, _, _ = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(verbose=False)
    viewer = Viewer(server=server, render_fn=render_fn, mode="rendering")
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)
