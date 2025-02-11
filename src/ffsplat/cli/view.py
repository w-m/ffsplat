import math
import time
from argparse import ArgumentParser

import torch
import viser
from gsplat.rendering import rasterization
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from ..io.ply import load_ply
from ..render.viewer import CameraState, Viewer

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive compression tool parameters")
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()

    means, quats, scales, opacities, sh0, shN = load_ply(args.data_path)
    means_t: Float[Tensor, "num_gaussians 3"] = torch.from_numpy(means).to("cuda")
    quats_t: Float[Tensor, "num_gaussians 4"] = torch.from_numpy(quats).to("cuda")
    scales_t: Float[Tensor, "num_gaussians 3"] = torch.from_numpy(scales).to("cuda")
    opacities_t: Float[Tensor, " num_gaussians"] = torch.from_numpy(opacities).to("cuda")
    sh0_t: Float[Tensor, "num_gaussians 1 3"] = torch.from_numpy(sh0).to("cuda")
    shN_t: Float[Tensor, "num_gaussians S 3"] = torch.from_numpy(shN).to("cuda")
    colors: Float[Tensor, "num_gaussians S_total 3"] = torch.cat([sh0_t, shN_t], dim=-2).to("cuda")

    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    scales_t = torch.exp(scales_t)
    opacities_t = torch.sigmoid(opacities_t)

    @torch.no_grad()
    def render_fn(camera_state: CameraState, img_wh: tuple[int, int]) -> NDArray:
        width, height = img_wh
        camera_c2w = camera_state.c2w
        camera_K = camera_state.get_K(img_wh)
        c2w: Float[Tensor, "4 4"] = torch.from_numpy(camera_c2w).float().to("cuda")
        K: Float[Tensor, "3 3"] = torch.from_numpy(camera_K).float().to("cuda")
        viewmat: Float[Tensor, "4 4"] = torch.linalg.inv(c2w)
        # We need to separate the type annotation from the unpacking
        raster_out: tuple[Float[Tensor, "1 H W 4"], Tensor, Tensor] = rasterization(
            means_t,  # [N, 3]
            quats_t,  # [N, 4]
            scales_t,  # [N, 3]
            opacities_t,  # [N]
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
        render_colors, _, _ = raster_out
        render_rgbs: Float[NDArray, "H W 3"] = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(verbose=False)
    viewer = Viewer(server=server, render_fn=render_fn, mode="rendering")
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)
