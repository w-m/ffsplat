import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import viser
from gsplat.rendering import rasterization
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from ..coding.scene_decoder import decode_gaussians
from ..render.viewer import CameraState, Viewer


@torch.no_grad()
def render_fn(
    means_t: Tensor,
    quats_t: Tensor,
    scales_t: Tensor,
    opacities_t: Tensor,
    colors: Tensor,
    sh_degree: int,
    camera_state: CameraState,
    img_wh: tuple[int, int],
) -> NDArray:
    width, height = img_wh
    camera_c2w = camera_state.c2w
    camera_K = camera_state.get_K(img_wh)
    c2w: Float[Tensor, "4 4"] = torch.from_numpy(camera_c2w).float().to("cuda")
    K: Float[Tensor, "3 3"] = torch.from_numpy(camera_K).float().to("cuda")
    viewmat: Float[Tensor, "4 4"] = torch.linalg.inv(c2w)
    # We need to separate the type annotation from the unpacking
    raster_out: tuple[Float[Tensor, "1 H W 4"], Tensor, dict] = rasterization(
        means_t,  # [N, 3]
        quats_t,  # [N, 4]
        scales_t,  # [N, 3]
        opacities_t.squeeze(),  # [N]
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


# Create render function with bound parameters
def bound_render_fn(camera_state: CameraState, img_wh: tuple[int, int], gaussians) -> NDArray:
    return render_fn(
        gaussians.means.data,
        gaussians.quaternions.data,
        gaussians.scales.data,
        gaussians.opacities.data,
        gaussians.sh.data,
        gaussians.sh_degree,
        camera_state,
        img_wh,
    )


def main():
    parser = ArgumentParser(description="Interactive compression tool parameters")
    parser.add_argument("--input", type=Path, required=True, help="Input file or directory path")
    # TODO: add support for guessing input format
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        help="Input format",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    cfg = parser.parse_args()

    gaussians = decode_gaussians(input_path=cfg.input, input_format=cfg.input_format, verbose=cfg.verbose).to("cuda")

    # The bound_render_fn needs the gaussians object. We can bind it here.
    render_fn_for_viewer = lambda camera_state, img_wh: bound_render_fn(camera_state, img_wh, gaussians)

    server = viser.ViserServer(verbose=False)
    Viewer(server=server, render_fn=render_fn_for_viewer, mode="rendering")
    print("Viewer running... Ctrl+C to exit.")
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting viewer...")
            break


if __name__ == "__main__":
    main()
