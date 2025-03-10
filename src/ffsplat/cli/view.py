import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
import viser
from gsplat.rendering import rasterization
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ffsplat.cli.eval import eval_step
from ffsplat.coding.scene_decoder import decode_gaussians
from ffsplat.datasets.blenderparser import BlenderParser
from ffsplat.datasets.colmapparser import ColmapParser
from ffsplat.datasets.dataset import Dataset
from ffsplat.render.viewer import CameraState, Viewer


# This class defines the functionality of the viewer that goes beyond the rendering
class InteractiveConversionTool:
    def __init__(self, input_path: Path, input_format: str, dataset_path: Path, results_path: Path):
        self.input_format = input_format
        self.dataset = dataset_path
        self.results = results_path
        self.write_images = results_path is not None
        self.results_path = results_path

        self.input_gaussians = decode_gaussians(input_path=input_path, input_format=input_format)
        self.gaussians = self.input_gaussians.to("cuda")

        self.server = viser.ViserServer(verbose=False)
        self.viewer = Viewer(server=self.server, render_fn=self.bound_render_fn, mode="rendering")

        if dataset_path is not None:
            colmap_path = os.path.join(dataset_path, "sparse/0/")
            if not os.path.exists(colmap_path):
                colmap_path = os.path.join(dataset_path, "sparse")
            if os.path.exists(colmap_path):
                dataparser = ColmapParser(dataset_path)
            elif os.path.exists(os.path.join(dataset_path, "transforms_train.json")):
                dataparser = BlenderParser(dataset_path)
            else:
                raise ValueError("could not identify type of dataset")

            self.dataset = Dataset(dataparser)
            self.viewer.add_eval(self.eval)

    def eval(self, _):
        print("Running evaluation...")
        device = "cuda"
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

        valloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        elapsed_time = 0
        metrics = defaultdict(list)

        for i, data in enumerate(valloader):
            elapsed_time += eval_step(
                self.gaussians,
                self.dataset.white_background,
                self.results_path,
                device,
                i,
                metrics,
                psnr,
                ssim,
                lpips,
                data,
            )

        elapsed_time /= len(valloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update({
            "elapsed_time": elapsed_time,
            "num_GS": len(self.gaussians.means),
        })
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
            f"Time: {stats['elapsed_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )

    @torch.no_grad()
    def render_fn(
        self,
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

    # Create render function with bound parameters
    def bound_render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]) -> NDArray:
        return self.render_fn(
            self.gaussians.means,
            self.gaussians.quaternions,
            self.gaussians.scales,
            self.gaussians.opacities,
            self.gaussians.sh,
            self.gaussians.sh_degree,
            camera_state,
            img_wh,
        )


def main(parser: ArgumentParser):
    cfg = parser.parse_args()
    InteractiveConversionTool(
        input_path=cfg.input,
        input_format=cfg.input_format,
        dataset_path=cfg.dataset_path,
        results_path=cfg.results_path,
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
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

    parser.add_argument("--dataset-path", type=Path, required=False, help="Path to dataset for evaluation")
    parser.add_argument("--results-path", type=Path, required=False, help="Path to save images from evaluation")

    main(parser)
