import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from ..coding.scene_decoder import decode_gaussians
from ..datasets.blenderparser import BlenderParser
from ..datasets.colmapparser import ColmapParser
from ..datasets.dataset import Dataset
from ..models.gaussians import Gaussians


@torch.no_grad()
def rasterize_splats(
    gaussians: Gaussians,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    masks: Tensor | None = None,
    use_white_background: bool = False,
) -> tuple[Tensor, Tensor, dict]:
    colors = gaussians.sh
    background = torch.ones(1, colors.shape[-1], device="cuda") if use_white_background else None
    render_colors, render_alphas, info = rasterization(
        means=gaussians.means,
        quats=gaussians.quaternions,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        sh_degree=gaussians.sh_degree,
        backgrounds=background,
    )
    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info


def eval_step(
    gaussians: Gaussians,
    white_background: bool,
    results_path: Path | None,
    device: str,
    step: int,
    metrics: dict[str, list],
    psnr: callable,
    ssim: callable,
    lpips: callable,
    data: dict[str, Float[NDArray, "3 3"] | Float[NDArray, "4 4"] | Float[NDArray, "N M"] | int],
) -> None:
    camtoworlds = data["camtoworld"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device)
    masks = data["mask"].to(device) if "mask" in data else None
    height, width = pixels.shape[1:3]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    colors, _, _ = rasterize_splats(
        gaussians=gaussians,
        camtoworlds=camtoworlds,
        Ks=Ks,
        width=width,
        height=height,
        masks=masks,
        use_white_background=white_background,
    )  # [1, H, W, 3]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0

    colors = torch.clamp(colors, 0.0, 1.0)
    canvas_list = [pixels, colors]

    # write images
    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
    canvas = (canvas * 255).astype(np.uint8)

    if results_path is not None:
        Image.fromarray(canvas).save(results_path / f"eval_{step:04d}.png")

    pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
    colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
    metrics["psnr"].append(psnr(colors_p, pixels_p))
    metrics["ssim"].append(ssim(colors_p, pixels_p))
    metrics["lpips"].append(lpips(colors_p, pixels_p))
    return elapsed_time


def evaluation(gaussians: Gaussians, valset: Dataset, results_path: Path) -> None:
    """Entry for evaluation."""
    print("Running evaluation...")
    device = "cuda"
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    elapsed_time = 0
    metrics = defaultdict(list)

    for i, data in enumerate(tqdm(valloader)):
        elapsed_time += eval_step(
            gaussians, valset.white_background, results_path, device, i, metrics, psnr, ssim, lpips, data
        )

    elapsed_time /= len(valloader)

    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats.update({
        "elapsed_time": elapsed_time,
        "num_GS": len(gaussians.means),
    })
    print(
        f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
        f"Time: {stats['elapsed_time']:.3f}s/image "
        f"Number of GS: {stats['num_GS']}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive compression tool parameters")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--results-path", type=Path)

    args = parser.parse_args()

    gaussians = decode_gaussians(args.data_path, "3DGS-INRIA.ply").to("cuda")

    dataset_dir = args.dataset_path

    colmap_dir = os.path.join(dataset_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(dataset_dir, "sparse")
    if os.path.exists(colmap_dir):
        dataparser = ColmapParser(dataset_dir)
    elif os.path.exists(os.path.join(dataset_dir, "transforms_train.json")):
        dataparser = BlenderParser(dataset_dir)
    else:
        raise ValueError("could not identify type of dataset")

    dataset = Dataset(dataparser)

    evaluation(gaussians, dataset, args.results_path)
