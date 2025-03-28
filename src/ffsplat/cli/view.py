import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from ffsplat.coding.scene_encoder import DecodingParams, EncodingParams, SceneEncoder
from ffsplat.datasets.blenderparser import BlenderParser
from ffsplat.datasets.colmapparser import ColmapParser
from ffsplat.datasets.dataset import Dataset
from ffsplat.render.viewer import CameraState, Viewer

# TODO: this is a duplicate with the viewer
encoding_formats: list[str] = ["3DGS_INRIA_ply", "SOG-web"]  # available formats

operations: dict[str, list[str]] = {"remapping": ["inverse-sigmoid", "log"]}  # methods for each operation


def create_update_field(dict_to_update: dict[str, Any], key: str):
    def update_field(gui_event):
        print("I am here")
        dict_to_update[key] = gui_event.target.value

    return update_field


def get_table_row(scene, scene_metrics: dict[str, float | int]) -> str:
    table_row = f"<tr><td>{scene}</td>"
    table_row += f"<td>{scene_metrics['psnr']:.3f}</td>"
    table_row += f"<td>{scene_metrics['ssim']:.4f}</td>"
    table_row += f"<td>{scene_metrics['lpips']:.3f}</td>"
    table_row += f"<td>{scene_metrics['num_GS']}</td>"
    table_row += "</tr>"
    return table_row


# This class defines the functionality of the viewer that goes beyond the rendering
class InteractiveConversionTool:
    table_base = """
<div style="overflow-x:auto;display: flex; justify-content: center;">
  <style>
    table {
      width: 80%;
      border-collapse: collapse;
      table-layout: auto;
    }

    th, td {
      padding: 8px;
      text-align: left;
      border: 1px solid grey;
      word-wrap: break-word; /* Ensures long words are broken onto the next line */
      white-space: normal; /* Prevents text from overflowing horizontally */
    }

    th {
      background-color: #f2f2f2;
    }

  </style>

  <table>
    <thead>
      <tr>
        <th>Data</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>LPIPS</th>
        <th>Number of GS</th>
      </tr>
    </thead>
    <tbody>
"""
    table_end = """
    </tbody>
  </table>
</div>
"""
    table_rows = ""
    encoding_params: EncodingParams | None = None

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
            self.viewer.add_eval(self._eval)

        self.viewer.add_convert(self._build_convert_options, self.convert)

        # self.viewer.add_test_functionality(self.change_scene)

    def convert(self, _):
        # we shouldn't redo the encoding completely every time, as resorting takes a lot of time
        encoding_params = self.encoding_params

        # TODO: use tempfile
        encoder = SceneEncoder(
            encoding_params=encoding_params,
            output_path=Path("./temp/gaussians"),
            fields=self.gaussians.to_dict(),
            decoding_params=DecodingParams(
                container_identifier="smurfx",
                container_version="0.1",
                packer="ffsplat-v0.1",
                profile=encoding_params.profile,
                profile_version=encoding_params.profile_version,
                scene=encoding_params.scene,
            ),
        )
        # encode writes yaml to output_path but ply to working directory
        # also container meta does not contain the files and the scene data
        encoder.encode()

    def _build_convert_options(self, _):
        # this is not a class function of the viewer to store the handles in a dict in this class. this is to separate the logic of the conversion from the viewer. this way we can store the handles and their data needed for the conversion in this class
        # clear previous gui conversion handles
        # TODO: should we store the handles in the viewer? or should we store them in this class?
        # is the split between the viewer necessary at all? i did this to separate the logic of the evaluation from the viewer, but the split is not as clean as initially thought
        for handle in self.viewer.convert_gui_handles:
            handle.remove()

        # load the default encoding params
        self.encoding_handler = {}
        output_format = self.viewer._output_dropdown.value
        self.encoding_params = EncodingParams.from_yaml_file(Path(f"src/ffsplat/conf/format/{output_format}.yaml"))
        print(self.encoding_params)

        # TODO: why is this in this with block?
        with self.viewer.convert_folder:
            self.viewer.convert_gui_handles = []

            for field_name, field_operations in self.encoding_params.fields.items():
                with self.viewer.server.gui.add_folder(field_name) as field_folder:
                    field_is_customizable = False
                    for idx, field_operation in enumerate(field_operations):
                        for operation, operation_options in operations.items():
                            if operation in field_operation:
                                field_is_customizable = True
                                dropdown_handle = self.viewer.server.gui.add_dropdown(
                                    operation, operation_options, field_operation[operation]["method"]
                                )
                                dropdown_handle.on_update(
                                    create_update_field(
                                        self.encoding_params.fields[field_name][idx][operation], "method"
                                    )
                                )

                    if not field_is_customizable:
                        field_folder.remove()
                    else:
                        self.viewer.convert_gui_handles.append(field_folder)

    def change_scene(self, _):
        print("Changing scene...")
        new_gaussians = decode_gaussians(
            input_path=Path(
                "../mini-splatting2/results/minisplat_dense/360/flowers/point_cloud/iteration_18000/point_cloud.ply"
            ),
            input_format=self.input_format,
        )
        self.gaussians = new_gaussians.to("cuda")
        self.viewer.rerender(_)
        print("Scene changed.")

    # TODO:
    # how to handle multiple scenes after running conversion?
    # should evaluate original only once and only recompute the converted scene
    # store results in class and wrapper for eval that runs for the necessary scenes?
    # how do i store converted scenes?
    # could store in temp folder or just a list of scenes?
    # add file size to table? and to eval.py
    def _eval(self, _):
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
            self.viewer.eval_progress.value = (i + 1) / len(valloader) * 100

        elapsed_time /= len(valloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update({
            "elapsed_time": elapsed_time,
            "num_GS": len(self.gaussians.means),
        })
        self.table_rows += get_table_row("input", stats)
        self.viewer.eval_table.content = self.table_base + self.table_rows + self.table_end

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
    # TODO: clean up temp folder on keyboard interrupt
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
