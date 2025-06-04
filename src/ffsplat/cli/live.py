import copy
import os
import shutil
import tempfile
import threading
import time
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import viser
import yaml
from gsplat.rendering import rasterization
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ..cli.eval import eval_step, get_directory_size
from ..coding.scene_decoder import decode_gaussians
from ..coding.scene_encoder import DecodingParams, EncodingParams, SceneEncoder
from ..datasets.blenderparser import BlenderParser
from ..datasets.colmapparser import ColmapParser
from ..datasets.dataset import Dataset
from ..models.transformations import get_dynamic_params
from ..render.viewer import CameraState, Viewer


def get_table_row(scene, scene_metrics: dict[str, float | int]) -> str:
    table_row = f"<tr><td>{scene}</td>"
    table_row += f"<td>{scene_metrics["psnr"]:.3f}</td>"
    table_row += f"<td>{scene_metrics["ssim"]:.4f}</td>"
    table_row += f"<td>{scene_metrics["lpips"]:.3f}</td>"
    table_row += f"<td>{scene_metrics["size"] / 1024 / 1024:.3f}</td>"
    table_row += f"<td>{scene_metrics["num_GS"]}</td>"
    table_row += "</tr>"
    return table_row


@dataclass
class SceneData:
    id: int
    description: str
    data_path: Path
    input_format: str
    encoding_params: EncodingParams
    scene_metrics: dict[str, float | int] = None
    # gaussians: Tensor


# This class defines the functionality of the viewer that goes beyond the rendering
class InteractiveConversionTool:
    table_base = """
<div style="overflow-x:auto;display: flex; justify-content: center;">
  <style>
    table {
      width: 90%;
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
        <th>id</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>LPIPS</th>
        <th>Size (MB)</th>
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
    temp_dir = tempfile.TemporaryDirectory()
    encoding_params: EncodingParams | None = None

    def __init__(
        self, input_path: Path, input_format: str, dataset_path: Path, results_path: Path, verbose: bool = False
    ):
        self.scenes: list[SceneData] = []
        self.dataset: Path = dataset_path
        self.current_scene = 0
        self.verbose = verbose
        self.enable_scene_loading: bool = True

        self.input_gaussians = decode_gaussians(input_path=input_path, input_format=input_format, verbose=self.verbose)
        self.input_gaussians = self.input_gaussians.to("cuda")

        self.gaussians = self.input_gaussians

        self.server = viser.ViserServer(verbose=False)
        self.viewer = Viewer(server=self.server, render_fn=self.bound_render_fn, mode="rendering")

        self.viewer.add_scenes(results_path)
        self.lock_scenes = threading.Lock()

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
            self.viewer.add_eval(self.full_evaluation)

        self.viewer.add_convert(self.reset_dynamic_params_gui, self.conversion_wrapper, self.live_preview_callback)

        self._add_scene("input", input_path, input_format, None)

        self.conversion_queue: list[EncodingParams] = []
        self.conversion_running: bool = False
        self.lock_queue = threading.Lock()

        self.preview_in_scenes = False

        # self.viewer.add_test_functionality(self.change_scene)

    def remove_last_scene(self):
        self.viewer.load_buttons.pop()
        self.viewer.last_scene_folder.remove()
        self.scenes.pop()
        self.update_eval_table()

    def conversion_wrapper(self, _, from_update: bool = False):
        with self.lock_scenes:
            if not from_update and self.viewer._live_preview_checkbox.value:
                if self.preview_in_scenes:
                    self.remove_last_scene()
                output_format = self.viewer._output_dropdown.value
                output_path = Path(self.temp_dir.name + "/gaussians_live_preview")
                self._add_scene(
                    self._build_description(self.encoding_params, output_format),
                    output_path,
                    "smurfx",
                    copy.deepcopy(self.encoding_params),
                )
                self._add_scene("Live preview from conversion", output_path, "smurfx", self.encoding_params)
                return
            if from_update and not self.viewer._live_preview_checkbox.value:
                return

            with self.lock_queue:
                self.conversion_queue.append(copy.deepcopy(self.encoding_params))
                self.viewer.eval_button.disabled = True

                # we only want to work on one conversion simultaneously
                if self.conversion_running:
                    return
                else:
                    self.conversion_running = True
                    self.viewer._live_preview_checkbox.disabled = True
                    self.viewer._convert_button.disabled = True

            while True:
                with self.lock_queue:
                    if not self.conversion_queue:
                        self.viewer._convert_button.disabled = False
                        self.viewer._live_preview_checkbox.disabled = False
                        self.viewer.eval_button.disabled = False
                        self.conversion_running = False
                        break
                    encoding_params = self.conversion_queue.pop()
                    self.conversion_queue.clear()
                self.viewer.scene_label.content = "Running conversion for live preview..."
                self.convert(encoding_params)
                self.viewer.scene_label.content = "Showing live preview"

    def convert(self, encoding_params: EncodingParams):
        print("Converting scene...")
        output_path = Path(self.temp_dir.name + f"/gaussians{len(self.scenes)}")
        if self.viewer._live_preview_checkbox.value:
            # clear previous live preview
            if output_path.exists() and output_path.is_dir():
                shutil.rmtree(output_path)
            if self.preview_in_scenes:
                self.remove_last_scene()
            else:
                self.preview_in_scenes = True

        encoder = SceneEncoder(
            encoding_params=encoding_params,
            output_path=output_path,
            fields=self.input_gaussians.to_field_dict(),
            decoding_params=DecodingParams(
                container_identifier="smurfx",
                container_version="0.1",
                packer="ffsplat-v0.1",
                profile=encoding_params.profile,
                profile_version=encoding_params.profile_version,
                scene=encoding_params.scene,
            ),
        )
        try:
            encoder.encode(verbose=self.verbose)
        except Exception as e:
            print(f"Exception occured: {e}")
            return

        # add scene to scene list and load it to view the scene

        if not self.viewer._live_preview_checkbox.value:
            output_format = self.viewer._output_dropdown.value
            self._add_scene(
                self._build_description(encoding_params, output_format), output_path, "smurfx", encoding_params
            )
        else:
            output_format = self.viewer._output_dropdown.value
            self._add_scene(
                self._build_description(encoding_params, output_format), output_path, "smurfx", encoding_params
            )
            # self._add_scene("Live preview from conversion", output_path, "smurfx", encoding_params)
        self.viewer.rerender(None)

    def _add_scene(self, description, data_path, input_format, encoding_params):
        self.scenes.append(
            SceneData(
                id=len(self.scenes),
                description=description,
                data_path=data_path,
                input_format=input_format,
                encoding_params=copy.deepcopy(encoding_params),
            )
        )
        self.viewer.add_to_scene_tab(
            len(self.scenes) - 1, description, self._load_scene, self._save_scene, self._save_encoding_params
        )
        self._load_scene(len(self.scenes) - 1)

    def _build_description(self, encoding_params: EncodingParams, output_format: str) -> str:
        description = "**Template**  \n"
        description += f"{output_format}  \n"
        changed_params_desc = ""

        initial_params: EncodingParams = EncodingParams.from_yaml_file(
            Path(f"src/ffsplat/conf/format/{output_format}.yaml")
        )
        for op_id, op in enumerate(self.encoding_params.ops):
            for transform_id, transform in enumerate(op["transforms"]):
                if transform != initial_params.ops[op_id]["transforms"][transform_id]:
                    if isinstance(op["input_fields"], list):
                        changed_params_desc += "```\n" + "input fields:\n" + yaml.dump(op["input_fields"])
                    else:
                        changed_params_desc += (
                            f"input fields from prefix: {op["input_fields"]["from_fields_with_prefix"]}\n"
                        )
                    changed_params_desc += yaml.dump(transform, default_flow_style=False) + "```  \n"

        if changed_params_desc != "":
            description += "**Customized Transformations**  \n"
            description += changed_params_desc

        return description

    def _save_scene(self, scene_id):
        print(f"Saving scene {scene_id}...")
        scene_path = Path(self.viewer.results_path_input.value) / Path(f"scene_{scene_id}/data")
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        shutil.copytree(self.scenes[scene_id].data_path, scene_path, dirs_exist_ok=True)

    def _save_encoding_params(self, scene_id):
        print(f"Saving encoding parameters from scene {scene_id}...")
        params_path = Path(self.viewer.results_path_input.value) / Path(f"scene_{scene_id}/encoding_params")
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        if self.scenes[scene_id].encoding_params:
            self.scenes[scene_id].encoding_params.to_yaml_file(params_path)

    def _load_scene(self, scene_id):
        print(f"Loading scene {scene_id}...")

        # update the load buttons
        if self.enable_scene_loading:
            self.viewer.load_buttons[self.current_scene].disabled = False
            self.current_scene = scene_id
            self.viewer.load_buttons[self.current_scene].disabled = True

        if self.viewer._live_preview_checkbox.value and scene_id == len(self.scenes) - 1:
            self.viewer.scene_label.content = "Showing live preview"
        else:
            self.viewer.scene_label.content = f"Showing scene: {scene_id}"

        # load scene
        scene = self.scenes[scene_id]
        self.gaussians = decode_gaussians(scene.data_path, input_format=scene.input_format, verbose=self.verbose).to(
            "cuda"
        )

        self.viewer.rerender(None)

    def update_eval_table(self):
        self.table_rows = ""
        for scene in self.scenes:
            if scene.scene_metrics is not None:
                self.table_rows += get_table_row(scene.id, scene.scene_metrics)
                self.viewer.eval_table.content = self.table_base + self.table_rows + self.table_end

    def full_evaluation(self, _):
        self._disable_load_buttons()
        self.viewer._convert_button.disabled = True
        self.viewer.eval_button.disabled = True
        self.viewer._live_preview_checkbox.disabled = True
        self.table_rows = ""
        with self.lock_scenes:
            for scene in self.scenes:
                if scene.scene_metrics is None:
                    self.viewer.eval_info.visible = True
                    self.viewer.eval_info.content = f"Running evaluation for scene {scene.id}..."
                    self.viewer.eval_progress.visible = True
                    self.viewer.eval_progress.value = 0
                    self._load_scene(scene.id)
                    self._eval(scene.id)
                self.table_rows += get_table_row(scene.id, scene.scene_metrics)
                self.viewer.eval_table.content = self.table_base + self.table_rows + self.table_end
        self.viewer.eval_info.visible = False
        self.viewer.eval_progress.visible = False
        self.viewer.eval_button.disabled = False
        self.viewer._convert_button.disabled = False
        self.viewer._live_preview_checkbox.disabled = False
        self._enable_load_buttons()

    def deactivate_live_preview(
        self,
    ):
        self.viewer._live_preview_checkbox.value = False

    def _build_transform_folder(self, transform_folder, description, transformation, transform_type):
        # clear transform folder for rebuild
        for child in tuple(transform_folder._children.values()):
            child.remove()
        dynamic_params_conf = get_dynamic_params(transformation)
        initial_values = transformation[transform_type]

        with transform_folder:
            self.viewer.server.gui.add_markdown(description)
            rebuild_fn = partial(
                self._build_transform_folder, transform_folder, description, transformation, transform_type
            )
            self._build_options_for_transformation(dynamic_params_conf, initial_values, rebuild_fn)

    def _build_options_for_transformation(self, dynamic_params_conf, initial_values, rebuild_fn):
        for item in dynamic_params_conf:
            match item:
                case {
                    "label": label,
                    "type": "number",
                    "min": minimum_value,
                    "max": maximum_value,
                    "step": stepsize,
                    "dtype": to_type,
                    **params,
                }:
                    key = params.get("set", label)

                    default_value = initial_values[key]

                    mapping = params.get("mapping")
                    inverse_mapping = params.get("inverse_mapping")
                    if inverse_mapping:
                        default_value = inverse_mapping(float(default_value))

                    number_handle = self.viewer.server.gui.add_slider(
                        label,
                        to_type(minimum_value),
                        to_type(maximum_value),
                        to_type(stepsize),
                        to_type(default_value),
                    )
                    number_handle.on_update(self.create_update_field(initial_values, key, mapping=mapping))

                case {"label": label, "type": "bool", **params}:
                    key = params.get("set", label)
                    rebuild = params.get("rebuild", False)
                    to_values = params.get("to", None)

                    if to_values:
                        checkbox_handle = self.viewer.server.gui.add_checkbox(
                            label, to_values[1] == initial_values[key]
                        )
                    else:
                        checkbox_handle = self.viewer.server.gui.add_checkbox(label, initial_values[key])

                    local_rebuild_fn = rebuild_fn
                    if rebuild:
                        local_rebuild_fn = rebuild_fn

                    checkbox_handle.on_update(
                        self.create_update_field_from_bool(initial_values, key, local_rebuild_fn, to_values)
                    )

                case {"label": label, "type": "dropdown", "values": values, **params}:
                    dropdown_handle = self.viewer.server.gui.add_dropdown(label, values, str(initial_values[label]))
                    rebuild = params.get("rebuild", False)
                    to_values = params.get("data_type", str)

                    local_rebuild_fn = rebuild_fn
                    if rebuild:
                        local_rebuild_fn = rebuild_fn

                    dropdown_handle.on_update(
                        self.create_update_field(initial_values, label, local_rebuild_fn, to_values)
                    )

                case {
                    "label": label,
                    "type": "heading",
                    "params": params_conf,
                }:
                    self.viewer.server.gui.add_markdown(f"{label}:")
                    self._build_options_for_transformation(params_conf, initial_values[label], rebuild_fn)

    def reset_dynamic_params_gui(self, _):
        self._load_encoding_params()
        self._build_convert_options()
        self.conversion_wrapper(None, True)

    def _load_encoding_params(self):
        output_format = self.viewer._output_dropdown.value
        self.encoding_params = EncodingParams.from_yaml_file(Path(f"src/ffsplat/conf/format/{output_format}.yaml"))

    def _build_convert_options(self):
        for handle in self.viewer.convert_gui_handles:
            handle.remove()

        self.viewer.convert_gui_handles = []

        with self.viewer.convert_tab:
            for operation in self.encoding_params.ops:
                # list input fields
                for transformation in operation["transforms"]:
                    # get list with customizable options

                    dynamic_transform_conf = get_dynamic_params(transformation)
                    if len(dynamic_transform_conf) == 0:
                        continue
                    transform_type = next(iter(transformation.keys()))
                    transform_folder = self.viewer.server.gui.add_folder(transform_type)
                    self.viewer.convert_gui_handles.append(transform_folder)
                    if isinstance(operation["input_fields"], list):
                        description = f"input fields: {operation["input_fields"]}"
                    else:
                        description = (
                            f"input fields from prefix: {operation["input_fields"]["from_fields_with_prefix"]}"
                        )

                    self._build_transform_folder(transform_folder, description, transformation, transform_type)

    def _eval(self, scene_id):
        print("Running evaluation...")
        device = "cuda"
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

        valloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        elapsed_time = 0
        metrics = defaultdict(list)

        imgs_path = Path(self.viewer.results_path_input.value) / Path(f"scene_{scene_id}/imgs")
        if self.viewer._live_preview_checkbox.value and scene_id == len(self.scenes) - 1:
            imgs_path = None

        for i, data in enumerate(valloader):
            elapsed_time += eval_step(
                self.gaussians,
                self.dataset.white_background,
                imgs_path,
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

        size = get_directory_size(self.scenes[scene_id].data_path)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update({
            "elapsed_time": elapsed_time,
            "num_GS": len(self.gaussians.means.data),
            "size": size,
        })
        self.scenes[scene_id].scene_metrics = stats

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

    def _disable_load_buttons(self):
        self.enable_scene_loading = False
        for button in self.viewer.load_buttons:
            button.disabled = True

    def _enable_load_buttons(self):
        self.enable_scene_loading = True
        for button in self.viewer.load_buttons:
            button.disabled = False
        self.viewer.load_buttons[self.current_scene].disabled = True

    # Create render function with bound parameters
    def bound_render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]) -> NDArray:
        return self.render_fn(
            self.gaussians.means.data,
            self.gaussians.quaternions.data,
            self.gaussians.scales.data,
            self.gaussians.opacities.data,
            self.gaussians.sh.data,
            self.gaussians.sh_degree,
            camera_state,
            img_wh,
        )

    def create_update_field(
        self,
        dict_to_update: dict[str, Any],
        key: str,
        rebuild_fn: Callable | None = None,
        to_type: type | None = None,
        mapping: Callable | None = None,
    ):
        def update_field(gui_event):
            if to_type:
                dict_to_update[key] = to_type(gui_event.target.value)
            else:
                dict_to_update[key] = gui_event.target.value
            if mapping:
                dict_to_update[key] = mapping(dict_to_update[key])

            if rebuild_fn:
                rebuild_fn()
            self.conversion_wrapper(None, from_update=True)

        return update_field

    def create_update_field_from_bool(
        self,
        dict_to_update: dict[str, Any],
        key: str,
        rebuild_fn: Callable | None = None,
        to_values: None | list[Any] = None,
    ):
        def update_field(gui_event):
            if to_values:
                dict_to_update[key] = to_values[gui_event.target.value]
            else:
                dict_to_update[key] = gui_event.target.value
            if rebuild_fn:
                rebuild_fn()
            self.conversion_wrapper(None, from_update=True)

        return update_field

    def live_preview_callback(self, _):
        print("running live preview callback")
        if not self.viewer._live_preview_checkbox.value:
            if self.preview_in_scenes:
                self.preview_in_scenes = False
                self.remove_last_scene()
                self.current_scene = len(self.scenes) - 1
                self._load_scene(len(self.scenes) - 1)
            self.viewer._convert_button.label = "Convert"
        else:
            self.viewer._convert_button.label = "Save to scene list"
        self.conversion_wrapper(None, True)


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

    parser.add_argument("--dataset-path", type=Path, required=False, help="Path to dataset for evaluation")
    parser.add_argument("--results-path", type=Path, required=False, help="Path to save images from evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    cfg = parser.parse_args()
    InteractiveConversionTool(
        input_path=cfg.input,
        input_format=cfg.input_format,
        dataset_path=cfg.dataset_path,
        results_path=cfg.results_path,
        verbose=cfg.verbose,
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()
