import copy
import io
import os
import shutil
import sys
import tempfile
import threading
import time
from argparse import ArgumentParser
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, override

import numpy as np
import torch
import viser
import yaml
from gsplat.rendering import rasterization
from jaxtyping import Float, Float32, UInt8
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

available_output_format: list[str] = [
    "SOG-PlayCanvas",
    "SOG-web",
    "3DGS_INRIA_ply",
    "3DGS_INRIA_nosh_ply",
    "SOG-web-png",
    "SOG-web-nosh",
    "SOG-web-sh-split",
]


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


class InteractiveConversionViewer(Viewer):
    """This class is an extension of the nerfview viewer for the InteractiveConversionTool."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, tuple[int, int]],
            UInt8[NDArray, "H W 3"] | tuple[UInt8[NDArray, "H W 3"], Float32[NDArray, "H W"] | None],
        ],
        mode: Literal["rendering", "training"] = "rendering",
    ) -> None:
        super().__init__(server, render_fn, mode)
        self.convert_gui_handles: list = []
        self.load_buttons: list = []

    @override
    def _define_guis(self):
        self.server.gui.configure_theme(control_width="large", control_layout="fixed")
        self.scene_label = self.server.gui.add_markdown("Showing scene: None")
        super()._define_guis()

        # ------------------------------------------------------------------
        # Stdout viewer - collapsible folder showing recent console output.
        # ------------------------------------------------------------------
        with self.server.gui.add_folder("Stdout Logs", expand_by_default=True) as self._stdout_folder:
            # Use a <pre> block inside a scrollable container. Keep a
            # reference so we can update it from the stdout redirector.
            self._stdout_html = self.server.gui.add_html(
                "<pre style='white-space:pre-wrap;font-family:monospace;margin:0;padding:8px;'></pre>"
            )

        # Hook Python stdout/stderr so new prints appear in the GUI.
        # We keep the original streams so behaviour in the terminal is
        # unchanged.
        class _GuiStdout(io.TextIOBase):
            """Wraps an underlying stream and mirrors the last few lines to a
            viser HTML handle."""

            def __init__(self, orig, html_handle):
                self._orig = orig
                self._html_handle = html_handle
                self._buf: deque[str] = deque(maxlen=5)
                # Access to GUI API for sending JS scroll command.
                self._gui_api = html_handle._impl.gui_api  # type: ignore[attr-defined]

            # Required TextIOBase overrides ---------------------------------
            def write(self, text: str):  # type: ignore[override]
                self._orig.write(text)

                # If tqdm or similar writes carriage-return updates, update
                # the last line instead of appending.
                if "\r" in text and "\n" not in text:
                    last = text.split("\r")[-1].rstrip()
                    if self._buf:
                        self._buf[-1] = last
                    else:
                        self._buf.append(last)
                else:
                    for part in text.split("\n"):
                        if part == "":
                            continue
                        self._buf.append(part)

                self._refresh_html()
                return len(text)

            def flush(self):  # type: ignore[override]
                self._orig.flush()

            # Internal ------------------------------------------------------
            def _refresh_html(self):
                content = (
                    "<pre style='white-space:pre-wrap;font-family:monospace;margin:0;padding:8px;'>"
                    + "\n".join(self._buf)
                    + "</pre>"
                )
                self._html_handle.content = content

        # Redirect once.
        if not isinstance(sys.stdout, _GuiStdout):
            gui_stream = _GuiStdout(sys.stdout, self._stdout_html)
            sys.stdout = gui_stream  # type: ignore[assignment]
            sys.stderr = gui_stream  # mirror stderr as well.
        self.tab_group = self.server.gui.add_tab_group()

    @override
    def _connect_client(self, client: viser.ClientHandle):
        super()._connect_client(client)
        client.camera.up_direction = np.array([0, -1, 0])

    def add_scenes(self, results_path: Path):
        self.scenes_tab = self.tab_group.add_tab("Scenes")
        with self.scenes_tab:
            self.results_path_input = self.server.gui.add_text("store at:", results_path.as_posix())

    def add_eval(self, eval_fn: Callable):
        with self.tab_group.add_tab("Evaluation") as self.eval_tab:
            self.eval_button = self.server.gui.add_button("Run evaluation")
            self.eval_button.on_click(eval_fn)
            self.eval_info = self.server.gui.add_markdown("")
            self.eval_info.visible = False
            self.eval_progress = self.server.gui.add_progress_bar(0.0)
            self.eval_progress.visible = False
            self.eval_table = self.server.gui.add_html("")

    def add_convert(
        self, reset_dynamic_params_gui_fn: Callable, convert_button_fn: Callable, live_preview_fn: Callable
    ):
        with self.tab_group.add_tab("Convert") as self.convert_tab:
            self._output_dropdown = self.server.gui.add_dropdown("Output format", available_output_format)
            self._output_dropdown.on_update(reset_dynamic_params_gui_fn)
            self._convert_button = self.server.gui.add_button("Convert")
            self._convert_button.on_click(convert_button_fn)
            self._live_preview_checkbox = self.server.gui.add_checkbox("live preview", False)
            self._live_preview_checkbox.on_update(live_preview_fn)
        reset_dynamic_params_gui_fn(None)

    def add_to_scene_tab(
        self, scene_id: int, description: str, load_fn: Callable, save_fn: Callable, save_params_fn: Callable
    ):
        with self.scenes_tab, self.server.gui.add_folder(f"Scene {scene_id}") as self.last_scene_folder:
            self.last_scene_description = self.server.gui.add_markdown(description)
            load_button = self.server.gui.add_button("Load scene")
            load_button.on_click(lambda _: load_fn(scene_id))
            save_button = self.server.gui.add_button("Save scene")
            save_button.on_click(lambda _: save_fn(scene_id))
            save_button = self.server.gui.add_button("Save encoding parameters")
            save_button.on_click(lambda _: save_params_fn(scene_id))
            self.load_buttons.append(load_button)

    def add_test_functionality(self, test_fn: Callable):
        """This function is for development only. Adds a button to run a new function."""
        with self.server.gui.add_folder("Test") as self.test_folder:
            self._test_button = self.server.gui.add_button("Run test")
            self._test_button.on_click(test_fn)
            self._test_button.on_click(self.rerender)


class InteractiveConversionTool:
    """This class manages the data needed for the interactive tool.
    It defines the functionality of GUI elements of the InteractiveConversionViewer and manages
    the interactions between the Runner and the InteractiveConversionViewer.
    Dynamically build GUI elements are also added by this class.
    """

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

    enable_scene_loading: bool = True

    live_preview_active: bool = False

    def __init__(
        self, input_path: Path, input_format: str, dataset_path: Path, results_path: Path, verbose: bool = False
    ):
        self.dataset: Path = dataset_path
        self.verbose = verbose
        self.scenes: list[SceneData] = []

        # Load gaussians
        self.input_gaussians = decode_gaussians(input_path=input_path, input_format=input_format, verbose=self.verbose)
        self.input_gaussians = self.input_gaussians.to("cuda")

        # set the current gaussians to the input gaussians
        self.current_scene = 0
        self.gaussians = self.input_gaussians

        self.runner: Runner = Runner(self)

        # start viewer
        self.server = viser.ViserServer(verbose=False)
        self.viewer = InteractiveConversionViewer(server=self.server, render_fn=self.bound_render_fn, mode="rendering")

        # adds scenes tab to viewer
        if results_path is None:
            results_path = Path("./results")
        results_path = results_path / Path(datetime.now().replace(microsecond=0).isoformat())
        self.viewer.add_scenes(results_path)

        # add evaluation tab to viewer
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

        # add convert tab to viewer
        self.viewer.add_convert(self.reset_dynamic_params_gui, self.convert_button_callback, self.live_preview_callback)

        # input gaussians are added to the scenes
        self.add_scene("input", input_path, input_format, None)

        # self.viewer.add_test_functionality(self.change_scene)

    def remove_last_scene(self):
        self.viewer.load_buttons.pop()
        self.viewer.last_scene_folder.remove()
        self.scenes.pop()
        self.update_eval_table()

    def add_scene(self, description, data_path, input_format, encoding_params):
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
            len(self.scenes) - 1, description, self.load_scene, self._save_scene, self._save_encoding_params
        )
        self.load_scene(len(self.scenes) - 1)

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

    def load_scene(self, scene_id):
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
        for idx, scene in enumerate(self.scenes):
            if scene.scene_metrics is not None:
                if idx == len(self.scenes) - 1 and self.live_preview_active:
                    self.table_rows += get_table_row("live", scene.scene_metrics)
                else:
                    self.table_rows += get_table_row(scene.id, scene.scene_metrics)
                self.viewer.eval_table.content = self.table_base + self.table_rows + self.table_end

    def full_evaluation(self, _):
        self.disable_load_buttons()
        self.disable_convert_ui()
        self.viewer.eval_button.disabled = True
        self.viewer.eval_info.visible = True
        self.viewer.eval_info.content = "Running evaluation..."
        self.update_eval_progress(0)
        self.viewer.eval_progress.visible = True

        self.runner.run_evaluation()

        self.viewer.eval_info.visible = False
        self.viewer.eval_progress.visible = False
        self.viewer.eval_button.disabled = False
        self.enable_convert_ui()
        self.enable_load_buttons()

    def _build_transform_folder(self, transform_folder, operation, transformation, transform_type):
        # clear transform folder for rebuild
        for child in tuple(transform_folder._children.values()):
            child.remove()

        if isinstance(operation["input_fields"], list):
            input_field = operation["input_fields"]
            description = f"input fields: {input_field}"
        else:
            input_field = operation["input_fields"]["from_fields_with_prefix"]
            description = f"input fields from prefix: {input_field}"
        dynamic_params_conf = get_dynamic_params(transformation, input_field)
        initial_values = transformation[transform_type]

        with transform_folder:
            self.viewer.server.gui.add_markdown(description)
            rebuild_fn = partial(
                self._build_transform_folder, transform_folder, operation, transformation, transform_type
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
        self.runner.update_live_preview(copy.deepcopy(self.encoding_params))

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

                    dynamic_transform_conf = get_dynamic_params(transformation, operation["input_fields"])
                    if len(dynamic_transform_conf) == 0:
                        continue
                    transform_type = next(iter(transformation.keys()))
                    transform_folder = self.viewer.server.gui.add_folder(transform_type)
                    self.viewer.convert_gui_handles.append(transform_folder)

                    self._build_transform_folder(transform_folder, operation, transformation, transform_type)

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

    def disable_load_buttons(self):
        self.enable_scene_loading = False
        for button in self.viewer.load_buttons:
            button.disabled = True

    def enable_load_buttons(self):
        self.enable_scene_loading = True
        for button in self.viewer.load_buttons:
            button.disabled = False
        self.viewer.load_buttons[self.current_scene].disabled = True

    def disable_convert_ui(self):
        self.viewer._convert_button.disabled = True
        self.viewer._live_preview_checkbox.disabled = True

    def enable_convert_ui(self):
        self.viewer._convert_button.disabled = False
        self.viewer._live_preview_checkbox.disabled = False

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
            self.runner.update_live_preview(copy.deepcopy(self.encoding_params))

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
            self.runner.update_live_preview(copy.deepcopy(self.encoding_params))

        return update_field

    def live_preview_callback(self, _):
        """callback for live preview checkbox"""
        self.live_preview_active = self.viewer._live_preview_checkbox.value
        if not self.live_preview_active:
            self.viewer._convert_button.label = "Convert"
        else:
            self.viewer._convert_button.label = "Save to scene list"
        self.runner.update_live_preview(copy.deepcopy(self.encoding_params))

    def convert_button_callback(self, _):
        if self.live_preview_active:
            self.runner.save_live_preview()
        else:
            self.runner.run_conversion(copy.deepcopy(self.encoding_params))

    def update_eval_progress(self, value: int):
        self.viewer.eval_progress.value = 0


class Runner:
    """
    This class handles all evaluations and conversions from the InteractiveConversionTool.
    It exposes three main functionalities: update_live_preview, save_live_preview, run_conversion and run_evaluation.
    These functions are executed in a thread pool.
    """

    def __init__(self, conv_tool: InteractiveConversionTool) -> None:
        self.conv_tool: InteractiveConversionTool = conv_tool
        self.conversion_queue: list[EncodingParams] = []
        self.live_conversion_running: bool = False
        self.preview_in_scenes: bool = False

        # locks access to the conversion_queue
        self.queue_lock = threading.Lock()

        # ensures that only one process is currently running
        self.process_lock = threading.Lock()

    def save_live_preview(self):
        """Save live preview to scenes."""

        if not self.conv_tool.live_preview_active:
            return

        with self.process_lock:
            # copy results from live preview to saved scene directory
            output_path = Path(self.conv_tool.temp_dir.name + f"/gaussians{len(self.conv_tool.scenes)}")
            preview_path = Path(self.conv_tool.temp_dir.name + "/livepreview")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            shutil.copytree(preview_path, output_path, dirs_exist_ok=True)

            self.conv_tool.scenes[-1].data_path = output_path
            self.conv_tool.scenes[-1].description = self._build_description(self.conv_tool.encoding_params)
            self.conv_tool.viewer.last_scene_description.content = self.conv_tool.scenes[-1].description

            self.conv_tool.add_scene(
                "Live preview from conversion", preview_path, "smurfx", self.conv_tool.encoding_params
            )

            self.conv_tool.scenes[-1].scene_metrics = self.conv_tool.scenes[-2].scene_metrics
            self.conv_tool.update_eval_table()

    def update_live_preview(self, new_encoding_params: EncodingParams):
        """Adds task to the conversion_queue. If no live preview update is running, it starts updating the preview, otherwise it returns."""

        if not self.conv_tool.live_preview_active:
            if self.preview_in_scenes:
                self.preview_in_scenes = False
                self.conv_tool.remove_last_scene()
                self.conv_tool.current_scene = len(self.conv_tool.scenes) - 1
                self.conv_tool.load_scene(len(self.conv_tool.scenes) - 1)
            return

        with self.queue_lock:
            self.conversion_queue.append(new_encoding_params)

            # we only want to work on one conversion simultaneously
            if self.live_conversion_running:
                return
            else:
                self.live_conversion_running = True
                self.conv_tool.disable_convert_ui()

        with self.process_lock:
            while True:
                with self.queue_lock:
                    if not self.conversion_queue:
                        self.conv_tool.enable_convert_ui()
                        self.live_conversion_running = False
                        break
                    encoding_params = self.conversion_queue.pop()
                    self.conversion_queue.clear()
                self.conv_tool.viewer.scene_label.content = "Running conversion for live preview..."
                # clear previous live preview
                output_path = Path(self.conv_tool.temp_dir.name + "/livepreview")
                if output_path.exists() and output_path.is_dir():
                    shutil.rmtree(output_path)
                if self.preview_in_scenes:
                    self.conv_tool.remove_last_scene()
                else:
                    self.preview_in_scenes = True

                try:
                    self._convert(encoding_params, output_path)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    self.conv_tool.viewer.scene_label.content = "Conversion for live preview failed."

    def run_conversion(self, new_encoding_params: EncodingParams):
        with self.process_lock:
            self.conv_tool.disable_convert_ui()
            output_path = Path(self.conv_tool.temp_dir.name + f"/gaussians{len(self.conv_tool.scenes)}")
            try:
                self._convert(new_encoding_params, output_path)
            except Exception as e:
                self.conv_tool.enable_convert_ui()
                print(f"Exception occurred: {e}")
                return
            self.conv_tool.enable_convert_ui()

    # Do we want to modify this, so that the evaluation runs in the background and does not load every scene?
    def run_evaluation(self):
        with self.process_lock:
            for scene in self.conv_tool.scenes:
                if scene.scene_metrics is None:
                    self.conv_tool.viewer.eval_info.content = f"Running evaluation for scene {scene.id}..."
                    self.conv_tool.update_eval_progress(0)
                    self.conv_tool.load_scene(scene.id)
                    self._eval(scene.id)
                self.conv_tool.update_eval_table()

    def _convert(self, encoding_params: EncodingParams, output_path: Path):
        print("Converting scene...")
        encoder = SceneEncoder(
            encoding_params=encoding_params,
            output_path=output_path,
            fields=self.conv_tool.input_gaussians.to_field_dict(),
            decoding_params=DecodingParams.default_ffsplat_packer(
                profile=encoding_params.profile,
                profile_version=encoding_params.profile_version,
                scene=encoding_params.scene,
            ),
        )
        encoder.encode(verbose=self.conv_tool.verbose)

        # add scene to scene list and load it to view the scene
        if not self.conv_tool.live_preview_active:
            self.conv_tool.add_scene(self._build_description(encoding_params), output_path, "smurfx", encoding_params)
        else:
            self.conv_tool.add_scene("Live preview from conversion", output_path, "smurfx", encoding_params)
        self.conv_tool.viewer.rerender(None)

    def _eval(self, scene_id):
        print(f"Running evaluation for scene {scene_id}")
        device = "cuda"
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

        valloader = torch.utils.data.DataLoader(self.conv_tool.dataset, batch_size=1, shuffle=False, num_workers=1)
        elapsed_time = 0
        metrics = defaultdict(list)

        imgs_path = Path(self.conv_tool.viewer.results_path_input.value) / Path(f"scene_{scene_id}/imgs")
        if self.conv_tool.live_preview_active and scene_id == len(self.conv_tool.scenes) - 1:
            imgs_path = None

        for i, data in enumerate(valloader):
            elapsed_time += eval_step(
                self.conv_tool.gaussians,
                self.conv_tool.dataset.white_background,
                imgs_path,
                device,
                i,
                metrics,
                psnr,
                ssim,
                lpips,
                data,
            )
            self.conv_tool.viewer.eval_progress.value = (i + 1) / len(valloader) * 100

        elapsed_time /= len(valloader)

        size = get_directory_size(self.conv_tool.scenes[scene_id].data_path)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update({
            "elapsed_time": elapsed_time,
            "num_GS": len(self.conv_tool.gaussians.means.data),
            "size": size,
        })
        self.conv_tool.scenes[scene_id].scene_metrics = stats

    def _build_description(self, encoding_params: EncodingParams) -> str:
        output_format = self.conv_tool.viewer._output_dropdown.value

        description = "**Template**  \n"
        description += f"{output_format}  \n"
        changed_params_desc = ""

        initial_params: EncodingParams = EncodingParams.from_yaml_file(
            Path(f"src/ffsplat/conf/format/{output_format}.yaml")
        )
        for op_id, op in enumerate(encoding_params.ops):
            for transform_id, transform in enumerate(op["transforms"]):
                if transform != initial_params.ops[op_id]["transforms"][transform_id]:
                    if isinstance(op["input_fields"], list):
                        changed_params_desc += "```\ninput fields:\n" + yaml.dump(op["input_fields"])
                    else:
                        changed_params_desc += (
                            f"```\ninput fields from prefix: {op["input_fields"]["from_fields_with_prefix"]}\n"
                        )
                    changed_params_desc += yaml.dump(transform, default_flow_style=False) + "```  \n"

        if changed_params_desc != "":
            description += "**Customized Transformations**  \n"
            description += changed_params_desc

        return description


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
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting viewer...")
            break


if __name__ == "__main__":
    main()
