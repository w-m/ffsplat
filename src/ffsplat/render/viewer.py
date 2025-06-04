# This file is based on https://github.com/hangg7/nerfview


import dataclasses
import io
import sys
import time
from collections.abc import Callable
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8

from ._renderer import Renderer, RenderTask

available_output_format: list[str] = [
    "3DGS_INRIA_ply",
    "3DGS_INRIA_nosh_ply",
    "SOG-web",
    "SOG-web-nosh",
    "SOG-web-sh-split",
]


@dataclasses.dataclass
class CameraState:
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array([
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0],
        ])
        return K


@dataclasses.dataclass
class ViewerState:
    num_train_rays_per_sec: float | None = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal["rendering", "preparing", "training", "paused", "completed"] = "rendering"


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer:
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, tuple[int, int]],
            UInt8[np.ndarray, "H W 3"] | tuple[UInt8[np.ndarray, "H W 3"], Float32[np.ndarray, "H W"] | None],
        ],
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        if self.mode == "rendering":
            self.state.status = "rendering"
        self.convert_gui_handles: list = []
        self.load_buttons: list = []

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()

    def _define_guis(self):
        self.server.gui.configure_theme(control_width="large", control_layout="fixed")
        self.scene_label = self.server.gui.add_markdown("Showing scene: None")
        with self.server.gui.add_folder("Stats", visible=self.mode == "training") as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder("Training", visible=self.mode == "training") as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)
            self.render_quality_dropdown = self.server.gui.add_dropdown(
                "Render Format:", ["png", "jpeg"], initial_value="jpeg"
            )
            self.render_quality_dropdown.on_update(self.rerender)

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

    def add_scenes(self, results_path: Path):
        self.scenes_tab = self.tab_group.add_tab("Scenes")
        with self.scenes_tab:
            if results_path is None:
                results_path = Path("./results")
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

    def add_convert(self, reset_dynamic_params_gui_fn: Callable, convert_fn: Callable):
        with self.tab_group.add_tab("Convert") as self.convert_tab:
            self._output_dropdown = self.server.gui.add_dropdown("Output format", available_output_format)
            self._output_dropdown.on_update(reset_dynamic_params_gui_fn)
            self._convert_button = self.server.gui.add_button("Convert")
            self._convert_button.on_click(convert_fn)
        reset_dynamic_params_gui_fn(None)

    def add_to_scene_tab(
        self, scene_id: int, description: str, load_fn: Callable, save_fn: Callable, save_params_fn: Callable
    ):
        with self.scenes_tab, self.server.gui.add_folder(f"Scene {scene_id}"):
            self.server.gui.add_markdown(description)
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

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            if camera_state is None:
                raise ValueError()
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(viewer=self, client=client, lock=self.lock)
        self._renderers[client_id].start()
        client.camera.up_direction = np.array([0, -1, 0])

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate([vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            "`update` method is only available in training mode."
            raise ValueError()
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            if self.state.num_train_rays_per_sec is None:
                raise ValueError()

            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = train_util * view_time / (train_time - train_util * train_time)
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    if camera_state is None:
                        raise ValueError()
                    self._renderers[client_id].submit(RenderTask("update", camera_state))
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
