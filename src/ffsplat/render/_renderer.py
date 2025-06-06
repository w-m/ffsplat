# This file is from https://github.com/hangg7/nerfview

import dataclasses
import os
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Literal, Optional, get_args

import viser

if TYPE_CHECKING:
    from .viewer import CameraState, Viewer

RenderState = Literal["low_move", "low_static", "high"]
RenderAction = Literal["rerender", "move", "static", "update"]


@dataclasses.dataclass
class RenderTask:
    action: RenderAction
    camera_state: Optional["CameraState"] = None


class InterruptRenderException(Exception):
    pass


class set_trace_context:
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, *_, **__):
        sys.settrace(None)


class Renderer(threading.Thread):
    """This class is responsible for rendering images in the background."""

    def __init__(
        self,
        viewer: "Viewer",
        client: viser.ClientHandle,
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)

        self.viewer = viewer
        self.client = client
        self.lock = lock

        self.running = True
        self.is_prepared_fn = lambda: self.viewer.state.status != "preparing"

        self._render_event = threading.Event()
        self._state: RenderState = "low_static"
        self._task: RenderTask | None = None

        self._target_fps = 30
        self._may_interrupt_render = False

        self._define_transitions()

    def _define_transitions(self):
        transitions: dict[RenderState, dict[RenderAction, RenderState]] = {
            s: {a: s for a in get_args(RenderAction)} for s in get_args(RenderState)
        }
        transitions["low_move"]["static"] = "low_static"
        transitions["low_static"]["static"] = "high"
        transitions["low_static"]["update"] = "high"
        transitions["low_static"]["move"] = "low_move"
        transitions["high"]["move"] = "low_move"
        transitions["high"]["rerender"] = "low_static"
        self.transitions = transitions

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line" and self._may_interrupt_render:
            self._may_interrupt_render = False
            raise InterruptRenderException
        return self._may_interrupt_trace

    def _get_img_wh(self, aspect: float) -> tuple[int, int]:
        max_img_res = self.viewer._max_img_res_slider.value
        if self._state == "high":
            #  if True:
            H = max_img_res
            W = int(H * aspect)
            if max_img_res < W:
                W = max_img_res
                H = int(W / aspect)
        elif self._state in ["low_move", "low_static"]:
            num_view_rays_per_sec = self.viewer.state.num_view_rays_per_sec
            target_fps = self._target_fps
            num_viewer_rays = num_view_rays_per_sec / target_fps
            H = (num_viewer_rays / aspect) ** 0.5
            H = int(round(H, -1))
            H = max(min(max_img_res, H), 30)
            W = int(H * aspect)
            if max_img_res < W:
                W = max_img_res
                H = int(W / aspect)
        else:
            "unkown state"
            raise ValueError()
        return W, H

    def submit(self, task: RenderTask):
        if self._task is None:
            self._task = task
        elif task.action == "update" and (self._state == "low_move" or self._task.action in ["move", "rerender"]):
            return
        else:
            self._task = task

        if self._state == "high" and self._task.action in ["move", "rerender"]:
            self._may_interrupt_render = True
        self._render_event.set()

    def run(self):
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if not self._render_event.wait(0.2):
                self.submit(RenderTask("static", self.viewer.get_camera_state(self.client)))
            self._render_event.clear()
            task = self._task
            if task is None:
                raise ValueError()
            #  print(self._state, task.action, self.transitions[self._state][task.action])
            if self._state == "high" and task.action == "static":
                continue
            self._state = self.transitions[self._state][task.action]
            if task.camera_state is None:
                raise ValueError
            try:
                with self.lock, set_trace_context(self._may_interrupt_trace):
                    tic = time.time()
                    W, H = img_wh = self._get_img_wh(task.camera_state.aspect)
                    rendered = self.viewer.render_fn(task.camera_state, img_wh)
                    if isinstance(rendered, tuple):
                        img, depth = rendered
                    else:
                        img, depth = rendered, None
                    self.viewer.state.num_view_rays_per_sec = (W * H) / (max(time.time() - tic, 1e-6))
            except InterruptRenderException:
                continue
            except Exception:
                traceback.print_exc()
                os._exit(1)

            self.client.scene.set_background_image(
                img,
                format=self.viewer.render_quality_dropdown.value,
                jpeg_quality=70 if task.action in ["static", "update"] else 40,
                depth=depth,
            )
