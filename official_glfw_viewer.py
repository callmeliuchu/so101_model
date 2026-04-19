#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo viewer based on the official visualization API and GLFW callbacks."
    )
    parser.add_argument(
        "--model",
        choices=("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old"),
        default="task_scene",
        help="Model entrypoint to load.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        help="Named camera to use. Falls back to a free camera when missing.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with simulation paused.",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="Create the GLFW window hidden. Useful for smoke tests.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Automatically close after N seconds.",
    )
    return parser


def resolve_model_path(model_name: str) -> Path:
    return Path(__file__).resolve().parent / f"{model_name}.xml"


class OfficialViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, title: str, camera_name: str, hidden: bool) -> None:
        self.model = model
        self.data = data
        self.title = title
        self.camera_name = camera_name
        self.hidden = hidden

        self.window: glfw._GLFWwindow | None = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context: mujoco.MjrContext | None = None

        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.paused = False

    def initialize(self, paused: bool) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE if self.hidden else glfw.TRUE)

        self.window = glfw.create_window(1400, 1000, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self._configure_camera()
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.paused = paused

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        print("Official-style MuJoCo viewer controls")
        print("  Left drag: rotate camera")
        print("  Right drag: pan camera")
        print("  Scroll / middle drag: zoom")
        print("  Shift + drag: horizontal move/rotate variant")
        print("  Space: pause/resume")
        print("  Backspace: reset simulation")
        print("  Esc: close window")

    def _configure_camera(self) -> None:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id != -1:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id
            return

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = np.array([0.18, 0.0, 0.18], dtype=float)
        self.cam.distance = 1.4
        self.cam.azimuth = 145.0
        self.cam.elevation = -24.0

    def _key_callback(self, window: glfw._GLFWwindow, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        if key == glfw.KEY_SPACE:
            self.paused = not self.paused
            print("paused" if self.paused else "running")
            return

        if key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            print("simulation reset")

    def _mouse_button_callback(self, window: glfw._GLFWwindow, _button: int, _action: int, _mods: int) -> None:
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        if not (self.button_left or self.button_middle or self.button_right):
            return

        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / max(1, height),
            dy / max(1, height),
            self.scene,
            self.cam,
        )

    def _scroll_callback(self, _window: glfw._GLFWwindow, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )

    def render_loop(self, duration: float | None = None) -> None:
        if self.window is None or self.context is None:
            raise RuntimeError("Viewer not initialized.")

        start = time.time()
        simstart = self.data.time

        while not glfw.window_should_close(self.window):
            if duration is not None and time.time() - start >= duration:
                break

            time_prev = self.data.time
            while not self.paused and self.data.time - time_prev < 1.0 / 60.0:
                mujoco.mj_step(self.model, self.data)

            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            if self.paused:
                time.sleep(1.0 / 60.0)

        _ = simstart

    def close(self) -> None:
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def main() -> None:
    args = build_parser().parse_args()
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    viewer = OfficialViewer(
        model=model,
        data=data,
        title=f"SO101 Official GLFW Viewer - {args.model}",
        camera_name=args.camera,
        hidden=args.hidden,
    )
    viewer.initialize(paused=args.paused)
    try:
        viewer.render_loop(duration=args.duration)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
