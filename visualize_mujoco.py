#!/usr/bin/env python3

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ACTUATOR_KEYS = [
    ("shoulder_pan", "1", "Q"),
    ("shoulder_lift", "2", "W"),
    ("elbow_flex", "3", "E"),
    ("wrist_flex", "4", "R"),
    ("wrist_roll", "5", "T"),
    ("gripper", "6", "Y"),
]

STEP_SIZE_RAD = 0.08
BODY_KEYS = ["cube", "target_bin"]
KEY_SPACE = 32
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51
KEY_4 = 52
KEY_5 = 53
KEY_6 = 54
KEY_E = 69
KEY_O = 79
KEY_P = 80
KEY_Q = 81
KEY_R = 82
KEY_T = 84
KEY_W = 87
KEY_Y = 89

KEY_NAME_TO_CODE = {
    "SPACE": KEY_SPACE,
    "1": KEY_1,
    "2": KEY_2,
    "3": KEY_3,
    "4": KEY_4,
    "5": KEY_5,
    "6": KEY_6,
    "E": KEY_E,
    "O": KEY_O,
    "P": KEY_P,
    "Q": KEY_Q,
    "R": KEY_R,
    "T": KEY_T,
    "W": KEY_W,
    "Y": KEY_Y,
}


def maybe_reexec_with_mjpython(headless: bool, render_video: bool) -> None:
    if headless or render_video or platform.system() != "Darwin":
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize SO101 MuJoCo models from the local so101_model folder.")
    parser.add_argument(
        "--model",
        choices=("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old"),
        default="scene",
        help="Model entrypoint to load.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run a short non-visual smoke test and print the final joint state.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of physics steps for headless or offscreen rendering.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use a passive viewer loop with keyboard joint control.",
    )
    parser.add_argument(
        "--render-video",
        type=Path,
        default=None,
        help="Render an offscreen MP4 instead of opening a GUI window.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        help="Camera name used for offscreen rendering.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Offscreen render width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Offscreen render height.",
    )
    return parser


def resolve_model_path(model_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / f"{model_name}.xml"


def actuator_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for idx in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        names.append(name or f"actuator_{idx}")
    return names


def print_help() -> None:
    print("SO101 MuJoCo controls")
    print("  Space: pause/resume")
    print("  R: reset pose")
    print("  P: print current joint targets")
    print("  O: print cube/bin/gripper positions (task scenes)")
    for name, inc, dec in ACTUATOR_KEYS:
        print(f"  {inc}/{dec}: increase/decrease {name}")


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)

    print("Headless run complete.")
    for idx, name in enumerate(actuator_names(model)):
        print(f"{name}: qpos={data.qpos[idx]: .4f} ctrl={data.ctrl[idx]: .4f}")


def scripted_targets(model: mujoco.MjModel, step_idx: int) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=float)
    phase = step_idx / 60.0
    if model.nu > 0:
        ctrl[0] = 0.35 * np.sin(phase * 0.4)
    if model.nu > 1:
        ctrl[1] = -0.55 + 0.20 * np.sin(phase * 0.9)
    if model.nu > 2:
        ctrl[2] = -1.05 + 0.28 * np.sin(phase * 0.7 + 0.8)
    if model.nu > 3:
        ctrl[3] = 0.10 + 0.20 * np.sin(phase * 1.1 + 1.4)
    if model.nu > 4:
        ctrl[4] = 0.40 * np.sin(phase * 0.6)
    if model.nu > 5:
        ctrl[5] = 0.25 + 0.65 * (0.5 + 0.5 * np.sin(phase * 1.3))
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    steps: int,
    camera_name: str,
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        raise ValueError(f"Unknown camera '{camera_name}'.")

    frames = []
    for step_idx in range(steps):
        data.ctrl[:] = scripted_targets(model, step_idx)
        mujoco.mj_step(model, data)
        if step_idx % 2 == 0:
            renderer.update_scene(data, camera=camera_name)
            frames.append(renderer.render().copy())

    imageio.mimwrite(output_path, frames, fps=max(1, int(0.5 / model.opt.timestep)))
    renderer.close()
    print(f"Saved video to {output_path}")


def configure_camera(model: mujoco.MjModel, camera_name: str, cam: mujoco.MjvCamera) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id != -1:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camera_id
        return

    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.4
    cam.azimuth = 145
    cam.elevation = -24
    cam.lookat[:] = np.array([0.18, 0.0, 0.18])


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, interactive: bool, camera_name: str) -> None:
    import glfw

    act_names = actuator_names(model)
    ctrl_range = model.actuator_ctrlrange.copy()
    targets = data.ctrl.copy()
    paused = False

    keymap: dict[int, tuple[int, float]] = {}
    for idx, (_name, inc_key, dec_key) in enumerate(ACTUATOR_KEYS[: model.nu]):
        keymap[KEY_NAME_TO_CODE[inc_key]] = (idx, STEP_SIZE_RAD)
        keymap[KEY_NAME_TO_CODE[dec_key]] = (idx, -STEP_SIZE_RAD)

    home_qpos = data.qpos.copy()
    home_ctrl = data.ctrl.copy()
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in BODY_KEYS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) != -1
    }
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = None

    def show_targets() -> None:
        print("")
        for idx, name in enumerate(act_names):
            print(f"{name:>14}: qpos={data.qpos[idx]: .4f} ctrl={targets[idx]: .4f}")
        print("")

    def show_task_state() -> None:
        print("")
        if site_id != -1:
            grip = data.site_xpos[site_id]
            print(f"{'gripperframe':>14}: x={grip[0]: .3f} y={grip[1]: .3f} z={grip[2]: .3f}")
        for name, body_id in body_ids.items():
            pos = data.xpos[body_id]
            print(f"{name:>14}: x={pos[0]: .3f} y={pos[1]: .3f} z={pos[2]: .3f}")
        print("")

    def key_callback(keycode: int) -> None:
        nonlocal paused, targets

        if keycode == KEY_SPACE:
            paused = not paused
            print("paused" if paused else "running")
            return

        if keycode == KEY_R:
            mujoco.mj_resetData(model, data)
            data.qpos[:] = home_qpos
            targets[:] = home_ctrl
            data.ctrl[:] = home_ctrl
            mujoco.mj_forward(model, data)
            print("reset to home pose")
            return

        if keycode == KEY_P:
            show_targets()
            return

        if keycode == KEY_O:
            show_task_state()
            return

        if keycode in keymap:
            actuator_idx, delta = keymap[keycode]
            low, high = ctrl_range[actuator_idx]
            targets[actuator_idx] = float(min(max(targets[actuator_idx] + delta, low), high))
            print(f"{act_names[actuator_idx]} target -> {targets[actuator_idx]:.4f}")

    def on_key(_window: object, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
                return
            key_callback(key)

    if not glfw.init():
        raise RuntimeError(
            "GLFW initialization failed. If Homebrew installed GLFW in /usr/local/lib, "
            "run with DYLD_LIBRARY_PATH=/usr/local/lib ./run_mujoco.sh ..."
        )

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(1280, 960, f"SO101 MuJoCo - {model.__class__.__name__}", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, on_key)

    mujoco.mjv_defaultOption(opt)
    configure_camera(model, camera_name, cam)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    if interactive:
        print_help()
    else:
        print("Viewer controls: Esc to close window")

    try:
        while not glfw.window_should_close(window):
            if not paused:
                data.ctrl[:] = targets
                mujoco.mj_step(model, data)

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(model.opt.timestep)
    finally:
        if context is not None:
            context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    args = build_parser().parse_args()
    maybe_reexec_with_mjpython(args.headless, args.render_video is not None)

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    if args.headless:
        run_headless(model, data, args.steps)
    elif args.render_video is not None:
        render_video(model, data, args.render_video, args.steps, args.camera, args.width, args.height)
    else:
        run_viewer(model, data, interactive=args.interactive, camera_name=args.camera)


if __name__ == "__main__":
    main()
