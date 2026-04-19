#!/usr/bin/env python3

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


DEFAULT_LOOKAT = np.array([0.18, 0.0, 0.18], dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a static snapshot from a local SO101 MuJoCo XML file.")
    parser.add_argument(
        "--model",
        choices=("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old"),
        default="task_scene",
        help="Model entrypoint to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/snapshot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        help="Named camera to use. Falls back to a free camera if the name does not exist.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Render width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="Render height.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.4,
        help="Free camera distance used when the named camera is missing.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=145.0,
        help="Free camera azimuth used when the named camera is missing.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-24.0,
        help="Free camera elevation used when the named camera is missing.",
    )
    return parser


def resolve_model_path(model_name: str) -> Path:
    return Path(__file__).resolve().parent / f"{model_name}.xml"


def configure_camera(model: mujoco.MjModel, camera_name: str, distance: float, azimuth: float, elevation: float):
    named_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if named_camera_id != -1:
        return camera_name, f"named camera '{camera_name}'"

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = DEFAULT_LOOKAT
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera, "free camera fallback"


def main() -> None:
    args = build_parser().parse_args()
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera, camera_desc = configure_camera(model, args.camera, args.distance, args.azimuth, args.elevation)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    finally:
        renderer.close()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)

    print(f"Model: {model_path}")
    print(f"Camera: {camera_desc}")
    print(f"Saved snapshot to {output_path}")


if __name__ == "__main__":
    main()
