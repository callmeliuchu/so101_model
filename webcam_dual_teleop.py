#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import glfw
import mediapipe as mp
import mujoco
import numpy as np

from lerobot_command_control import SO101SimRobot
from official_glfw_viewer import OfficialViewer
from webcam_skeleton_teleop import (
    CartesianIKTeleopMapper,
    JointTeleopMapper,
    create_landmarkers,
    draw_overlay,
    draw_pose_points,
    extract_hand_metrics,
    extract_pose_points,
    open_camera,
)

MODEL_CHOICES = ("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive the SO101 sim, real arm, or both from webcam skeleton/hand tracking."
    )
    parser.add_argument("--target", choices=("sim", "real", "both"), default="sim", help="Control backend.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="scene", help="MuJoCo model entrypoint.")
    parser.add_argument("--camera", default="overview", help="Named MuJoCo camera for the sim viewer.")
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera device index.")
    parser.add_argument("--backend", choices=("auto", "avfoundation", "default"), default="auto")
    parser.add_argument("--arm-side", choices=("right", "left"), default="right")
    parser.add_argument("--mapping", choices=("ik", "joint"), default="ik")
    parser.add_argument("--apply-mode", choices=("qpos", "actuator"), default="qpos")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-sim-viewer", action="store_true")
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    parser.add_argument("--pan-gain", type=float, default=220.0)
    parser.add_argument("--lift-gain", type=float, default=220.0)
    parser.add_argument("--lateral-gain-m", type=float, default=0.32)
    parser.add_argument("--vertical-gain-m", type=float, default=0.24)
    parser.add_argument("--reach-gain-m", type=float, default=0.18)

    parser.add_argument("--lerobot-root", type=Path, default=Path("/Users/liuchu/codes/lerobot"))
    parser.add_argument(
        "--lerobot-python",
        default="python",
        help="Python executable for the real-robot bridge. Use the same env that can run lerobot-record.",
    )
    parser.add_argument("--robot-port", default=None, help="Real SO101 follower serial port.")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm", help="Real robot calibration id.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=8.0,
        help="LeRobot safety clamp for each real-robot action step in degrees / normalized units.",
    )
    parser.add_argument(
        "--start-real-enabled",
        action="store_true",
        help="Start with real robot command sending enabled. Default is disabled for safety.",
    )
    return parser


class RealRobotBridge:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process

    def send_action(self, action: dict[str, float]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Real robot bridge stdin is not available.")
        message = json.dumps({"type": "action", "action": action})
        self.process.stdin.write(message + "\n")
        self.process.stdin.flush()

    def disconnect(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"type": "close"}) + "\n")
                self.process.stdin.flush()
            except Exception:
                pass
            try:
                self.process.stdin.close()
            except Exception:
                pass
        self.process.wait(timeout=5)


def render_sim_view(viewer: OfficialViewer) -> None:
    if viewer.window is None or viewer.context is None:
        return
    width, height = glfw.get_framebuffer_size(viewer.window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(
        viewer.model,
        viewer.data,
        viewer.opt,
        None,
        viewer.cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        viewer.scene,
    )
    mujoco.mjr_render(viewport, viewer.scene, viewer.context)
    glfw.swap_buffers(viewer.window)
    glfw.poll_events()


def make_mapper(args: argparse.Namespace, sim_robot: SO101SimRobot):
    if args.mapping == "ik":
        return CartesianIKTeleopMapper(
            robot=sim_robot,
            arm_side=args.arm_side,
            smooth_alpha=args.smooth_alpha,
            lateral_gain_m=args.lateral_gain_m,
            vertical_gain_m=args.vertical_gain_m,
            reach_gain_m=args.reach_gain_m,
        )
    return JointTeleopMapper(
        arm_side=args.arm_side,
        pan_gain=args.pan_gain,
        lift_gain=args.lift_gain,
        smooth_alpha=args.smooth_alpha,
    )


def make_real_robot(args: argparse.Namespace):
    if args.target not in ("real", "both"):
        return None
    if not args.robot_port:
        raise ValueError("--robot-port is required when --target is real or both.")
    bridge_script = Path(__file__).resolve().parent / "lerobot_real_robot_bridge.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [
            args.lerobot_python,
            str(bridge_script),
            "--lerobot-root",
            str(args.lerobot_root),
            "--robot-port",
            args.robot_port,
            "--robot-id",
            args.robot_id,
            "--max-relative-target",
            str(args.max_relative_target),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to create real robot bridge stdout pipe.")
    ready_line = process.stdout.readline().strip()
    if ready_line != "bridge_ready":
        raise RuntimeError(
            "Real robot bridge failed to start. "
            f"Expected 'bridge_ready', got '{ready_line}'. "
            "Use --lerobot-python with the same Python env that can run lerobot-record."
        )
    return RealRobotBridge(process)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent

    sim_robot = SO101SimRobot(args.model)
    mapper = make_mapper(args, sim_robot)

    viewer = None
    if args.target in ("sim", "both") and not args.no_sim_viewer:
        viewer = OfficialViewer(
            model=sim_robot.model,
            data=sim_robot.data,
            title="SO101 Dual Webcam Teleop",
            camera_name=args.camera,
            hidden=False,
        )
        viewer.initialize(paused=False)

    real_robot = make_real_robot(args)
    real_enabled = args.start_real_enabled

    cap = open_camera(args.device, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {args.device}.")

    pose_landmarker, hand_landmarker = create_landmarkers(root)
    last_action = mapper.smoothed_action.copy()
    action_limits = sim_robot.action_limits("degrees")
    last_frame_time = time.perf_counter()
    gripper_site_id = mujoco.mj_name2id(sim_robot.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_h, frame_w = frame.shape[:2]
            pose_points = extract_pose_points(pose_result, args.arm_side, frame_w, frame_h)
            hand_metrics = extract_hand_metrics(hand_result, args.arm_side, frame_w, frame_h)
            draw_pose_points(frame, pose_points)

            action = mapper.action_from_points(pose_points, hand_metrics, frame_w, frame_h)
            if action is not None:
                clamped_action = sim_robot.clamp_action(action, body_mode="degrees")
                last_action = clamped_action
                if args.target in ("sim", "both"):
                    sim_robot.send_action(clamped_action, body_mode="degrees", apply_mode=args.apply_mode)
                if real_robot is not None and real_enabled:
                    real_robot.send_action(clamped_action)

            if args.target in ("sim", "both"):
                sim_robot.step(1, args.apply_mode)

            if viewer is not None:
                if viewer.window is not None and glfw.window_should_close(viewer.window):
                    break
                render_sim_view(viewer)

            status_lines = [
                f"target={args.target}",
                f"mapping={args.mapping}",
                f"real_enabled={real_enabled}",
                f"pose_detected={pose_points is not None}",
                f"hand_detected={hand_metrics is not None}",
                (
                    "pan_range="
                    f"{action_limits['shoulder_pan.pos'][0]:.0f}..{action_limits['shoulder_pan.pos'][1]:.0f}"
                ),
                "space: calibrate neutral pose",
                "e: toggle real robot send",
                "q or esc: quit",
            ]
            draw_overlay(
                frame,
                args.arm_side,
                args.mapping,
                last_action,
                mapper.neutral is not None,
                extra_lines=status_lines,
            )
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last_frame_time)
            last_frame_time = now
            cv2.putText(
                frame,
                f"fps={fps:.1f}",
                (16, frame_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SO101 Dual Webcam Teleop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                gripper_xyz = (
                    sim_robot.data.site_xpos[gripper_site_id].copy() if gripper_site_id != -1 else np.zeros(3, dtype=float)
                )
                success = mapper.capture_neutral(pose_points, hand_metrics, gripper_xyz)
                print("neutral captured" if success else "neutral capture failed")
            if key == ord("e") and real_robot is not None:
                real_enabled = not real_enabled
                print(f"real robot send {'enabled' if real_enabled else 'disabled'}")
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()
        if real_robot is not None:
            real_robot.disconnect()


if __name__ == "__main__":
    main()
