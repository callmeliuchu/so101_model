#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from official_glfw_viewer import OfficialViewer, resolve_model_path

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
BODY_JOINTS = JOINT_ORDER[:-1]
MODEL_CHOICES = ("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the SO101 MuJoCo model with LeRobot-style action dictionaries."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="task_scene",
        help="Model entrypoint to load.",
    )
    parser.add_argument(
        "--body-mode",
        choices=("degrees", "radians", "range_m100_100"),
        default="degrees",
        help="Input/output units for the five arm joints. Gripper stays in 0..100.",
    )
    parser.add_argument(
        "--apply-mode",
        choices=("qpos", "actuator"),
        default="qpos",
        help="Use direct joint mirroring ('qpos') or MuJoCo position actuators ('actuator').",
    )
    parser.add_argument(
        "--action-json",
        default=None,
        help='One LeRobot-style action dict, for example \'{"shoulder_pan.pos": 10, "gripper.pos": 70}\'.',
    )
    parser.add_argument(
        "--action-file",
        type=Path,
        default=None,
        help="Path to a JSON file containing one LeRobot-style action dict.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="JOINT.POS=VALUE",
        help="Set one action field from the command line. May be repeated.",
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=None,
        help="Path to a JSON list of actions or {action, steps} objects.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Physics steps to run after each command in non-viewer mode.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the GLFW viewer while applying commands.",
    )
    parser.add_argument(
        "--camera",
        default="overview",
        help="Named camera for viewer mode.",
    )
    parser.add_argument(
        "--close-when-done",
        action="store_true",
        help="In viewer mode, close automatically after the sequence finishes.",
    )
    parser.add_argument(
        "--print-observation",
        action="store_true",
        help="Print the observation dict after the last command.",
    )
    parser.add_argument(
        "--print-sent-action",
        action="store_true",
        help="Print the clipped action actually sent to the MuJoCo actuators.",
    )
    return parser


def parse_action_pairs(items: list[str]) -> dict[str, float]:
    action: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected JOINT.POS=VALUE.")
        key, value = item.split("=", 1)
        action[key.strip()] = float(value.strip())
    return action


def load_action_dict(args: argparse.Namespace) -> dict[str, float]:
    action: dict[str, float] = {}
    if args.action_json:
        action.update(json.loads(args.action_json))
    if args.action_file:
        action.update(json.loads(args.action_file.read_text()))
    action.update(parse_action_pairs(args.set))
    return action


def load_sequence(path: Path, default_steps: int) -> list[tuple[dict[str, float], int]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Sequence file must contain a JSON list: {path}")

    sequence: list[tuple[dict[str, float], int]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict) and "action" in item:
            action = item["action"]
            steps = int(item.get("steps", default_steps))
        else:
            action = item
            steps = default_steps
        if not isinstance(action, dict):
            raise ValueError(f"Sequence entry {idx} must be a dict or an object with an 'action' field.")
        sequence.append((action, steps))
    return sequence


class SO101SimRobot:
    def __init__(self, model_name: str) -> None:
        model_path = resolve_model_path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.actuator_ids = self._resolve_actuator_ids()
        self.joint_ids = self._resolve_joint_ids()
        self.targets = self.data.ctrl.copy()
        self.pinned_qpos = {
            name: self._joint_position(name)
            for name in JOINT_ORDER
        }
        mujoco.mj_forward(self.model, self.data)

    def _resolve_actuator_ids(self) -> dict[str, int]:
        actuator_ids: dict[str, int] = {}
        for name in JOINT_ORDER:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id == -1:
                raise ValueError(f"Actuator '{name}' not found in model.")
            actuator_ids[name] = actuator_id
        return actuator_ids

    def _resolve_joint_ids(self) -> dict[str, int]:
        joint_ids: dict[str, int] = {}
        for name in JOINT_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found in model.")
            joint_ids[name] = joint_id
        return joint_ids

    def _ctrl_range(self, name: str) -> tuple[float, float]:
        actuator_id = self.actuator_ids[name]
        low, high = self.model.actuator_ctrlrange[actuator_id]
        return float(low), float(high)

    def _joint_range(self, name: str) -> tuple[float, float]:
        joint_id = self.joint_ids[name]
        low, high = self.model.jnt_range[joint_id]
        return float(low), float(high)

    def _joint_position(self, name: str) -> float:
        joint_id = self.joint_ids[name]
        qpos_adr = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[qpos_adr])

    def _joint_velocity_address(self, name: str) -> int:
        joint_id = self.joint_ids[name]
        return int(self.model.jnt_dofadr[joint_id])

    def _clip_ctrl(self, name: str, value: float) -> float:
        low, high = self._ctrl_range(name)
        return float(np.clip(value, low, high))

    def _clip_joint(self, name: str, value: float) -> float:
        low, high = self._joint_range(name)
        return float(np.clip(value, low, high))

    def _body_to_ctrl(self, name: str, value: float, body_mode: str) -> float:
        low, high = self._ctrl_range(name)
        if body_mode == "degrees":
            ctrl = np.deg2rad(value)
        elif body_mode == "radians":
            ctrl = float(value)
        else:
            clipped = float(np.clip(value, -100.0, 100.0))
            alpha = (clipped + 100.0) / 200.0
            ctrl = low + alpha * (high - low)
        return self._clip_ctrl(name, ctrl)

    def _gripper_to_ctrl(self, value: float) -> float:
        low, high = self._ctrl_range("gripper")
        clipped = float(np.clip(value, 0.0, 100.0))
        alpha = clipped / 100.0
        return self._clip_ctrl("gripper", low + alpha * (high - low))

    def _ctrl_to_body(self, name: str, ctrl: float, body_mode: str) -> float:
        low, high = self._ctrl_range(name)
        if body_mode == "degrees":
            return float(np.rad2deg(ctrl))
        if body_mode == "radians":
            return float(ctrl)
        if high <= low:
            return 0.0
        alpha = (ctrl - low) / (high - low)
        return float(-100.0 + 200.0 * alpha)

    def _ctrl_to_gripper(self, ctrl: float) -> float:
        low, high = self._ctrl_range("gripper")
        if high <= low:
            return 0.0
        alpha = (ctrl - low) / (high - low)
        return float(100.0 * alpha)

    def action_limits(self, body_mode: str) -> dict[str, tuple[float, float]]:
        limits: dict[str, tuple[float, float]] = {}
        for name in BODY_JOINTS:
            low, high = self._ctrl_range(name)
            if body_mode == "degrees":
                limits[f"{name}.pos"] = (float(np.rad2deg(low)), float(np.rad2deg(high)))
            elif body_mode == "radians":
                limits[f"{name}.pos"] = (low, high)
            else:
                limits[f"{name}.pos"] = (-100.0, 100.0)
        limits["gripper.pos"] = (0.0, 100.0)
        return limits

    def clamp_action(self, action: dict[str, float], body_mode: str) -> dict[str, float]:
        limits = self.action_limits(body_mode)
        clipped: dict[str, float] = {}
        for key, value in action.items():
            if key not in limits:
                continue
            low, high = limits[key]
            clipped[key] = float(np.clip(value, low, high))
        return clipped

    def _pin_joint_position(self, name: str, position: float) -> None:
        joint_id = self.joint_ids[name]
        qpos_adr = self.model.jnt_qposadr[joint_id]
        dof_adr = self._joint_velocity_address(name)
        clipped = self._clip_joint(name, position)
        self.data.qpos[qpos_adr] = clipped
        self.data.qvel[dof_adr] = 0.0
        self.targets[self.actuator_ids[name]] = clipped
        self.pinned_qpos[name] = clipped

    def send_action(self, action: dict[str, float], body_mode: str, apply_mode: str) -> dict[str, float]:
        sent: dict[str, float] = {}
        for key, raw_value in action.items():
            if not key.endswith(".pos"):
                raise ValueError(f"Unsupported action key '{key}'. Expected '*.pos'.")
            name = key.removesuffix(".pos")
            if name not in self.actuator_ids:
                raise ValueError(f"Unknown joint '{name}'.")

            if name == "gripper":
                ctrl = self._gripper_to_ctrl(float(raw_value))
                sent[key] = self._ctrl_to_gripper(ctrl)
            else:
                ctrl = self._body_to_ctrl(name, float(raw_value), body_mode)
                sent[key] = self._ctrl_to_body(name, ctrl, body_mode)

            self.targets[self.actuator_ids[name]] = ctrl
            if apply_mode == "qpos":
                self._pin_joint_position(name, ctrl)

        self.data.ctrl[:] = self.targets
        if apply_mode == "qpos":
            mujoco.mj_forward(self.model, self.data)
        return sent

    def step(self, steps: int, apply_mode: str) -> None:
        if apply_mode == "actuator":
            for _ in range(max(0, steps)):
                self.data.ctrl[:] = self.targets
                mujoco.mj_step(self.model, self.data)
            return

        for _ in range(max(0, steps)):
            for name, position in self.pinned_qpos.items():
                self._pin_joint_position(name, position)
            self.data.ctrl[:] = self.targets
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        for name, position in self.pinned_qpos.items():
            self._pin_joint_position(name, position)
        mujoco.mj_forward(self.model, self.data)

    def get_observation(self, body_mode: str) -> dict[str, float]:
        obs: dict[str, float] = {}
        for name in BODY_JOINTS:
            pos = self._joint_position(name)
            obs[f"{name}.pos"] = self._ctrl_to_body(name, pos, body_mode)
        obs["gripper.pos"] = self._ctrl_to_gripper(self._joint_position("gripper"))
        return obs


def build_command_sequence(args: argparse.Namespace) -> list[tuple[dict[str, float], int]]:
    if args.sequence_file:
        return load_sequence(args.sequence_file, args.steps)

    action = load_action_dict(args)
    if not action:
        raise ValueError("No action provided. Use --set, --action-json, --action-file, or --sequence-file.")
    return [(action, args.steps)]


def print_json(label: str, payload: dict[str, float]) -> None:
    print(label)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_non_viewer(
    robot: SO101SimRobot,
    sequence: list[tuple[dict[str, float], int]],
    body_mode: str,
    apply_mode: str,
    print_sent_action: bool,
    print_observation: bool,
) -> None:
    last_sent: dict[str, float] = {}
    for idx, (action, steps) in enumerate(sequence):
        last_sent = robot.send_action(action, body_mode, apply_mode)
        robot.step(steps, apply_mode)
        if print_sent_action:
            print_json(f"sent_action[{idx}]", last_sent)

    if print_observation:
        print_json("observation", robot.get_observation(body_mode))


def run_viewer(
    robot: SO101SimRobot,
    sequence: list[tuple[dict[str, float], int]],
    body_mode: str,
    apply_mode: str,
    camera_name: str,
    close_when_done: bool,
    print_sent_action: bool,
    print_observation: bool,
) -> None:
    viewer = OfficialViewer(
        model=robot.model,
        data=robot.data,
        title="SO101 LeRobot Command Viewer",
        camera_name=camera_name,
        hidden=False,
    )
    viewer.initialize(paused=False)

    command_index = 0
    steps_left = 0

    try:
        while viewer.window is None or viewer.context is None:
            time.sleep(0.01)

        while not glfw.window_should_close(viewer.window):
            if command_index < len(sequence) and steps_left <= 0:
                action, steps_left = sequence[command_index]
                sent_action = robot.send_action(action, body_mode, apply_mode)
                if print_sent_action:
                    print_json(f"sent_action[{command_index}]", sent_action)
                command_index += 1

            if not viewer.paused:
                robot.step(1, apply_mode)
                if steps_left > 0:
                    steps_left -= 1

            width, height = glfw.get_framebuffer_size(viewer.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                robot.model,
                robot.data,
                viewer.opt,
                None,
                viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                viewer.scene,
            )
            mujoco.mjr_render(viewport, viewer.scene, viewer.context)
            glfw.swap_buffers(viewer.window)
            glfw.poll_events()

            if close_when_done and command_index >= len(sequence) and steps_left <= 0:
                break

            time.sleep(1.0 / 120.0)
    finally:
        viewer.close()

    if print_observation:
        print_json("observation", robot.get_observation(body_mode))


def main() -> None:
    args = build_parser().parse_args()
    sequence = build_command_sequence(args)
    robot = SO101SimRobot(args.model)

    if args.viewer:
        run_viewer(
            robot=robot,
            sequence=sequence,
            body_mode=args.body_mode,
            apply_mode=args.apply_mode,
            camera_name=args.camera,
            close_when_done=args.close_when_done,
            print_sent_action=args.print_sent_action,
            print_observation=args.print_observation,
        )
    else:
        run_non_viewer(
            robot=robot,
            sequence=sequence,
            body_mode=args.body_mode,
            apply_mode=args.apply_mode,
            print_sent_action=args.print_sent_action,
            print_observation=args.print_observation,
        )


if __name__ == "__main__":
    main()
