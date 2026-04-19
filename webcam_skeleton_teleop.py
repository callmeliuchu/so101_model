#!/usr/bin/env python3

from __future__ import annotations

import argparse
import platform
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import glfw
import mediapipe as mp
import mujoco
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from lerobot_command_control import SO101SimRobot
from official_glfw_viewer import OfficialViewer

MODEL_CHOICES = ("scene", "task_scene", "so101_new_calib", "so101_old_calib", "task_scene_old")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)
POSE_POINTS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
}
HAND_POINTS = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_tip": 8,
    "pinky_mcp": 17,
}


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def vector_angle_degrees(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def ensure_model_asset(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            output_path.write_bytes(response.read())
    return output_path


@dataclass
class NeutralPose:
    wrist_x: float
    wrist_y: float
    upper_arm_angle: float
    forearm_angle: float
    hand_roll_angle: float
    elbow_bend: float
    gripper_xyz: np.ndarray


class JointTeleopMapper:
    def __init__(self, arm_side: str, pan_gain: float, lift_gain: float, smooth_alpha: float) -> None:
        self.arm_side = arm_side
        self.pan_gain = pan_gain
        self.lift_gain = lift_gain
        self.smooth_alpha = smooth_alpha
        self.neutral: NeutralPose | None = None
        self.smoothed_action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -20.0,
            "elbow_flex.pos": 40.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 70.0,
        }

    def capture_neutral(
        self,
        pose_points: dict[str, np.ndarray] | None,
        hand_metrics: tuple[float, float] | None,
        gripper_xyz: np.ndarray | None,
    ) -> bool:
        if pose_points is None:
            return False

        upper_vec = pose_points["elbow"] - pose_points["shoulder"]
        forearm_vec = pose_points["wrist"] - pose_points["elbow"]
        elbow_bend = clamp(
            180.0 - angle_degrees(pose_points["shoulder"], pose_points["elbow"], pose_points["wrist"]),
            0.0,
            120.0,
        )
        hand_roll = hand_metrics[1] if hand_metrics is not None else 0.0

        self.neutral = NeutralPose(
            wrist_x=float(pose_points["wrist"][0]),
            wrist_y=float(pose_points["wrist"][1]),
            upper_arm_angle=vector_angle_degrees(upper_vec),
            forearm_angle=vector_angle_degrees(forearm_vec),
            hand_roll_angle=hand_roll,
            elbow_bend=elbow_bend,
            gripper_xyz=np.array(gripper_xyz if gripper_xyz is not None else np.zeros(3), dtype=float),
        )
        return True

    def action_from_points(
        self,
        pose_points: dict[str, np.ndarray] | None,
        hand_metrics: tuple[float, float] | None,
        image_width: int,
        image_height: int,
    ) -> dict[str, float] | None:
        if pose_points is None:
            return None

        upper_vec = pose_points["elbow"] - pose_points["shoulder"]
        forearm_vec = pose_points["wrist"] - pose_points["elbow"]
        elbow_angle = angle_degrees(pose_points["shoulder"], pose_points["elbow"], pose_points["wrist"])
        elbow_flex = clamp(180.0 - elbow_angle, 0.0, 120.0)

        if self.neutral is None:
            self.capture_neutral(pose_points, hand_metrics, None)

        assert self.neutral is not None

        dx = (float(pose_points["wrist"][0]) - self.neutral.wrist_x) / max(1.0, image_width)
        dy = (self.neutral.wrist_y - float(pose_points["wrist"][1])) / max(1.0, image_height)
        forearm_angle = vector_angle_degrees(forearm_vec)

        shoulder_pan = clamp(self.pan_gain * dx, -90.0, 90.0)
        shoulder_lift = clamp(-20.0 + self.lift_gain * dy, -95.0, 70.0)
        wrist_flex = clamp((forearm_angle - self.neutral.forearm_angle) * 0.8, -100.0, 100.0)

        if hand_metrics is not None:
            pinch_norm, hand_roll = hand_metrics
            gripper = clamp((pinch_norm - 0.18) / 0.32 * 100.0, 0.0, 100.0)
            wrist_roll = clamp((hand_roll - self.neutral.hand_roll_angle) * 1.5, -100.0, 100.0)
        else:
            gripper = self.smoothed_action["gripper.pos"]
            wrist_roll = self.smoothed_action["wrist_roll.pos"]

        raw_action = {
            "shoulder_pan.pos": shoulder_pan,
            "shoulder_lift.pos": shoulder_lift,
            "elbow_flex.pos": elbow_flex,
            "wrist_flex.pos": wrist_flex,
            "wrist_roll.pos": wrist_roll,
            "gripper.pos": gripper,
        }

        smoothed: dict[str, float] = {}
        alpha = self.smooth_alpha
        for key, value in raw_action.items():
            smoothed[key] = float(alpha * value + (1.0 - alpha) * self.smoothed_action[key])
        self.smoothed_action = smoothed
        return smoothed


class IKSolver:
    def __init__(self, robot: SO101SimRobot, site_name: str = "gripperframe") -> None:
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in model.")
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        self.dof_ids = [int(self.model.jnt_dofadr[robot.joint_ids[name]]) for name in self.joint_names]
        self.qpos_ids = [int(self.model.jnt_qposadr[robot.joint_ids[name]]) for name in self.joint_names]

    def current_xyz(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def solve(self, target_xyz: np.ndarray, wrist_roll_deg: float, gripper_0_100: float) -> dict[str, float]:
        q = np.array([self.robot._joint_position(name) for name in self.joint_names], dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        damping = 1e-3

        for _ in range(30):
            for idx, name in enumerate(self.joint_names):
                self.robot._pin_joint_position(name, q[idx])
            mujoco.mj_forward(self.model, self.data)

            current = self.data.site_xpos[self.site_id].copy()
            error = target_xyz - current
            if np.linalg.norm(error) < 2e-3:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            j = jacp[:, self.dof_ids]
            lhs = j @ j.T + damping * np.eye(3)
            dq = j.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq, -0.08, 0.08)
            q += dq

            for idx, name in enumerate(self.joint_names):
                q[idx] = self.robot._clip_joint(name, q[idx])

        action = {
            f"{name}.pos": float(np.rad2deg(q[idx]))
            for idx, name in enumerate(self.joint_names)
        }
        action["wrist_roll.pos"] = wrist_roll_deg
        action["gripper.pos"] = gripper_0_100
        return action


class CartesianIKTeleopMapper:
    def __init__(
        self,
        robot: SO101SimRobot,
        arm_side: str,
        smooth_alpha: float,
        lateral_gain_m: float,
        vertical_gain_m: float,
        reach_gain_m: float,
    ) -> None:
        self.robot = robot
        self.arm_side = arm_side
        self.smooth_alpha = smooth_alpha
        self.lateral_gain_m = lateral_gain_m
        self.vertical_gain_m = vertical_gain_m
        self.reach_gain_m = reach_gain_m
        self.ik = IKSolver(robot)
        self.neutral: NeutralPose | None = None
        self.smoothed_action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -20.0,
            "elbow_flex.pos": 40.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 70.0,
        }

    def capture_neutral(
        self,
        pose_points: dict[str, np.ndarray] | None,
        hand_metrics: tuple[float, float] | None,
        gripper_xyz: np.ndarray | None = None,
    ) -> bool:
        if pose_points is None:
            return False

        upper_vec = pose_points["elbow"] - pose_points["shoulder"]
        forearm_vec = pose_points["wrist"] - pose_points["elbow"]
        elbow_bend = clamp(
            180.0 - angle_degrees(pose_points["shoulder"], pose_points["elbow"], pose_points["wrist"]),
            0.0,
            120.0,
        )
        hand_roll = hand_metrics[1] if hand_metrics is not None else 0.0
        if gripper_xyz is None:
            gripper_xyz = self.ik.current_xyz()

        self.neutral = NeutralPose(
            wrist_x=float(pose_points["wrist"][0]),
            wrist_y=float(pose_points["wrist"][1]),
            upper_arm_angle=vector_angle_degrees(upper_vec),
            forearm_angle=vector_angle_degrees(forearm_vec),
            hand_roll_angle=hand_roll,
            elbow_bend=elbow_bend,
            gripper_xyz=np.array(gripper_xyz, dtype=float),
        )
        return True

    def action_from_points(
        self,
        pose_points: dict[str, np.ndarray] | None,
        hand_metrics: tuple[float, float] | None,
        image_width: int,
        image_height: int,
    ) -> dict[str, float] | None:
        if pose_points is None:
            return None

        elbow_bend = clamp(
            180.0 - angle_degrees(pose_points["shoulder"], pose_points["elbow"], pose_points["wrist"]),
            0.0,
            120.0,
        )
        if self.neutral is None:
            self.capture_neutral(pose_points, hand_metrics)

        assert self.neutral is not None

        dx = (float(pose_points["wrist"][0]) - self.neutral.wrist_x) / max(1.0, image_width)
        dy = (self.neutral.wrist_y - float(pose_points["wrist"][1])) / max(1.0, image_height)
        bend_delta = (self.neutral.elbow_bend - elbow_bend) / 100.0

        target = self.neutral.gripper_xyz.copy()
        side_sign = -1.0 if self.arm_side == "right" else 1.0
        target[1] += side_sign * dx * self.lateral_gain_m
        target[2] += dy * self.vertical_gain_m
        target[0] += bend_delta * self.reach_gain_m
        target[0] = clamp(target[0], 0.02, 0.42)
        target[1] = clamp(target[1], -0.28, 0.28)
        target[2] = clamp(target[2], 0.02, 0.35)

        if hand_metrics is not None:
            pinch_norm, hand_roll = hand_metrics
            gripper = clamp((pinch_norm - 0.18) / 0.32 * 100.0, 0.0, 100.0)
            wrist_roll = clamp((hand_roll - self.neutral.hand_roll_angle) * 1.5, -100.0, 100.0)
        else:
            gripper = self.smoothed_action["gripper.pos"]
            wrist_roll = self.smoothed_action["wrist_roll.pos"]

        raw_action = self.ik.solve(target, wrist_roll_deg=wrist_roll, gripper_0_100=gripper)

        smoothed: dict[str, float] = {}
        alpha = self.smooth_alpha
        for key, value in raw_action.items():
            smoothed[key] = float(alpha * value + (1.0 - alpha) * self.smoothed_action[key])
        self.smoothed_action = smoothed
        return smoothed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the SO101 MuJoCo model from webcam skeleton and hand landmarks."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="scene", help="Model entrypoint to load.")
    parser.add_argument("--camera", default="overview", help="Named MuJoCo camera for the sim viewer.")
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera device index.")
    parser.add_argument(
        "--backend",
        choices=("auto", "avfoundation", "default"),
        default="auto",
        help="OpenCV video backend. On macOS, 'auto' tries AVFoundation first.",
    )
    parser.add_argument("--arm-side", choices=("right", "left"), default="right", help="Which arm to track.")
    parser.add_argument(
        "--mapping",
        choices=("ik", "joint"),
        default="ik",
        help="Use end-effector IK mapping or the older direct-joint heuristic mapping.",
    )
    parser.add_argument("--apply-mode", choices=("qpos", "actuator"), default="qpos", help="Sim control mode.")
    parser.add_argument("--pan-gain", type=float, default=220.0, help="Horizontal wrist motion to shoulder pan gain.")
    parser.add_argument("--lift-gain", type=float, default=220.0, help="Vertical wrist motion to shoulder lift gain.")
    parser.add_argument("--lateral-gain-m", type=float, default=0.32, help="IK lateral workspace gain in meters.")
    parser.add_argument("--vertical-gain-m", type=float, default=0.24, help="IK vertical workspace gain in meters.")
    parser.add_argument("--reach-gain-m", type=float, default=0.18, help="IK forward/back reach gain in meters.")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA smoothing factor for actions.")
    parser.add_argument("--no-sim-viewer", action="store_true", help="Disable the MuJoCo viewer window.")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam image for more natural control.")
    return parser


def open_camera(device: int, backend: str) -> cv2.VideoCapture:
    candidates: list[int | None]
    if backend == "avfoundation":
        candidates = [cv2.CAP_AVFOUNDATION]
    elif backend == "default":
        candidates = [None]
    else:
        if platform.system() == "Darwin":
            candidates = [cv2.CAP_AVFOUNDATION, None]
        else:
            candidates = [None]

    last_capture = None
    for candidate in candidates:
        if candidate is None:
            capture = cv2.VideoCapture(device)
        else:
            capture = cv2.VideoCapture(device, candidate)
        if capture.isOpened():
            return capture
        last_capture = capture
        capture.release()

    if last_capture is None:
        last_capture = cv2.VideoCapture()
    return last_capture


def create_landmarkers(root: Path) -> tuple[vision.PoseLandmarker, vision.HandLandmarker]:
    pose_model_path = ensure_model_asset(POSE_MODEL_URL, root / ".models" / "pose_landmarker_full.task")
    hand_model_path = ensure_model_asset(HAND_MODEL_URL, root / ".models" / "hand_landmarker.task")

    running_mode = vision.RunningMode.VIDEO
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return pose, hands


def extract_pose_points(pose_result, arm_side: str, width: int, height: int) -> dict[str, np.ndarray] | None:
    if not pose_result.pose_landmarks:
        return None
    landmarks = pose_result.pose_landmarks[0]
    indices = POSE_POINTS[arm_side]

    points: dict[str, np.ndarray] = {}
    for name, index in indices.items():
        landmark = landmarks[index]
        if getattr(landmark, "visibility", 1.0) < 0.5:
            return None
        points[name] = np.array([landmark.x * width, landmark.y * height], dtype=float)
    return points


def extract_hand_metrics(hand_result, arm_side: str, width: int, height: int) -> tuple[float, float] | None:
    if not hand_result.hand_landmarks:
        return None

    for hand_index, landmarks in enumerate(hand_result.hand_landmarks):
        handedness = hand_result.handedness[hand_index][0].category_name.lower()
        if handedness != arm_side:
            continue

        def point(index: int) -> np.ndarray:
            landmark = landmarks[index]
            return np.array([landmark.x * width, landmark.y * height], dtype=float)

        thumb_tip = point(HAND_POINTS["thumb_tip"])
        index_tip = point(HAND_POINTS["index_tip"])
        index_mcp = point(HAND_POINTS["index_mcp"])
        pinky_mcp = point(HAND_POINTS["pinky_mcp"])
        wrist = point(HAND_POINTS["wrist"])

        pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
        hand_length = float(np.linalg.norm(index_mcp - wrist))
        hand_roll = vector_angle_degrees(index_mcp - pinky_mcp)
        if hand_length < 1e-6:
            hand_length = 1.0
        return pinch_distance / hand_length, hand_roll

    return None


def draw_pose_points(frame, pose_points: dict[str, np.ndarray] | None) -> None:
    if pose_points is None:
        return

    pairs = [("shoulder", "elbow"), ("elbow", "wrist")]
    for start_name, end_name in pairs:
        start = tuple(np.round(pose_points[start_name]).astype(int))
        end = tuple(np.round(pose_points[end_name]).astype(int))
        cv2.line(frame, start, end, (0, 220, 255), 3)

    for point in pose_points.values():
        x, y = tuple(np.round(point).astype(int))
        cv2.circle(frame, (x, y), 6, (50, 255, 50), -1)


def draw_overlay(frame, arm_side: str, mapping: str, action: dict[str, float] | None, neutral_ready: bool) -> None:
    lines = [
        f"arm={arm_side}",
        f"mapping={mapping}",
        "space: calibrate neutral pose",
        "q or esc: quit",
        f"neutral={'ready' if neutral_ready else 'auto'}",
    ]
    if action is not None:
        lines.extend(
            [
                f"pan={action['shoulder_pan.pos']:.1f}",
                f"lift={action['shoulder_lift.pos']:.1f}",
                f"elbow={action['elbow_flex.pos']:.1f}",
                f"wrist_flex={action['wrist_flex.pos']:.1f}",
                f"wrist_roll={action['wrist_roll.pos']:.1f}",
                f"gripper={action['gripper.pos']:.1f}",
            ]
        )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )


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


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent

    robot = SO101SimRobot(args.model)
    viewer = None
    if not args.no_sim_viewer:
        viewer = OfficialViewer(
            model=robot.model,
            data=robot.data,
            title="SO101 Webcam Skeleton Teleop",
            camera_name=args.camera,
            hidden=False,
        )
        viewer.initialize(paused=False)

    cap = open_camera(args.device, args.backend)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera device "
            f"{args.device}. On macOS, grant Camera permission to the app that launched this script "
            "(for example Terminal, iTerm2, or Codex) in System Settings > Privacy & Security > Camera, "
            "then rerun from that same app. If needed, try --backend avfoundation."
        )

    if args.mapping == "ik":
        mapper = CartesianIKTeleopMapper(
            robot=robot,
            arm_side=args.arm_side,
            smooth_alpha=args.smooth_alpha,
            lateral_gain_m=args.lateral_gain_m,
            vertical_gain_m=args.vertical_gain_m,
            reach_gain_m=args.reach_gain_m,
        )
    else:
        mapper = JointTeleopMapper(
            arm_side=args.arm_side,
            pan_gain=args.pan_gain,
            lift_gain=args.lift_gain,
            smooth_alpha=args.smooth_alpha,
        )
    pose_landmarker, hand_landmarker = create_landmarkers(root)

    last_action = mapper.smoothed_action.copy()
    last_frame_time = time.perf_counter()

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
                last_action = robot.send_action(action, body_mode="degrees", apply_mode=args.apply_mode)

            robot.step(1, args.apply_mode)

            if viewer is not None:
                if viewer.window is not None and glfw.window_should_close(viewer.window):
                    break
                render_sim_view(viewer)

            draw_overlay(frame, args.arm_side, args.mapping, last_action, mapper.neutral is not None)
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
            cv2.imshow("SO101 Skeleton Teleop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                success = mapper.capture_neutral(
                    pose_points,
                    hand_metrics,
                    robot.data.site_xpos[mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")].copy(),
                )
                print("neutral captured" if success else "neutral capture failed")
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
