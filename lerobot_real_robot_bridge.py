#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge process that sends LeRobot-style actions to a real SO101.")
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--max-relative-target", type=float, default=8.0)
    return parser


def make_robot(args: argparse.Namespace):
    lerobot_src = args.lerobot_root / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.utils import make_robot_from_config

    joint_limits = {
        "shoulder_pan": args.max_relative_target,
        "shoulder_lift": args.max_relative_target,
        "elbow_flex": args.max_relative_target,
        "wrist_flex": args.max_relative_target,
        "wrist_roll": args.max_relative_target,
        "gripper": max(4.0, args.max_relative_target),
    }
    config = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.robot_port,
        cameras={},
        max_relative_target=joint_limits,
        use_degrees=True,
    )
    return make_robot_from_config(config)


def main() -> None:
    args = build_parser().parse_args()
    robot = make_robot(args)
    robot.connect(calibrate=False)
    print("bridge_ready", flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            msg_type = payload.get("type")
            if msg_type == "action":
                robot.send_action(payload["action"])
            elif msg_type == "ping":
                print("pong", flush=True)
            elif msg_type == "close":
                break
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
