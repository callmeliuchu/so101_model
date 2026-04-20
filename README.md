# SO101 Model Notes

This directory contains the SO101 URDF and MuJoCo model files plus a local set of tools for loading, rendering, and inspecting them.

## Model Files

- `so101_new_calib.xml`: MuJoCo model using the newer midpoint-based joint calibration
- `so101_old_calib.xml`: MuJoCo model using the older fully-extended calibration
- `scene.xml`: bare robot scene, includes `so101_new_calib.xml` by default
- `task_scene.xml`: tabletop task scene with cube, bin, and named cameras
- `so101_new_calib.urdf` / `so101_old_calib.urdf`: URDF exports of the same robot

The meshes were generated from an Onshape CAD model via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot), then adjusted to use relative mesh paths.

## Calibration Notes

The MuJoCo files support two calibration conventions:

- `so101_new_calib.xml`: joint virtual zero is near the middle of each joint range
- `so101_old_calib.xml`: joint virtual zero is near the fully-extended horizontal pose

Important limitation:

- LeRobot uses the gripper as a linear `0..100` opening value
- The current URDF/MuJoCo files still represent the gripper as a hinge joint
- Any real-to-sim synchronization needs an explicit gripper mapping layer

## Local Environment

A standalone conda environment lives in:

- [`.conda`](/Users/liuchu/codes/so101_model/.conda)

Create or refresh it with:

```bash
cd /Users/liuchu/codes/so101_model
./setup_conda_env.sh
```

This installs a local Python, MuJoCo, GLFW, NumPy, and image rendering dependencies without relying on `lerobot`.

## What Was Verified

These paths have been tested successfully on this machine:

- Static XML loading with `mujoco.MjModel.from_xml_path(...)`
- Offscreen rendering of a single PNG
- Offscreen rendering of MP4 video
- Official MuJoCo desktop UI via the native `simulate` binary
- A custom interactive GLFW viewer built on MuJoCo's official visualization API

This path did not work reliably on this machine:

- Python `mujoco.viewer.launch(...)`
- Python `mujoco.viewer.launch_passive(...)`

The failure is in the Python viewer/UI layer, not in model loading or rendering.

## Static Snapshot

Render one PNG from a local XML file:

```bash
cd /Users/liuchu/codes/so101_model
./.conda/bin/python ./render_snapshot.py --model task_scene
```

Default output:

- [snapshot.png](/Users/liuchu/codes/so101_model/outputs/snapshot.png)

Script:

- [render_snapshot.py](/Users/liuchu/codes/so101_model/render_snapshot.py)

## Official MuJoCo UI

If you want the official MuJoCo desktop interface with the left/right panels, menus, and UI controls, use the native `simulate` app copied from the official MuJoCo macOS release.

Launch it with:

```bash
cd /Users/liuchu/codes/so101_model
./run_mujoco_official_ui.sh task_scene
```

Launcher:

- [run_mujoco_official_ui.sh](/Users/liuchu/codes/so101_model/run_mujoco_official_ui.sh)

Local app bundle:

- [MuJoCo.app](/Users/liuchu/codes/so101_model/.official-mujoco/MuJoCo.app)

The official `simulate` binary worked on this machine even though Python `mujoco.viewer` did not.

## Official-API GLFW Viewer

There is also a viewer built from MuJoCo's official low-level visualization API:

- GLFW window
- `mjv_updateScene`
- `mjr_render`
- `mjv_moveCamera`

Run it with:

```bash
cd /Users/liuchu/codes/so101_model
./run_official_viewer.sh --model task_scene
```

Files:

- [run_official_viewer.sh](/Users/liuchu/codes/so101_model/run_official_viewer.sh)
- [official_glfw_viewer.py](/Users/liuchu/codes/so101_model/official_glfw_viewer.py)

This viewer gives real-time interaction, but not the full official MuJoCo desktop side panels.

## Custom Local Viewer

There is also a separate local viewer/utility script for:

- headless stepping
- interactive keyboard control
- offscreen MP4 export

Run it with:

```bash
cd /Users/liuchu/codes/so101_model
./run_mujoco.sh --model task_scene --interactive
```

Useful commands:

```bash
# Bare robot scene
./run_mujoco.sh --model scene

# Tabletop task scene
./run_mujoco.sh --model task_scene

# Headless smoke test
./run_mujoco.sh --headless --steps 100

# Offscreen video render
./run_mujoco.sh --model task_scene --render-video ./outputs/so101_task.mp4 --steps 240
```

Files:

- [run_mujoco.sh](/Users/liuchu/codes/so101_model/run_mujoco.sh)
- [visualize_mujoco.py](/Users/liuchu/codes/so101_model/visualize_mujoco.py)

Keyboard controls in `--interactive` mode:

- `1/Q`: shoulder pan up/down
- `2/W`: shoulder lift up/down
- `3/E`: elbow flex up/down
- `4/R`: wrist flex up/down
- `5/T`: wrist roll up/down
- `6/Y`: gripper up/down
- `Space`: pause/resume
- `R`: reset to home pose
- `P`: print joint targets
- `O`: print gripper, cube and bin positions

## LeRobot-Style Command Control

There is now a control path that accepts the same action-key shape used by LeRobot follower robots:

- `shoulder_pan.pos`
- `shoulder_lift.pos`
- `elbow_flex.pos`
- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

Launcher:

- [run_lerobot_command.sh](/Users/liuchu/codes/so101_model/run_lerobot_command.sh)

Script:

- [lerobot_command_control.py](/Users/liuchu/codes/so101_model/lerobot_command_control.py)

Recommended mode for now:

- `--apply-mode qpos`
- This directly mirrors the commanded joint targets into MuJoCo joint positions
- It is the most stable option if you want the sim arm to follow the same command format as the real arm

Physics actuator mode is also available:

- `--apply-mode actuator`
- This uses MuJoCo position actuators through `data.ctrl`
- On this machine and with the current XML gains, the gripper follows well but the arm joints do not yet track targets reliably under gravity
- Keep this mode for future actuator tuning, not as the default command-mirroring path

Examples:

```bash
cd /Users/liuchu/codes/so101_model

# One LeRobot-style command from the shell
./run_lerobot_command.sh \
  --model task_scene \
  --apply-mode qpos \
  --set shoulder_pan.pos=10 \
  --set shoulder_lift.pos=-20 \
  --set elbow_flex.pos=35 \
  --set wrist_flex.pos=-10 \
  --set wrist_roll.pos=5 \
  --set gripper.pos=70 \
  --print-observation

# Same idea, but with a JSON action dict
./run_lerobot_command.sh \
  --model task_scene \
  --apply-mode qpos \
  --action-json '{"shoulder_pan.pos": 15, "shoulder_lift.pos": -30, "gripper.pos": 20}' \
  --print-sent-action \
  --print-observation

# Replay a small action sequence
./run_lerobot_command.sh \
  --model task_scene \
  --apply-mode qpos \
  --sequence-file ./example_lerobot_sequence.json \
  --print-observation

# Watch the sequence in the local GLFW viewer
./run_lerobot_command.sh \
  --model task_scene \
  --apply-mode qpos \
  --sequence-file ./example_lerobot_sequence.json \
  --viewer
```

Sequence example:

- [example_lerobot_sequence.json](/Users/liuchu/codes/so101_model/example_lerobot_sequence.json)

Units:

- Body joints default to `degrees`
- Use `--body-mode radians` if your command producer already emits radians
- Use `--body-mode range_m100_100` if you want a normalized `-100..100` body command range
- Gripper is always interpreted as `0..100`, matching LeRobot follower usage

## Public SO101 Episode Import

There is also a helper to pull public SO101 LeRobot episodes from Hugging Face and convert them into the local sequence format used by `run_lerobot_command.sh`.

Launcher:

- [import_hf_lerobot_episode.sh](/Users/liuchu/codes/so101_model/import_hf_lerobot_episode.sh)

Script:

- [import_hf_lerobot_episode.py](/Users/liuchu/codes/so101_model/import_hf_lerobot_episode.py)

Currently supported public sources:

- [samuelcombey/so101_data](https://huggingface.co/datasets/samuelcombey/so101_data)
- [BobChang/lerobot-so101](https://huggingface.co/datasets/BobChang/lerobot-so101)

These datasets expose `action` as:

- `shoulder_pan.pos`
- `shoulder_lift.pos`
- `elbow_flex.pos`
- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

Important note:

- The public SO101 datasets above use body-joint commands in a normalized `-100..100` range
- For those imported episodes, playback should use `--body-mode range_m100_100`

Example:

```bash
cd /Users/liuchu/codes/so101_model

# Download one public SO101 episode and convert it to a local sequence
./import_hf_lerobot_episode.sh \
  --dataset samuelcombey/so101_data \
  --episode-index 0 \
  --stride 5 \
  --max-frames 60 \
  --merge-duplicates

# Play the imported episode in the sim
./run_lerobot_command.sh \
  --model task_scene \
  --apply-mode qpos \
  --body-mode range_m100_100 \
  --sequence-file ./external_data/samuelcombey__so101_data_episode_000000_sequence.json \
  --viewer
```

Generated files are written under:

- `/Users/liuchu/codes/so101_model/external_data`

## Webcam Skeleton Teleop

There is now a first-pass webcam teleoperation tool that maps your arm pose and hand pinch gesture to the SO101 sim.

Launcher:

- [run_webcam_teleop.sh](/Users/liuchu/codes/so101_model/run_webcam_teleop.sh)

Script:

- [webcam_skeleton_teleop.py](/Users/liuchu/codes/so101_model/webcam_skeleton_teleop.py)

What it does:

- webcam RGB stream -> MediaPipe pose + hand landmarks
- shoulder / elbow / wrist motion -> robot arm joints
- thumb-index pinch distance -> `gripper.pos`
- output uses the same LeRobot-style action keys as the other control scripts

Current status:

- this is a practical first-pass teleop demo, not a calibrated retargeting pipeline
- it works best with one visible arm and a clear side view of your forearm and hand
- the default control path uses `--apply-mode qpos` for stable direct sim mirroring

First run:

```bash
cd /Users/liuchu/codes/so101_model
./setup_conda_env.sh
./run_webcam_teleop.sh --model scene --mirror
```

Recommended controls:

- `Space`: capture the current arm pose as the neutral reference
- `Q` or `Esc`: quit

Recommended startup:

```bash
./run_webcam_teleop.sh \
  --model scene \
  --mirror \
  --arm-side right \
  --mapping ik
```

Notes:

- the script auto-downloads the official MediaPipe `.task` models into `/Users/liuchu/codes/so101_model/.models`
- use `--no-sim-viewer` if you only want the camera window
- `--pan-gain` and `--lift-gain` let you tune motion sensitivity
- `--mapping ik` uses hand-wrist position to drive the robot end effector and is the recommended mode
- `--mapping joint` keeps the older direct joint heuristic mapping
- `--lateral-gain-m`, `--vertical-gain-m`, and `--reach-gain-m` tune the IK workspace mapping
- `--apply-mode actuator` is available, but `qpos` is still the recommended default
- on macOS, Camera permission must be granted to the host app that launches the script, such as `Terminal.app`, `iTerm2`, or `Codex`
- if camera open fails on macOS, try `--backend avfoundation`

## Real + Sim Teleop

There is also a dual-backend webcam teleop runner that can send the same skeleton-derived action to:

- the MuJoCo sim
- the real SO101 follower robot
- or both at once

Launcher:

- [run_webcam_dual_teleop.sh](/Users/liuchu/codes/so101_model/run_webcam_dual_teleop.sh)

Script:

- [webcam_dual_teleop.py](/Users/liuchu/codes/so101_model/webcam_dual_teleop.py)

Recommended usage:

```bash
cd /Users/liuchu/codes/so101_model

./run_webcam_dual_teleop.sh \
  --target both \
  --model scene \
  --mirror \
  --mapping ik \
  --robot-port /dev/tty.usbmodem5B421378821 \
  --robot-id my_awesome_follower_arm
```

Safety behavior:

- the real robot starts in `disabled` send mode by default
- press `e` in the webcam window to toggle real-robot command sending on or off
- press `Space` to capture a fresh neutral pose before enabling the real robot

Important notes:

- the real robot path reuses the local `lerobot` checkout under `/Users/liuchu/codes/lerobot`
- real robot commands are clamped with `--max-relative-target`, default `8.0`
- both real and sim teleop paths also clamp each outgoing joint command to the SO101 model's absolute joint range before sending
- start with `--target sim` or `--target both` and keep the real path disabled until the sim motion looks correct

## LeRobot Calibration Files

The LeRobot default calibration directory is not inside the repo. By default it resolves to:

- `~/.cache/huggingface/lerobot/calibration`

On this machine, the local calibration files were found and copied into this directory for inspection:

- [my_awesome_follower_arm.json](/Users/liuchu/codes/so101_model/my_awesome_follower_arm.json)
- [my_awesome_leader_arm.json](/Users/liuchu/codes/so101_model/my_awesome_leader_arm.json)

Original source files were:

- `/Users/liuchu/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`
- `/Users/liuchu/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_awesome_leader_arm.json`

## Real Robot Synchronization Assessment

Synchronizing the real arm to the sim is feasible at the joint-state level:

- read real joint positions
- map them to MuJoCo joint coordinates
- update sim `qpos`
- call `mj_forward`
- render virtual camera images

What is already good enough for that:

- kinematic structure
- mesh geometry
- named joints and gripper frame
- verified XML loading and rendering

What still needs explicit handling:

- choose the correct calibration convention: `new` vs `old`
- define a real-gripper to MuJoCo-hinge mapping
- verify motor sign conventions
- verify joint ordering against real robot telemetry

What is not yet guaranteed:

- high-fidelity contact dynamics
- real/sim grasp agreement
- exact servo response matching

So the current model is suitable for:

- pose mirroring
- visualization
- virtual camera data capture

But not yet a high-fidelity physics twin without additional tuning.
