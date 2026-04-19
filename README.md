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
