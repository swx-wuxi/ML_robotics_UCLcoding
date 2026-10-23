1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

## Collecting multiple expert trajectories

Run from the workspace root using the RoboEnv Python environment:

```bash
python final_project/robotnik_arm_task/collect_expert_data.py
```

The default run samples 1000 target positions uniformly by area in an annulus
centred at world `(0, 0)`, with radii 0.9 to 1.5 m and fixed height 0.5 m.
Edit `NUM_EPISODES`, `TARGET_CENTER_XY`, `TARGET_MIN_RADIUS`,
`TARGET_MAX_RADIUS`, and `TARGET_HEIGHT` near the top of the collector to change
the experiment. Set the inner radius to zero to sample a full disk.
This is an initial sampling region, not a guarantee that every target is reachable.

Each episode starts a fresh simulator using the unchanged configuration, so the
initial base pose, arm configuration, and velocities remain fixed. The expert
controllers are unchanged: the base aligns first, drives with gentle steering,
then adjusts its final heading. Continuous simultaneous turning and translation
is not enforced by the collector.

Files are created only when the script is run, in a new timestamped folder under
`robotnik_arm_task/data/`. The folder contains `targets.json` (planned targets and
splits), individual `episode_XXXX.npz` files, `episodes.json` (results),
`dataset.npz` (merged samples), and `metadata.json` (collection settings).
Episodes are saved immediately; final merging uses temporary disk arrays rather
than retaining all episodes in RAM. Allow space for both individual and merged
data and temporary uncompressed arrays. Interruptions preserve completed episode
files, but automatic resume is not implemented.

With 1000 unique targets, the split is 700 training, 150 validation, and 150 test
episodes. Failed reaches are retained and marked; this is 1000 attempted
trajectories, not necessarily 1000 successful trajectories. Training code can
filter using `episode_success`. Inspect success rates before using the dataset.

For a small trial or explicit targets:

```bash
python final_project/robotnik_arm_task/collect_expert_data.py --episodes 5
python final_project/robotnik_arm_task/collect_expert_data.py --target 0 1.1 0.5 --target 1.1 0 0.5
```

The seed defaults to 42. `--gui` enables the simulator window; `--output` selects
a new output folder. Existing output folders are never overwritten. To train
on the new merged dataset, update `DEFAULT_DATA` in `linear.py`.
