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
a new output folder. Existing output folders are never overwritten.

## Linear baseline

After collection finishes, run:

```bash
python final_project/robotnik_arm_task/linear.py
```

The script selects the latest timestamped multi-episode folder containing both
`dataset.npz` and the completion marker `metadata.json`. Set `DATA_PATH` at the
top of `linear.py` to pin a specific dataset for reproducibility. Incomplete
collections are skipped. No automatic training is triggered by collection.

Training preserves the collector's train/validation/test assignments and uses
successful episodes only. Each split must contain successful samples; there is
no fallback to splitting a single trajectory. Input normalization is fitted on
training samples only. Incremental QR least squares reduces training workspace
memory, although the selected dataset arrays are still loaded into RAM.

Outputs in `robotnik_arm_task/` are `linear_model.npz`, `linear_metrics.json`,
and `linear_predictions.png`; repeated runs overwrite these baseline outputs.
The figure shows one successful episode from each split. Metrics use all
successful samples, so longer episodes carry more weight. These are offline
expert-action prediction results, not closed-loop success rates or evaluations
over failed expert episodes. No simulator is started.

## Closed-loop final evaluation

```bash
python final_project/robotnik_arm_task/evaluation_final.py
```

This starts PyBullet and loads `linear_model.npz` without retraining. The evaluator
also reads the collection metadata and episode manifest referenced by the saved
model, so keep that data folder accessible. A fresh simulator is created for each
of 100 targets sampled with a new seed from the recorded collection region.
Targets exactly matching any collected target are excluded. Initial conditions,
contact settings, and finger control match the official task. Source/config hashes
must match those recorded at collection time.

Every simulation step uses the measured state to predict four wheel velocities
and seven arm torques. No expert arm control, gravity compensation, or navigation
commands are applied. The official base controller runs only as a shadow state
machine for the original navigation-gated stopping rules; its output commands are
discarded. This stopping supervisor is separate from the learned policy.

The default evaluates a bounded policy: arm torque is clipped to the recorded
expert limit, and wheel velocity is clipped to a conservative bound derived from
the recorded expert linear/angular speed limits. Raw and applied commands are
both recorded. Set `CLIP_COMMANDS = False` to explicitly evaluate the raw policy.
This is a simulation benchmark, not a hardware deployment script.

Edit `MODEL_PATH`, `NUM_TARGETS`, `TEST_SEED`, `USE_GUI`, and `PLAYBACK_SPEED` at
the top of the evaluator. GUI playback defaults to approximately real time;
100 trials can therefore take many minutes. Press `q` in PyBullet to abort the
batch and preserve results from completed steps. Closing the GUI is not the
supported abort path.

Outputs are created only on execution in a new timestamped `evaluation/final_*`
folder: `settings.json`, progressively updated `results.json`, and individual
`episode_XXXX.npz` files. NPZ fields include pre-action `X`, `raw_actions`,
applied `actions`, `wheel_cmds`, `torque_arm`, `time`, `ee_pos`, and `ee_error`.
The GUI displays the target, robot movement, trial number, time, and EE error.

The reported success rate is an observed closed-loop rate, not a probability
predicted by the linear regressor. Success requires the official reached message
and a post-action final EE distance below 0.01 m. `official_success_rate` is also
reported separately. Stalls and timeouts count as failures; an aborted batch is
marked incomplete. These fresh targets test generalization within the sampled
region, not new heights or new initial conditions.
