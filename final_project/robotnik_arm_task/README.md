### 0.Project overview and environment

This project trains separate base and arm MLPs by behaviour cloning for
mobile-manipulator reaching. The policies predict four wheel velocities and
seven joint torques from robot observations and the target position. Evaluation
uses PyBullet with a 1 cm end-effector position tolerance.

Activate the course-provided RoboEnv environment from the workspace root:

```bash
cd RoboEnv
pixi shell
cd ..
```

Run subsequent commands from the workspace root containing `RoboEnv/` and
`final_project/`. Keep `models/` and `configs/` under
`final_project/robotnik_arm_task/`.

#### Code sources and modifications

`robotnik_arm_task.py` and `robotnik_arm_control.py` are adapted from the
[COMP0245 course template](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/main/final/robotnik_arm_task),
which provides simulation setup and expert controllers using PyBullet,
Pinocchio and `simulation_and_control`. Robot models and configuration files
are supplied course assets.

We wrote the data collection, quality checks, augmentation, training, evaluation,
plotting and terminal feedback scripts. MLP training uses scikit-learn;
Bayesian optimisation uses scikit-optimize.

### 1. How to collect expert data

```bash
python final_project/robotnik_arm_task/collect_expert_data.py
```

The script collects expert demonstrations for 1500 targets sampled uniformly by area
in an annulus centred at `(0, 0)`, with radii 0.9–1.5 m and height 0.5 m
(default seed: 42). Each episode uses the original expert controller and the same
initial robot configuration. Target settings can be changed at the top of the script;
use `--episodes 5` for a quick trial or `--gui` to display the simulation.

Targets are split into 1050 training, 225 validation, and 225 test episodes.
Failed episodes are saved and marked by `episode_success`; training uses successful
episodes only. No data augmentation or model training is performed during collection.

Outputs are saved in a new `final_project/robotnik_arm_task/data/expert_*/` folder:
- `dataset.npz`: merged state–action samples.
- `episode_XXXX.npz`: individual trajectories.
- `targets.json`: sampled targets and split assignments.
- `episodes.json`: episode results.
- `metadata.json`: collection settings and data description.

#### Data quality checks

```bash
python final_project/robotnik_arm_task/check_data_quality.py
```

Audits the latest `data/expert_*/` dataset for array validity, time/state alignment,
target-split overlap, action limits and success-label consistency. It also reports
target and state coverage. Use `--dataset <folder>` to select a dataset and
`--check-merged` to compare merged core arrays against individual episode files.

Outputs are saved in `final_project/robotnik_arm_task/outputs/data_quality_*/`:

- `report.html` and `dataset_*.png`: report and diagnostic plots.
- `metrics.json`: metrics and findings.
- `comparison.csv` and `episodes.csv`: dataset and episode summaries.

`PASS_SCANNED` indicates no hard errors in the completed scan; `PRELIMINARY`
indicates incomplete coverage, and `ERROR` indicates failed checks.

### 2. Model training

#### MLP
Run Stage 1 to train separate Base and Arm MLPs from raw expert trajectories:

```bash
python final_project/robotnik_arm_task/neural_network_stage1.py
```

The Arm network uses velocity, angle and near-target augmentation. Angle samples
are generated automatically and cached on the first run. The intermediate model
is saved as `neural_network_near_target_augmented.npz`.

After geting this weight, run: 

```bash
python final_project/robotnik_arm_task/neural_network.py --bayes
```

Each trial starts from the intermediate model and updates only the Arm network,
mixing original, augmented and expert recovery samples. The Base network and
normalisation parameters remain fixed. Learning rate, L2 regularisation and the
recovery sampling ratio are selected using nominal and recovery validation errors.BO is applied for hyperparameter tuning. 

If the parameter `--bayes` in the command above is missing, it will not call BO functions and only perform regular fine-tuning.

Weights are saved as `trained_model_weights/neural_network_bo_recovery_*.npz`,which performs better than the intermediate mode in this simulation.

### 3. Model evaluation 
Set `MODEL_PATH` in `final_evaluation.py` to the model weight (saved in `trained_model_weights` repository)used to evaluate.Also you can change number of targets and test seed. 

Evaluation requires the collection's `metadata.json` and `episodes.json`, but not
`dataset.npz` or individual training trajectories; when using downloaded weights,
update `dataset_path` in `load_model()` to a local path whose parent contains
these JSON files, as the saved weights embed the original absolute path.

```bash
python final_project/robotnik_arm_task/final_evaluation.py 
```

Set `USE_GUI` to `True` if you want to see the Pybullet simulation environment.

Outputs are saved in `final_project/robotnik_arm_task/evaluation/final_*/`:
- `episode_XXXX.npz`: recorded states, actions and end-effector errors for each episode.
- `results.json`: success rate and per-episode outcomes, durations and final errors.(Most important file)
- `settings.json`: model path, test targets and evaluation configurations.

#### Generalisation and robustness tests

```bash
python final_project/robotnik_arm_task/performance_evaluation.py
```

Set `MODEL_PATH` in this script independently of `final_evaluation.py`.
Adjust `NUM_TARGETS`, `TEST_SEED` and `USE_GUI` as needed (defaults: 100, 42 and `False`).

- `TEST_GENERALIZATION = True`: test random targets at radii 0.75–1.65 m and
  heights 0.45–0.55 m for the original dataset. Change
  `GENERALIZATION_RADIUS_RANGE_M` and `GENERALIZATION_HEIGHT_OFFSETS_M` to adjust
  the absolute radius bounds and height offsets from the collection height.
- `TEST_ROBUSTNESS = True`: compare identical targets from the collection region
  with and without external forces and joint-observation noise. Three 0.1 s
  end-effector force pulses start 0.25, 1.0 and 2.0 s after arm activation,
  using world-frame forces `(4,0,0)`, `(0,-4,0)` and `(0,0,3)` N. During pulses,
  Gaussian position/velocity noise has standard deviations of 0.01 rad and
  0.05 rad/s. All perturbation settings are editable at the top of the script.

Both switches enabled run separate suites; both disabled run nothing.
Success uses the original tolerance and true simulator state. Early termination
may skip later pulses. Only success rates are printed and saved to
`final_project/robotnik_arm_task/evaluation/performance_*/summary.json`.

### 4. Arm diagnose
```bash
python final_project/robotnik_arm_task/diagnose_arm.py 
```
Outputs are saved in `final_project/robotnik_arm_task/diagnostics/arm_plots_*/`:
- `joint_angle_velocity.png`: expert and MLP joint angles and velocities.
- `joint_torque.png`: applied joint torques from the two independent rollouts.
- `ee_path_3d.png`: end-effector paths from a shared start to the same target.
- `ee_distance.png`: distance to the target over time, with a 1 cm threshold and a 0–15 cm zoom.

Set `DEFAULT_EPISODE` in `diagnose_arm.py` to an evaluation episode.

Set `RUNS` in `distance_evaluation.py` to the evaluation folders to compare.

### 5. Offline action prediction metrics

```bash
python final_project/robotnik_arm_task/offline_nn_metrics.py
```

Set `MODEL_PATH` in the script or pass `--model <path/to/model.npz>`.
The script evaluates saved weights on the original successful train, validation
and test samples without retraining. Predictions are inverse-standardised and
not clipped. Training metrics exclude augmented samples.

Outputs are saved in `final_project/robotnik_arm_task/outputs/offline_metrics_*/`:
- `metrics.json`: per-output and overall RMSE and R², sample counts and calculation notes.
- `metrics.csv`: the same scores in tabular form.
- `base_rmse_r2.png`: wheel-command RMSE and R² across train, validation and test sets.
- `arm_rmse_r2.png`: joint-torque RMSE and R² across the three sets.

### 6. Closed-loop distance comparison

```bash
python final_project/robotnik_arm_task/distance_evaluation.py
```

Outputs are saved in `final_project/robotnik_arm_task/outputs/ee_comparison_*/`:
- `comparison.csv`: success rates, final and minimum distances, distance MSE/RMSE and successful completion times.
- `episodes.csv`: individual episode metrics for checking failures.
- `metrics.json`: aggregate results and metric definitions.
- `ee_distance_comparison.png`: mean distance and mean squared distance during the first 5 s after Arm activation.

The curves use only targets with complete 5 s traces in every compared run;
the retained count is shown on the figure. Squared distances measure error to
the target, not deviation from an expert trajectory.


### 7. Add terminal feedback correction

```bash
python final_project/robotnik_arm_task/feedback_improve.py
```

This script combines the trained neural policy with controller-level terminal
feedback compensation to correct residual reaching errors during execution.
The model weights remain fixed; the feedback controller adds bounded corrections
to the predicted arm torques near the target. After running this command, the evaluation is started and it will output evaluation metrics.

This hybrid approach achieved the
best observed success rate of **94%** in our evaluation. Run this script to see the best results for this task.

The script uses `MODEL_PATH` from `final_evaluation.py` by default. Pass
`--model <path/to/model.npz>` to select a different trained model.

Outputs are saved in `final_project/robotnik_arm_task/evaluation/feedback_*/`:

- `episode_XXXX.npz`: states, actions, nominal arm torques and feedback corrections.
- `results.json`: success rate and per-episode outcomes.
- `settings.json`: model, targets, evaluation settings and feedback parameters.
