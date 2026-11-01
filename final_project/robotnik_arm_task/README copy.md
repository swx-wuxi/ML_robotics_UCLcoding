### 0.Project overview and environment
This project uses MLP to control a mobile manipulator for an end-effector reaching task. A decision tree policy was also explored as an initial baseline, but its closed-loop performance was insufficient for the reaching task. The final approach therefore focuses on MLP. 

Policies are trained on expert demonstrations to predict four wheel velocity commands and seven arm joint torques from the measured robot state and target position. Performance is evaluated in Pybullet, with successful reaching requiring an end-effector position error below 1 cm.

This README explains how to collect data, train and evaluate the models, and locate the generated outputs. 

Run the code using the RoboEnv Python environment provided for the course, which can be activated by 'pixi shell'.

Ensure that the `models/` and `configs/` folders are accessible at the paths used by the scripts. Run all commands below from the repository root. 

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

### 2. Model training

#### Decision tree
```bash
python final_project/robotnik_arm_task/decision_tree.py
```

Preliminary tests showed insufficient closed-loop performance, so the decision tree was not considered in the final evaluation.

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
Set `MODEL_PATH` in `final_evaluation.py` to the model weight (saved in `trained_model_weights` repository)used to evaluate.

```bash
python final_project/robotnik_arm_task/final_evaluation.py 
```

Outputs are saved in `final_project/robotnik_arm_task/evaluation/final_*/`:
- `episode_XXXX.npz`: recorded states, actions and end-effector errors for each episode.
- `results.json`: success rate and per-episode outcomes, durations and final errors.(Most important file)
- `settings.json`: model path, test targets and evaluation configurations.

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
