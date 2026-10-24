"""Closed-loop PyBullet evaluation of a saved linear or tree policy on new targets.

Run: python evaluation_final.py. Model weights are frozen; no training is done.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import robotnik_arm_task as task
from collections import deque
import warnings


ROOT = Path(__file__).resolve().parent
# ************************************************
# Change the model weights here for comparison
# ************************************************
MODEL_PATH = ROOT / "trained_model_weights"/"neural_network_model.npz"
NUM_TARGETS = 100
TEST_SEED = 20261024  # 32 1 
USE_GUI = True
PLAYBACK_SPEED = 1.0
# Keep bounded commands, and save raw predictions separately for reproducibility.
CLIP_COMMANDS = True
# Diagnostic shadow only: its torques are logged and NEVER applied to the robot.
LOG_EXPERT_DIAGNOSTICS = True
# CORE CHANGE: require a settled base before enabling the reaching tree.
BASE_SETTLE_SECONDS = 0.25
BASE_STOP_LINEAR_SPEED = 0.02
BASE_STOP_ANGULAR_SPEED = 0.05

FEATURE_NAMES = (["base_x", "base_y", "base_z", "base_yaw"]
                 + [f"arm_q_{i}" for i in range(1, 8)]
                 + [f"arm_qd_{i}" for i in range(1, 8)]
                 + ["target_x", "target_y", "target_z"])
ACTION_NAMES = (["wheel_FR", "wheel_FL", "wheel_BL", "wheel_BR"]
                + [f"arm_torque_{i}" for i in range(1, 8)])


PHASE_CODES = {'align': 0, 'drive': 1, 'orient': 2}
BASE_COLUMNS = [0, 1, 2, 3, 18, 19, 20]
TREE_KEYS = ('children_left', 'children_right', 'feature', 'threshold', 'values')


def base_network(model):
    return {k[5:]: v for k, v in model.items() if k.startswith('base_nn_')}


def stage_tree(model, stage):
    return {key: model[f'{stage}_{key}'] for key in TREE_KEYS}


def validate_tree(model, n_features=21, n_outputs=11):
    n = len(model['feature'])
    if n == 0:
        raise ValueError('Empty tree')
    for key in TREE_KEYS[:-1]:
        if model[key].shape != (n,) or not np.isfinite(model[key]).all():
            raise ValueError(f'Invalid tree parameter: {key}')
    for key in ('children_left', 'children_right', 'feature'):
        if model[key].dtype.kind not in 'iu':
            raise ValueError(f'Invalid integer tree parameter: {key}')
    if model['values'].shape != (n, n_outputs) or not np.isfinite(model['values']).all():
        raise ValueError('Invalid tree outputs')
    left, right = model['children_left'], model['children_right']
    internal = ~((left == -1) & (right == -1))
    for child in (left[internal], right[internal]):
        if np.any(child <= np.arange(n)[internal]) or np.any(child >= n):
            raise ValueError('Invalid tree child index')
    if np.any(model['feature'][internal] < 0) or np.any(model['feature'][internal] >= n_features):
        raise ValueError('Invalid tree feature index')


def predict_tree(model, X, n_features=21):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim not in (1, 2) or X.shape[-1] != n_features or not np.isfinite(X).all():
        raise ValueError('Invalid tree input')
    single = X.ndim == 1
    X = X.reshape(-1, n_features)
    nodes = np.zeros(len(X), dtype=np.intp)
    while True:
        rows = np.flatnonzero(model['children_left'][nodes] != -1)
        if not len(rows):
            break
        current = nodes[rows]
        go_left = X[rows, model['feature'][current]] <= model['threshold'][current]
        nodes[rows] = np.where(go_left, model['children_left'][current], model['children_right'][current])
    result = model['values'][nodes]
    return result[0] if single else result


def update_navigation(phase, position, yaw, goal, bearing, pos_tol, ang_tol):
    delta = goal - np.asarray(position)[:2]
    distance = np.linalg.norm(delta)
    heading_error = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2*np.pi) - np.pi
    if phase == 'align' and (distance < pos_tol or abs(heading_error) < ang_tol):
        phase = 'drive' if distance >= pos_tol else 'orient'
    if phase == 'drive' and distance < pos_tol:
        phase = 'orient'
    yaw_error = (bearing - yaw + np.pi) % (2*np.pi) - np.pi
    arrived = phase == 'orient' and distance < pos_tol and abs(yaw_error) < ang_tol
    if phase == 'orient' and distance >= pos_tol:
        phase = 'align'
    return phase, bool(arrived)


class ArmHistory:
    def __init__(self, steps, stride):
        if steps < 1 or stride < 1:
            raise ValueError('Invalid history parameters')
        self.stride = stride
        self.history = deque(maxlen=1 + (steps-1)*stride)

    def append(self, state):
        if not self.history:
            self.history.extend(np.asarray(state).copy() for _ in range(self.history.maxlen))
        else:
            self.history.append(np.asarray(state).copy())
        return np.asarray(self.history, dtype=np.float32)[::self.stride].reshape(-1)


def base_features(X, phase):
    """Exclude arm states so arm deviations cannot change navigation predictions."""
    X = np.asarray(X)
    return np.concatenate((X[..., BASE_COLUMNS],
                           np.broadcast_to(np.asarray(phase), X.shape[:-1])[..., None]), axis=-1)

def validate_nn(model, n_features=None, n_outputs=7):
    steps = np.asarray(model.get('arm_window_steps', 1))
    if steps.shape != () or steps.dtype.kind not in 'iu' or steps.item() < 1:
        raise ValueError('Invalid NN history length')
    inputs = n_features if n_features is not None else 21 * int(steps.item())
    for key, shape in {'nn_x_mean': (inputs,), 'nn_x_scale': (inputs,),
                       'nn_y_mean': (n_outputs,), 'nn_y_scale': (n_outputs,)}.items():
        if model[key].shape != shape or not np.isfinite(model[key]).all():
            raise ValueError(f'Invalid NN parameter: {key}')
    if np.any(model['nn_x_scale'] <= 0) or np.any(model['nn_y_scale'] <= 0):
        raise ValueError('NN scales must be positive')
    width = inputs
    for i in range(3):
        w, b = model[f'nn_w{i}'], model[f'nn_b{i}']
        if (w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],)
                or not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError(f'Invalid NN layer {i}')
        width = w.shape[1]
    if width != n_outputs:
        raise ValueError(f'Expected {n_outputs} NN outputs')

def predict_nn(model, X):
    X = np.asarray(X, dtype=np.float32)
    inputs = len(model['nn_x_mean'])
    if X.ndim not in (1, 2) or X.shape[-1] != inputs or not np.isfinite(X).all():
        raise ValueError(f'Expected finite input ({inputs},) or (N, {inputs})')
    h = (X - model['nn_x_mean']) / model['nn_x_scale']
    # CORE CHANGE: two Linear+ReLU hidden layers, linear regression output.
    for i in range(3):
        h = h @ model[f'nn_w{i}'] + model[f'nn_b{i}']
        if i < 2:
            h = np.maximum(h, 0)
    return h * model['nn_y_scale'] + model['nn_y_mean']


def load_model():
    with np.load(MODEL_PATH, allow_pickle=False) as archive:
        model_type = str(archive["model_type"].item()) if "model_type" in archive else "linear"
        if model_type == "staged_neural_network":
            base_keys = (tuple(k for k in archive.files if k.startswith("base_nn_"))
                         if "base_nn_x_mean" in archive else tuple(f"base_{key}" for key in TREE_KEYS))
            keys = (base_keys +
                    ("nn_x_mean", "nn_x_scale", "nn_y_mean", "nn_y_scale") +
                    tuple(f"nn_{kind}{i}" for i in range(3) for kind in ("w", "b")))
        elif model_type == "staged_decision_tree":
            keys = tuple(f"{stage}_{key}" for stage in ("base", "arm") for key in TREE_KEYS)
        elif model_type == "decision_tree":
            keys = ("children_left", "children_right", "feature", "threshold", "values")
        elif model_type == "linear":
            keys = ("mean", "scale", "weights", "bias")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model = {key: archive[key] for key in keys}
        model["model_type"] = model_type
        if model_type in ("staged_decision_tree", "staged_neural_network"):
            # Old single-frame models remain loadable; new windows come from the archive.
            for key, default in (("arm_window_steps", 1), ("arm_window_stride", 1)):
                value = archive[key].item() if key in archive else default
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"Invalid temporal parameter: {key}")
                model[key] = value
            if "sample_time_step" in archive:
                model["sample_time_step"] = float(archive["sample_time_step"].item())
        if archive["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Model feature order is incompatible")
        if archive["action_names"].tolist() != ACTION_NAMES:
            raise ValueError("Model action order is incompatible")
        dataset_path = Path(str(archive["dataset_path"].item()))
    if model_type == "staged_neural_network":
        if "base_nn_x_mean" in model:
            validate_nn(base_network(model), n_features=8, n_outputs=4)
        else:
            validate_tree(stage_tree(model, "base"), 8, 4)
        validate_nn(model)
    elif model_type == "staged_decision_tree":
        validate_tree(stage_tree(model, "base"), 8, 4)
        validate_tree(stage_tree(model, "arm"), 21 * model["arm_window_steps"], 7)
    elif model_type == "decision_tree":
        validate_tree(model)
    else:
        for key, shape in {"mean": (21,), "scale": (21,),
                           "weights": (21, 11), "bias": (11,)}.items():
            if model[key].shape != shape or not np.isfinite(model[key]).all():
                raise ValueError(f"Invalid model parameter: {key}")
        if np.any(model["scale"] <= 0):
            raise ValueError("Model scale must be positive")
    # Use the exact collection settings associated with the trained model.
    metadata = json.loads((dataset_path.parent / "metadata.json").read_text())
    # Whole-file hashes also change for comments; log mismatches without blocking.
    changed = []
    for name, expected in metadata.get("source_sha256", {}).items():
        source = ROOT / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            changed.append(name)
    if changed:
        warnings.warn("Collection source/config differs: " + ", ".join(changed), stacklevel=2)
    model["source_config_changes"] = changed
    return model, dataset_path, metadata


def sample_test_targets(metadata, dataset_path):
    region = metadata["target_sampling"]
    if region["mode"] != "uniform_area_annulus":
        raise ValueError("This evaluator expects annulus sampling metadata")
    previous = json.loads((dataset_path.parent / "episodes.json").read_text())
    excluded = {tuple(item["target"]) for item in previous}
    rng = np.random.default_rng(TEST_SEED)
    targets = []
    while len(targets) < NUM_TARGETS:
        theta = rng.uniform(-np.pi, np.pi)
        radius = np.sqrt(rng.uniform(region["min_radius"]**2, region["max_radius"]**2))
        target = (region["center_xy"][0] + radius * np.cos(theta),
                  region["center_xy"][1] + radius * np.sin(theta), region["height"])
        if target not in excluded:
            targets.append(target)
            excluded.add(target)
    return np.asarray(targets)


def run_episode(model, target, metadata, episode_id):
    """Evaluate learned actions; staged policies also use expert posture holding."""
    sim = None
    rows = []
    reason = "manual_exit"
    aborted = False
    try:
        sim = task.pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(ROOT),
                                   use_gui=USE_GUI)
        sim.SetFloorFriction(1.0)
        client = sim.pybullet_client
        rid = sim.bot[0].bot_pybullet
        for j in range(client.getNumJoints(rid)):
            if "wheel" in client.getJointInfo(rid, j)[1].decode():
                client.changeDynamics(rid, j, contactStiffness=30000,
                                      contactDamping=1000,
                                      anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        task.configure_gripper_fingers(sim, client, rid, task.FIXED_FINGER_ANGLES)
        visual = client.createVisualShape(task.p.GEOM_SPHERE, radius=0.03,
                                          rgbaColor=[1, 0, 0, 0.85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                               basePosition=target.tolist())
        if USE_GUI:
            client.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=45,
                                              cameraPitch=-35,
                                              cameraTargetPosition=[0, 0, 0.5])
        names = np.expand_dims(np.asarray(sim.getNameActiveJoints()), 0)
        dynamics = task.PinWrapper("robotnik_arm.json", "pybullet", names,
                                   ["pybullet"], False, 0, str(ROOT))
        frame = dynamics.pin_model.getFrameId("robot_j2s7s300_end_effector")

        measured_q_pin = None

        def measure():
            nonlocal measured_q_pin
            pos = np.asarray(sim.GetBasePosition())
            ori = np.asarray(sim.GetBaseOrientation())
            q = np.asarray(sim.GetMotorAngles(0))
            qd = np.asarray(sim.GetMotorVelocities(0))
            yaw = task.quaternion2bearing(ori[3], ori[0], ori[1], ori[2])
            q_pin = dynamics.ReoderJoints2PinVec(np.concatenate((pos, ori, q)), "pos")
            measured_q_pin = q_pin
            task.pin.forwardKinematics(dynamics.pin_model, dynamics.pin_data, q_pin)
            task.pin.updateFramePlacements(dynamics.pin_model, dynamics.pin_data)
            ee = dynamics.pin_data.oMf[frame].translation.copy()
            return np.concatenate((pos, [yaw], q[4:11], qd[4:11], target)), ee

        staged = model["model_type"] in ("staged_decision_tree", "staged_neural_network")
        neural_arm = model["model_type"] == "staged_neural_network"
        base_nn = base_network(model) if "base_nn_x_mean" in model else None
        base_tree = stage_tree(model, "base") if staged and base_nn is None else None
        arm_tree = stage_tree(model, "arm") if staged and not neural_arm else None
        arm_history = ArmHistory(model.get("arm_window_steps", 1),
                                 model.get("arm_window_stride", 1)) if staged else None
        hold_q = np.asarray(sim.GetMotorAngles(0))[4:11].copy()
        arm_params = task.ArmControllerParams(**metadata["arm_params"])
        phase = "align"
        arm_active = False
        settled_time = 0.0
        previous_pose = None
        phase_log = []
        expert_log = []
        expert_q_ref = hold_q.copy()
        shoulder_frame = dynamics.pin_model.getFrameId("robot_j2s7s300_link_2")
        lower, upper = sim.GetBotJointsLimit()
        arm_lower, arm_upper = np.asarray(lower)[4:11], np.asarray(upper)[4:11]
        arm_start_time = None
        # For legacy models the navigation state remains a shadow for stop criteria.
        # Its expert wheel commands are discarded and never applied or fed to the policy.
        nav_state = task.BaseNavState()
        nav_params = task.BaseControllerParams(**metadata["base_params"])
        standoff = float(np.interp(target[2], task.STANDOFF_Z_POINTS, task.STANDOFF_POINTS))
        bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
        max_steps = int(metadata["max_steps"])
        criteria = task.StopCriteria(reach_tolerance=task.REACH_TOLERANCE,
                                      max_steps=max_steps)
        dt = float(sim.GetTimeStep())
        if staged and "sample_time_step" in model:
            if not np.isclose(dt, model["sample_time_step"], rtol=1e-5, atol=1e-9):
                raise ValueError("Simulation timestep differs from temporal training data")
        torque_limit = metadata["arm_params"]["max_torque"]
        wheel_limit = ((nav_params.max_lin_vel
                        + nav_params.max_ang_vel * nav_params.wheel_base_width / 2)
                       / nav_params.wheel_radius)
        limits = np.asarray([wheel_limit] * 4 + [torque_limit] * 7)
        cmd = task.MotorCommands()
        label_id = -1
        start_wall = time.perf_counter()
        # Match the template's Step-before-check_stop ordering, including its cap.
        for step in range(max_steps + 1):
            tick = time.perf_counter()
            X, ee = measure()
            if not np.isfinite(X).all() or not np.isfinite(ee).all():
                reason = "non_finite_state"
                break
            error = float(np.linalg.norm(target - ee))
            if staged:
                # CORE CHANGE: the supervisor chooses the stage, never wheel actions.
                # Arm joints are actively held with PD + gravity until base arrival.
                if not arm_active:
                    phase, arrived = update_navigation(
                        phase, X[:3], X[3], goal, bearing,
                        nav_params.pos_tol, nav_params.ang_tol)
                    stopped = False
                    if previous_pose is not None:
                        linear_speed = np.linalg.norm(X[:2] - previous_pose[:2]) / dt
                        yaw_delta = (X[3] - previous_pose[3] + np.pi) % (2*np.pi) - np.pi
                        stopped = (linear_speed < BASE_STOP_LINEAR_SPEED and
                                   abs(yaw_delta) / dt < BASE_STOP_ANGULAR_SPEED)
                    settled_time = settled_time + dt if arrived and stopped else 0.0
                    if settled_time >= BASE_SETTLE_SECONDS:
                        arm_active = True
                        arm_start_time = step * dt
                        nav_state.done = True
                        criteria.best_ee_error = error
                        criteria.best_ee_error_time = step * dt
                    elif step * dt > nav_params.nav_timeout:
                        reason = "base_navigation_timeout"
                        break
                if arm_active:
                    # CORE CHANGE: wheels remain stopped throughout arm reaching.
                    # CORE CHANGE: append pre-action state once per arm step. The first
                    # reaching state pads missing history, exactly as during training.
                    arm_input = arm_history.append(X)
                    arm_action = (predict_nn(model, arm_input) if neural_arm else predict_tree(
                        arm_tree, arm_input, n_features=len(arm_input)))
                    raw = np.concatenate((np.zeros(4), arm_action))
                else:
                    nav_input = base_features(X, PHASE_CODES[phase])
                    wheels = (np.zeros(4) if arrived else
                              predict_nn(base_nn, nav_input) if base_nn is not None else
                              predict_tree(base_tree, nav_input, n_features=8))
                    gravity = task.pin.computeGeneralizedGravity(
                        dynamics.pin_model, dynamics.pin_data, measured_q_pin)
                    hold_torque = arm_params.kp * (hold_q - X[4:11]) - arm_params.kd * X[11:18]
                    hold_torque = np.clip(hold_torque + gravity[6:13], -torque_limit, torque_limit)
                    raw = np.concatenate((wheels, hold_torque))
                previous_pose = X[:4].copy()
            else:
                raw = (predict_tree(model, X) if model["model_type"] == "decision_tree" else
                       ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"])
            if not np.isfinite(raw).all():
                reason = "non_finite_prediction"
                break
            action = np.clip(raw, -limits, limits) if CLIP_COMMANDS else raw.copy()
            if not staged:
                task.base_controller(nav_state, X[:3], X[3], goal, bearing, step * dt, nav_params)
            # CORE DIAGNOSTIC: run the official expert on the SAME measured state.
            # Its reference evolves along the learned rollout, not an expert rollout.
            if staged and LOG_EXPERT_DIAGNOSTICS:
                expert_torque, expert_q_ref, expert_qd_ref = task.arm_controller(
                    dynamics, measured_q_pin, X[4:11], X[11:18], expert_q_ref,
                    hold_q, target, ee.copy(), frame, shoulder_frame,
                    arm_lower, arm_upper, dt, arm_params, reach_active=arm_active)
                expert_log.append((expert_torque.copy(), expert_q_ref.copy(),
                                   expert_qd_ref.copy()))
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))),
                              ["velocity"] * 4 + ["torque"] * 13)
            sim.Step(cmd, "torque")
            phase_log.append("arm" if staged and arm_active else (phase if staged else "legacy"))
            rows.append((X.copy(), raw.copy(), action.copy(), step * dt, ee.copy(), error))
            if USE_GUI:
                if step % 100 == 0:
                    label_id = client.addUserDebugText(
                        f"ML trial {episode_id + 1}/{NUM_TARGETS} | "
                        f"t={step * dt:.1f}s | error={error:.3f}m",
                        [0, 0, 1.7], textColorRGB=[0, 0, 0],
                        replaceItemUniqueId=label_id)
                keys = client.getKeyboardEvents()
                if keys.get(ord("q"), 0) & client.KEY_WAS_TRIGGERED:
                    reason, aborted = "user_abort", True
                    break
            message = task.check_stop(criteria, step * dt, dt, error, nav_state.done)
            if message:
                reason = message
                break
            if USE_GUI:
                delay = dt / PLAYBACK_SPEED - (time.perf_counter() - tick)
                if delay > 0:
                    time.sleep(delay)
        final_X, final_ee = measure()
        final_error = float(np.linalg.norm(target - final_ee))
        final_finite = np.isfinite(final_error)
        official_success = reason.startswith("Target reached")
        # Require the actual post-action final state to meet tolerance as well.
        success = bool(official_success and final_finite and final_error < task.REACH_TOLERANCE)
        summary = dict(episode_id=episode_id, target=target.tolist(), success=success,
                       official_success=official_success, stop_message=reason,
                       samples=len(rows), duration_s=len(rows) * dt,
                       wall_time_s=time.perf_counter() - start_wall,
                       final_error_m=final_error if final_finite else None,
                       final_ee_pos=final_ee.tolist() if np.isfinite(final_ee).all() else None,
                       final_base_pos=final_X[:3].tolist() if np.isfinite(final_X[:3]).all() else None,
                       aborted=aborted, arm_start_time_s=arm_start_time,
                       controller=("base MLP + arm MLP + PD/gravity posture hold" if base_nn is not None else
                                   "base tree + arm MLP + PD/gravity posture hold" if neural_arm else
                                   "two trees + PD/gravity posture hold" if staged else "legacy"))
        if rows:
            columns = list(zip(*rows))
            data = {key: np.asarray(values) for key, values in zip(
                ("X", "raw_actions", "actions", "time", "ee_pos", "ee_error"), columns)}
            data["control_phase"] = np.asarray(phase_log)
            if expert_log:
                data["expert_shadow_torque"] = np.asarray([v[0] for v in expert_log])
                data["expert_shadow_q_ref"] = np.asarray([v[1] for v in expert_log])
                data["expert_shadow_qd_ref"] = np.asarray([v[2] for v in expert_log])
            data["wheel_cmds"] = data["actions"][:, :4]
            data["torque_arm"] = data["actions"][:, 4:]
            summary["clipped_command_fraction"] = float(np.mean(
                data["actions"] != data["raw_actions"]))
            print("Last wheel commands (rad/s):", data["wheel_cmds"][-1])
            print("Last arm torques (N*m):", data["torque_arm"][-1])
        else:
            data = {}
            summary["clipped_command_fraction"] = None
        return data, summary
    finally:
        if sim is not None:
            sim.pybullet_client.disconnect()


def main():
    if NUM_TARGETS <= 0 or PLAYBACK_SPEED <= 0:
        raise ValueError("NUM_TARGETS and PLAYBACK_SPEED must be positive")
    model, dataset_path, metadata = load_model()
    targets = sample_test_targets(metadata, dataset_path)
    out = ROOT / "evaluation" / datetime.now(timezone.utc).strftime("final_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(source_config_changes=model.get("source_config_changes", []), model=str(MODEL_PATH), model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    dataset=str(dataset_path), seed=TEST_SEED, targets=targets.tolist(),
                    region=metadata["target_sampling"], gui=USE_GUI,
                    clip_commands=CLIP_COMMANDS, max_steps=metadata["max_steps"],
                    reach_tolerance_m=task.REACH_TOLERANCE,
                    arm_window_steps=model.get("arm_window_steps", 1),
                    arm_window_stride=model.get("arm_window_stride", 1),
                    expert_shadow_diagnostics=LOG_EXPERT_DIAGNOSTICS,
                    base_settle_seconds=BASE_SETTLE_SECONDS,
                    base_stop_linear_speed=BASE_STOP_LINEAR_SPEED,
                    base_stop_angular_speed=BASE_STOP_ANGULAR_SPEED,
                    stop_policy="Staged models: geometric arrival and settling gate; navigation timeout fails. "
                                "Legacy models: shadow navigation. Official task success tolerance retained.")
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    results = []
    for i, target in enumerate(targets):
        print(f"\nEvaluating {i + 1}/{NUM_TARGETS}: {target}", flush=True)
        data, result = run_episode(model, target, metadata, i)
        np.savez_compressed(out / f"episode_{i:04d}.npz", **data,
                            target=target, feature_names=np.asarray(FEATURE_NAMES),
                            action_names=np.asarray(ACTION_NAMES))
        results.append(result)
        successes = sum(item["success"] for item in results)
        official = sum(item["official_success"] for item in results)
        report = dict(planned_targets=NUM_TARGETS, evaluated_targets=len(results),
                      complete=len(results) == NUM_TARGETS and not result["aborted"],
                      successes=successes, success_rate=successes / len(results),
                      official_success_rate=official / len(results), episodes=results)
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Success={result['success']}; final error={result['final_error_m']} m; "
              f"running success rate={successes}/{len(results)} ({100 * successes / len(results):.1f}%)",
              flush=True)
        if result["aborted"]:
            break
    print(f"\nClosed-loop success rate: {100 * report['success_rate']:.1f}%")
    print(f"Results and action trajectories: {out}")


if __name__ == "__main__":
    main()
