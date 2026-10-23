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
from tree_policy import predict_tree, validate_tree


ROOT = Path(__file__).resolve().parent
# ************************************************
# Change the model weights here for comparison
# ************************************************
MODEL_PATH = ROOT / "trained_model_weights"/"decision_tree_model.npz"
NUM_TARGETS = 10
TEST_SEED = 20261024
USE_GUI = True
PLAYBACK_SPEED = 1.0
# Keep bounded commands, and save raw predictions separately for reproducibility.
CLIP_COMMANDS = True

FEATURE_NAMES = (["base_x", "base_y", "base_z", "base_yaw"]
                 + [f"arm_q_{i}" for i in range(1, 8)]
                 + [f"arm_qd_{i}" for i in range(1, 8)]
                 + ["target_x", "target_y", "target_z"])
ACTION_NAMES = (["wheel_FR", "wheel_FL", "wheel_BL", "wheel_BR"]
                + [f"arm_torque_{i}" for i in range(1, 8)])


def load_model():
    with np.load(MODEL_PATH, allow_pickle=False) as archive:
        model_type = str(archive["model_type"].item()) if "model_type" in archive else "linear"
        if model_type == "decision_tree":
            keys = ("children_left", "children_right", "feature", "threshold", "values")
        elif model_type == "linear":
            keys = ("mean", "scale", "weights", "bias")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model = {key: archive[key] for key in keys}
        model["model_type"] = model_type
        if archive["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Model feature order is incompatible")
        if archive["action_names"].tolist() != ACTION_NAMES:
            raise ValueError("Model action order is incompatible")
        dataset_path = Path(str(archive["dataset_path"].item()))
    if model_type == "decision_tree":
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
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Collection source/config changed: {name}")
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
    """Use the official simulation setup but apply only model-generated actions."""
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

        def measure():
            pos = np.asarray(sim.GetBasePosition())
            ori = np.asarray(sim.GetBaseOrientation())
            q = np.asarray(sim.GetMotorAngles(0))
            qd = np.asarray(sim.GetMotorVelocities(0))
            yaw = task.quaternion2bearing(ori[3], ori[0], ori[1], ori[2])
            q_pin = dynamics.ReoderJoints2PinVec(np.concatenate((pos, ori, q)), "pos")
            task.pin.forwardKinematics(dynamics.pin_model, dynamics.pin_data, q_pin)
            task.pin.updateFramePlacements(dynamics.pin_model, dynamics.pin_data)
            ee = dynamics.pin_data.oMf[frame].translation.copy()
            return np.concatenate((pos, [yaw], q[4:11], qd[4:11], target)), ee

        # This shadow state is used ONLY by the unchanged official stop criteria.
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
            raw = (predict_tree(model, X) if model["model_type"] == "decision_tree" else
                   ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"])
            if not np.isfinite(raw).all():
                reason = "non_finite_prediction"
                break
            action = np.clip(raw, -limits, limits) if CLIP_COMMANDS else raw.copy()
            task.base_controller(nav_state, X[:3], X[3], goal, bearing, step * dt, nav_params)
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))),
                              ["velocity"] * 4 + ["torque"] * 13)
            sim.Step(cmd, "torque")
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
                       aborted=aborted)
        if rows:
            columns = list(zip(*rows))
            data = {key: np.asarray(values) for key, values in zip(
                ("X", "raw_actions", "actions", "time", "ee_pos", "ee_error"), columns)}
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
    settings = dict(model=str(MODEL_PATH), model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    dataset=str(dataset_path), seed=TEST_SEED, targets=targets.tolist(),
                    region=metadata["target_sampling"], gui=USE_GUI,
                    clip_commands=CLIP_COMMANDS, max_steps=metadata["max_steps"],
                    reach_tolerance_m=task.REACH_TOLERANCE,
                    stop_policy="Official check_stop with shadow expert navigation state; "
                                "expert commands discarded. Success also requires post-action tolerance.")
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
