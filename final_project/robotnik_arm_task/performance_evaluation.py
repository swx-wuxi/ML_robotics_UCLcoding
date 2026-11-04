"""Generalisation and robustness evaluation using final_evaluation's rollout.

Run from the repository root in RoboEnv:
    python final_project/robotnik_arm_task/performance_evaluation.py
No training. Both switches run separate suites, never a combined stress test.
"""

from datetime import datetime, timezone
import json

import numpy as np

import final_evaluation as evaluation


# ===================== Test switches =====================
TEST_GENERALIZATION = False  # Random targets in the configured radius and height ranges.
TEST_ROBUSTNESS = True   # Matched targets: clean baseline vs force and observation noise.

ROOT = evaluation.ROOT
MODEL_PATH = ROOT / "trained_model_weights/neural_network_bo_recovery_20261028T190338_972438Z.npz"
NUM_TARGETS = 100
TEST_SEED = 42
USE_GUI = False
PLAYBACK_SPEED = 1.0

# Absolute radii around the collection centre; this includes part of the training region.
GENERALIZATION_RADIUS_RANGE_M = (0.75, 1.65)
# Uniform height offsets from the collection height: z=0.5 -> z=0.4..0.6 m.
GENERALIZATION_HEIGHT_OFFSETS_M = (-0.05, 0.05)

# Times are measured FROM ARM ACTIVATION, not from base navigation start.
# Each pulse applies a world-frame force at the end-effector centre of mass.
PERTURBATION_TIMES_S = (0.25, 1.0, 2.0)
PERTURBATION_DURATION_S = 0.10
# Moderate pulses: half the original force magnitude.
PERTURBATION_FORCES_N = ((4.0, 0.0, 0.0), (0.0, -4.0, 0.0), (0.0, 0.0, 3.0))
# Independent zero-mean Gaussian measurement noise, only DURING each pulse.
# Set forces or noise standard deviations to zero to isolate the other factor.
JOINT_POSITION_NOISE_STD_RAD = 0.01
JOINT_VELOCITY_NOISE_STD_RAD_S = 0.05
NOISE_SEED = 20261104


def sample_generalization_targets(metadata, count, seed):
    region = metadata["target_sampling"]
    if region["mode"] != "uniform_area_annulus":
        raise ValueError("Generalisation requires annulus collection metadata")
    low, high = GENERALIZATION_RADIUS_RANGE_M
    z_low, z_high = GENERALIZATION_HEIGHT_OFFSETS_M
    if (count <= 0 or not np.isfinite([low, high, z_low, z_high]).all()
            or not 0 <= low < high or not z_low < z_high):
        raise ValueError("Use ordered nonnegative radii and finite ordered height offsets")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, count)
    # Uniform area sampling in XY and independent uniform sampling in height.
    radius = np.sqrt(rng.uniform(low**2, high**2, count))
    z = region["height"] + rng.uniform(z_low, z_high, count)
    return np.column_stack((region["center_xy"][0] + radius * np.cos(theta),
                            region["center_xy"][1] + radius * np.sin(theta), z))


class ArmPerturbation:
    """Inject noise only into policy inputs; all scores use true simulator state."""

    def __init__(self, episode_id):
        self.times = np.asarray(PERTURBATION_TIMES_S, dtype=float)
        self.forces = np.asarray(PERTURBATION_FORCES_N, dtype=float)
        self.duration = float(PERTURBATION_DURATION_S)
        self.noise_std = np.repeat([JOINT_POSITION_NOISE_STD_RAD,
                                    JOINT_VELOCITY_NOISE_STD_RAD_S], 7)
        if (self.times.ndim != 1 or not len(self.times)
                or not np.isfinite(self.times).all() or np.any(self.times < 0)
                or not np.isfinite(self.duration) or self.duration <= 0
                or np.any(np.diff(self.times) < self.duration)
                or self.forces.shape != (len(self.times), 3)
                or not np.isfinite(self.forces).all()
                or not np.isfinite(self.noise_std).all() or np.any(self.noise_std < 0)):
            raise ValueError("Invalid/non-ordered/overlapping perturbation pulses or noise settings")
        self.rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, episode_id]))
        self.arm_start = None
        self.event_id = -1
        self.noise = np.zeros(21)
        self.force = np.zeros(3)
        self.applied_steps = np.zeros(len(self.times), dtype=int)

    def bind(self, client, robot_id):
        self.client, self.robot_id = client, robot_id
        links = {client.getJointInfo(robot_id, j)[12].decode(): j
                 for j in range(client.getNumJoints(robot_id))}
        self.link_id = links["robot_j2s7s300_end_effector"]

    def observe(self, state, now, arm_start, dt):
        self.arm_start = arm_start
        self.noise = np.zeros(21)
        self.force = np.zeros(3)
        self.event_id = -1
        # Round starts/durations UP to physics steps so even short pulses execute.
        elapsed_step = int(round((now - arm_start) / dt))
        starts = np.ceil(self.times / dt - 1e-9).astype(int)
        width = max(1, int(np.ceil(self.duration / dt - 1e-9)))
        active = np.flatnonzero((elapsed_step >= starts) & (elapsed_step < starts + width))
        if len(active):
            self.event_id = int(active[0])
            self.force = self.forces[self.event_id].copy()
            self.noise[4:18] = self.rng.normal(0.0, self.noise_std)
        return state + self.noise

    def before_step(self):
        if np.any(self.force):
            # PyBullet clears external forces after each step: reapply throughout pulse.
            position = self.client.getLinkState(self.robot_id, self.link_id,
                                                computeForwardKinematics=True)[0]
            self.client.applyExternalForce(self.robot_id, self.link_id,
                                           self.force.tolist(), position,
                                           self.client.WORLD_FRAME)

    def record(self, true_state):
        # Count executed pulse steps without storing trajectories.
        if self.event_id >= 0:
            self.applied_steps[self.event_id] += 1

    def finish(self, data, summary, dt):
        width = max(1, int(np.ceil(self.duration / dt - 1e-9)))
        return {}, dict(all_events_completed=bool(np.all(self.applied_steps == width)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_suite(name, targets, model, metadata, robustness=False):
    successes = 0
    evaluated = 0
    aborted = False
    for i, target in enumerate(targets):
        disturbance = ArmPerturbation(i) if robustness else None
        _, result = evaluation.run_episode(model, target, metadata, i, disturbance, verbose=False)
        aborted = result["aborted"]
        if aborted:
            break
        evaluated += 1
        successes += int(result["success"])
        print(f"{name} success rate: {100 * successes / evaluated:.1f}% "
              f"({successes}/{evaluated})", flush=True)
    return (successes / evaluated if evaluated else None), aborted


def main():
    if not TEST_GENERALIZATION and not TEST_ROBUSTNESS:
        print("No tests enabled. Set TEST_GENERALIZATION or TEST_ROBUSTNESS to True.")
        return
    if NUM_TARGETS <= 0 or not np.isfinite(PLAYBACK_SPEED) or PLAYBACK_SPEED <= 0:
        raise ValueError("NUM_TARGETS and PLAYBACK_SPEED must be positive")
    if TEST_ROBUSTNESS:
        ArmPerturbation(0)  # Validate before spending time on any simulation.
    evaluation.MODEL_PATH = MODEL_PATH
    evaluation.NUM_TARGETS = NUM_TARGETS
    evaluation.TEST_SEED = TEST_SEED
    evaluation.USE_GUI = USE_GUI
    evaluation.PLAYBACK_SPEED = PLAYBACK_SPEED
    evaluation.LOG_EXPERT_DIAGNOSTICS = False
    model, dataset_path, metadata = evaluation.load_model()
    suites = []
    if TEST_GENERALIZATION:
        suites.append(("generalization", sample_generalization_targets(metadata, NUM_TARGETS, TEST_SEED), False))
    if TEST_ROBUSTNESS:
        targets = evaluation.sample_test_targets(metadata, dataset_path)
        suites.extend((("robustness_baseline", targets, False), ("robustness", targets, True)))
    out = ROOT / "evaluation" / datetime.now(timezone.utc).strftime("performance_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, targets, robustness in suites:
        success_rate, aborted = run_suite(name, targets, model, metadata, robustness)
        reports[name] = dict(success_rate=success_rate)
        write_json(out / "summary.json", reports)
        if aborted:
            break


if __name__ == "__main__":
    main()
