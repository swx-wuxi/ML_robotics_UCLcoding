"""Explicit hybrid evaluation: frozen neural policy + bounded terminal feedback.

Run: python feedback_improve.py [--model path/to/model.npz]. No training is done.
The existing evaluator supplies the same state, dynamics, region and stop rules.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

import final_evaluation as evaluation


@dataclass(frozen=True)
class FeedbackSettings:
    kp: float = 80.0                     # N/m
    kd: float = 12.0                     # N*s/m
    ki: float = 30.0                     # N/(m*s)
    enable_distance_m: float = 0.08
    full_distance_m: float = 0.03
    integral_distance_m: float = 0.03
    integral_speed_m_s: float = 0.03
    max_force_n: float = 5.0
    max_integral_force_n: float = 3.0
    max_correction_nm: float = 2.0       # Per joint, before the existing total torque clip.
    integral_release_s: float = 0.5

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError('Feedback settings must be finite')
        if any(values[name] < 0 for name in ('kp', 'kd', 'ki')):
            raise ValueError('Feedback gains must be nonnegative')
        if any(value <= 0 for key, value in values.items() if key not in ('kp', 'kd', 'ki')):
            raise ValueError('Feedback distances, speeds and limits must be positive')
        if not self.integral_distance_m <= self.full_distance_m < self.enable_distance_m:
            raise ValueError('Require integral distance <= full distance < enable distance')


def limit_norm(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


class TerminalFeedback:
    """Task-space PI + damping, with a fresh integrator in each episode."""

    def __init__(self, metadata, settings):
        self.settings = settings
        self.torque_limit = float(metadata['arm_params']['max_torque'])
        self.limit_margin = float(metadata['arm_params']['limit_margin'])
        self.integral_force = np.zeros(3)
        self.trace = []

    def correct(self, nominal, dynamics, q_pin, X, ee, frame, lower, upper, dt):
        nominal = np.asarray(nominal, dtype=float)
        if nominal.shape != (7,):
            raise ValueError('Expected seven neural arm torques')
        if not np.isfinite(nominal).all():
            return nominal  # Keep the original evaluator's non_finite_prediction handling.
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Expected a positive simulation timestep')
        cfg = self.settings
        error = np.asarray(X[18:21]) - ee
        distance = float(np.linalg.norm(error))
        ramp = np.clip((cfg.enable_distance_m - distance)
                       / (cfg.enable_distance_m - cfg.full_distance_m), 0.0, 1.0)
        weight = float(ramp * ramp * (3.0 - 2.0 * ramp))
        correction = np.zeros(7)
        limited = False
        if weight == 0.0:
            self.integral_force *= np.exp(-dt / cfg.integral_release_s)
        else:
            # Same full measured configuration and arm Jacobian convention as the expert.
            # This computes feedback geometry; no expert torque/reference is queried.
            pin = evaluation.task.pin
            pin.computeJointJacobians(dynamics.pin_model, dynamics.pin_data, q_pin)
            J = np.asarray(pin.computeFrameJacobian(
                dynamics.pin_model, dynamics.pin_data, q_pin, frame,
                pin.LOCAL_WORLD_ALIGNED))[:3, 6:13]
            if J.shape != (3, 7) or not np.isfinite(J).all():
                raise ValueError('Invalid arm position Jacobian')
            velocity = J @ X[11:18]  # Base is already stopped by the original supervisor.
            proposal = self.integral_force.copy()
            if distance <= cfg.integral_distance_m and np.linalg.norm(velocity) <= cfg.integral_speed_m_s:
                proposal = limit_norm(proposal + cfg.ki * weight * error * dt,
                                      cfg.max_integral_force_n)
            has_limits = lower < upper

            def residual(integral):
                raw_force = cfg.kp * error - cfg.kd * velocity + integral
                force = limit_norm(raw_force, cfg.max_force_n)
                raw_torque = weight * (J.T @ force)
                delta = np.clip(raw_torque, -cfg.max_correction_nm, cfg.max_correction_nm)
                pushing_limit = has_limits & (
                    ((X[4:11] <= lower + self.limit_margin) & (delta < 0))
                    | ((X[4:11] >= upper - self.limit_margin) & (delta > 0)))
                delta = np.where(pushing_limit, 0.0, delta)
                saturated = (np.linalg.norm(raw_force) > cfg.max_force_n
                             or np.any(np.abs(raw_torque) > cfg.max_correction_nm)
                             or pushing_limit.any()
                             or np.any(np.abs(nominal + delta) > self.torque_limit))
                return delta, bool(saturated)

            correction, limited = residual(proposal)
            # Conditional integration: do not accumulate error against force, joint,
            # correction-torque or actuator limits. P and damping remain active.
            if limited:
                correction, _ = residual(self.integral_force)
            else:
                self.integral_force = proposal
        self.trace.append((nominal.copy(), weight, self.integral_force.copy(), limited))
        return nominal + correction

    def annotate(self, data, summary):
        summary['controller'] += ' + bounded terminal Cartesian PI/damping feedback'
        summary['arm_feedback'] = asdict(self.settings)
        if 'X' not in data:
            return
        arm_rows = np.flatnonzero(data['control_phase'] == 'arm')
        # A prediction can precede an unexecuted final command if evaluation aborts.
        if len(self.trace) < len(arm_rows):
            raise RuntimeError('Feedback trace is shorter than the executed arm trajectory')
        nominal = data['raw_actions'][:, 4:11].copy()
        weights = np.zeros(len(data['X']))
        integral = np.zeros((len(data['X']), 3))
        limited = np.zeros(len(data['X']), dtype=bool)
        for row, record in zip(arm_rows, self.trace):
            nominal[row], weights[row], integral[row], limited[row] = record
        data['arm_nominal_torque'] = nominal
        data['terminal_feedback_torque'] = data['raw_actions'][:, 4:11] - nominal
        data['terminal_feedback_effective_torque'] = (
            data['actions'][:, 4:11] - np.clip(nominal, -self.torque_limit, self.torque_limit))
        data['terminal_feedback_weight'] = weights
        data['terminal_feedback_integral_force'] = integral
        data['terminal_feedback_limited'] = limited
        summary['terminal_feedback_active_arm_fraction'] = (
            float(np.mean(weights[arm_rows] > 0)) if len(arm_rows) else 0.0)
        summary['terminal_feedback_max_abs_effective_torque_nm'] = float(
            np.max(np.abs(data['terminal_feedback_effective_torque'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=None,
                        help='Frozen NPZ; defaults to current final_evaluation.MODEL_PATH')
    parser.add_argument('--kp', type=float, default=80.0)
    parser.add_argument('--kd', type=float, default=12.0)
    parser.add_argument('--ki', type=float, default=30.0)
    parser.add_argument('--max-correction', type=float, default=2.0, metavar='NM')
    args = parser.parse_args()
    cfg = FeedbackSettings(kp=args.kp, kd=args.kd, ki=args.ki,
                           max_correction_nm=args.max_correction)
    if args.model is not None:
        evaluation.MODEL_PATH = args.model.resolve()
    if not evaluation.MODEL_PATH.is_file():
        parser.error(f'Model does not exist: {evaluation.MODEL_PATH}')
    if not evaluation.CLIP_COMMANDS:
        parser.error('This hybrid variant requires the existing CLIP_COMMANDS=True setting')
    settings = dict(method='Frozen neural policy + terminal Cartesian PI/damping residual',
        parameters=asdict(cfg), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference='https://arxiv.org/abs/1812.03201',
        method_scope='Feedback/learning composition inspired by residual control; no online RL',
        training=False, interpretation='Hybrid controller success, not pure neural-network success')
    print(f'Hybrid evaluation; frozen model: {evaluation.MODEL_PATH}', flush=True)
    print('Same targets, navigation, region and success rules as final_evaluation.py.', flush=True)
    evaluation.main(arm_feedback_factory=lambda metadata: TerminalFeedback(metadata, cfg),
                    arm_feedback_settings=settings)


if __name__ == '__main__':
    main()
