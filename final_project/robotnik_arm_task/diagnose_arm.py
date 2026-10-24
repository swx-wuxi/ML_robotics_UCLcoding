"""Plot arm behaviour around closest approach; no simulator or retraining needed.

python diagnose_arm.py
python diagnose_arm.py --episode evaluation/final_.../episode_0000.npz --window 1.0
Shadow expert torques are available only in newly recorded evaluations.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def diagnose(path, window=1.0):
    with np.load(path, allow_pickle=False) as archive:
        d = dict(archive)
    t, X, error = d['time'], d['X'], d['ee_error']
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Need at least two samples with increasing time')
    active = d.get('control_phase', np.full(len(t), 'arm')) == 'arm'
    if not active.any():
        raise ValueError('No arm-reaching samples in this episode')
    candidates = np.flatnonzero(active)
    closest = candidates[np.argmin(error[candidates])]
    center = t[closest]
    selected = active & (np.abs(t - center) <= window)
    qd = X[:, 11:18]
    raw, applied = d['raw_actions'][:, 4:11], d['actions'][:, 4:11]
    # Delta is N*m per simulation step, NOT torque derivative.
    delta = np.vstack((np.zeros((1, 7)), np.diff(raw, axis=0)))
    distance_rate = np.gradient(error, t)
    summary = dict(episode=str(path.resolve()), closest_time_s=float(center),
                   closest_distance_m=float(error[closest]),
                   last_recorded_distance_m=float(error[-1]),
                   joint_velocity_at_closest_rad_s=qd[closest].tolist(),
                   max_abs_joint_velocity_in_window_rad_s=np.max(np.abs(qd[selected]), axis=0).tolist(),
                   max_abs_torque_change_per_step_Nm=np.max(np.abs(delta[selected]), axis=0).tolist(),
                   torque_clipping_fraction_in_window=float(np.mean(raw[selected] != applied[selected])),
                   expert_comparison='Shadow reference evolved along learned states; not an expert rollout')
    out = path.parent / (path.stem + '_diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, error * 100, label='EE distance')
    axes[0].axhline(1, color='green', linestyle='--', label='1 cm reference')
    axes[0].set_ylabel('Distance (cm)')
    axes[1].plot(t, distance_rate)
    axes[1].axhline(0, color='black', linewidth=.6)
    axes[1].set_ylabel('Distance rate (m/s)')
    axes[2].plot(t, qd)
    axes[2].set_ylabel('Joint speed (rad/s)')
    axes[2].set_xlabel('Time (s)')
    axes[0].legend()
    for ax in axes:
        ax.axvline(center, color='red', linestyle='--')
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'overview.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    expert = d.get('expert_shadow_torque')
    for j in range(7):
        axes[j, 0].plot(t[selected], qd[selected, j], label='Measured velocity')
        axes[j, 0].set_ylabel(f'Joint {j+1} (rad/s)')
        axes[j, 1].plot(t[selected], raw[selected, j], label='Policy raw')
        axes[j, 1].plot(t[selected], applied[selected, j], '--', label='Applied')
        if expert is not None:
            axes[j, 1].plot(t[selected], expert[selected, j], label='Expert shadow')
        axes[j, 1].set_ylabel('Torque (N*m)')
        for ax in axes[j]:
            ax.axvline(center, color='red', linestyle='--'); ax.grid(alpha=.3)
    axes[0, 0].legend(); axes[0, 1].legend()
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(out / 'closest_approach.png', dpi=150); plt.close(fig)
    if expert is not None:
        summary['policy_vs_shadow_rmse_Nm'] = np.sqrt(np.mean((raw[selected]-expert[selected])**2, axis=0)).tolist()
    else:
        summary['expert_comparison'] = 'Unavailable: rerun evaluation with LOG_EXPERT_DIAGNOSTICS=True'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2)); print(f'Plots saved: {out}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--window', type=float, default=1.0, help='Seconds before/after closest approach')
    args = parser.parse_args()
    if args.window <= 0:
        parser.error('--window must be positive')
    path = args.episode
    if path is None:
        episodes = sorted((ROOT / 'evaluation').glob('final_*/episode_*.npz'))
        if not episodes:
            parser.error('No evaluation episodes found')
        # Prefer an actual reaching episode from the latest run.
        for candidate in reversed(episodes):
            with np.load(candidate, allow_pickle=False) as a:
                if 'control_phase' not in a or np.any(a['control_phase'] == 'arm'):
                    path = candidate
                    break
        if path is None:
            parser.error('No arm-reaching episode found')
    diagnose(path, args.window)


if __name__ == '__main__':
    main()
