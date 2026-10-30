"""Compare saved closed-loop EE-target distances; no training or simulation."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Edit these folders to compare other saved runs. Targets must match.
RUNS = {
    'baseline': ROOT / 'evaluation/final_20261029T174215_772977Z',
    'velocity_angle': ROOT / 'evaluation/final_20261028T184902_804523Z',
    'best_model': ROOT / 'evaluation/final_20261028T173354_733906Z',
}
ARM_WINDOW_S = 5.0


def load_run(folder):
    settings = json.loads((folder / 'settings.json').read_text())
    result = json.loads((folder / 'results.json').read_text())
    episodes = {}
    for ep in result['episodes']:
        key = tuple(ep['target'])
        if key in episodes:
            raise ValueError(f'Duplicate target in {folder}')
        with np.load(folder / f"episode_{ep['episode_id']:04d}.npz", allow_pickle=False) as log:
            time = np.asarray(log['time'], dtype=float)
            distance = np.linalg.norm(log['ee_pos'] - np.asarray(ep['target']), axis=1)
            if time.shape != distance.shape or not len(time) or np.any(np.diff(time) <= 0):
                raise ValueError(f'Invalid trace: {folder}, episode {ep["episode_id"]}')
            if not np.isfinite(distance).all():
                raise ValueError('Nonfinite EE distance')
            final = ep.get('final_error_m')
            if final is None or not np.isfinite(final):
                raise ValueError('Missing finite terminal EE error')
            if ep['duration_s'] > time[-1]:
                time = np.append(time, ep['duration_s'])
                distance = np.append(distance, final)
            initial = log['X'][0, :18].copy()
        episodes[key] = dict(summary=ep, time=time, distance=distance, initial=initial)
    return settings, episodes


def window_metrics(time, distance):
    duration = time[-1] - time[0]
    if duration <= 0:
        raise ValueError('Distance integration requires positive duration')
    # Explicit trapezoidal integration works with both old and new NumPy.
    dt = np.diff(time)
    average = np.sum((distance[:-1] + distance[1:]) * 0.5 * dt) / duration
    squared = distance**2
    mse = np.sum((squared[:-1] + squared[1:]) * 0.5 * dt) / duration
    return float(average), float(mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', metavar='NAME=FOLDER', help='Repeat to override RUNS')
    parser.add_argument('--window', type=float, default=ARM_WINDOW_S)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError('--window must be positive')
    runs = dict(item.split('=', 1) for item in args.run) if args.run else RUNS
    loaded = {name: load_run(Path(path)) for name, path in runs.items()}
    reference_settings, reference = next(iter(loaded.values()))
    ignored = {'model', 'model_sha256', 'source_config_changes'}
    for name, (settings, episodes) in loaded.items():
        if set(episodes) != set(reference):
            raise ValueError(f'{name}: target sets differ; compare runs with identical targets')
        if {k:v for k,v in settings.items() if k not in ignored} != {k:v for k,v in reference_settings.items() if k not in ignored}:
            raise ValueError(f'{name}: evaluation configurations differ')
        for target, ep in episodes.items():
            if not np.array_equal(ep['initial'], reference[target]['initial']):
                raise ValueError(f'{name}: initial state differs')
    # Use the SAME targets for fixed-duration Arm metrics across all models.
    eligible = [target for target in reference if all(
        episodes[target]['summary'].get('arm_start_time_s') is not None and
        episodes[target]['time'][-1] >= episodes[target]['summary']['arm_start_time_s'] + args.window
        for _, episodes in loaded.values())]
    rows, aggregates, curves = [], {}, {}
    grid = np.linspace(0, args.window, 401)
    for name, (settings, episodes) in loaded.items():
        model_rows, model_curves = [], []
        for target, ep in episodes.items():
            s, t, d = ep['summary'], ep['time'], ep['distance']
            avg, mse = window_metrics(t, d)
            row = dict(model=name, episode_id=s['episode_id'], success=s['success'],
                       final_distance_m=s['final_error_m'], min_distance_m=float(d.min()),
                       trajectory_mean_distance_m=avg, trajectory_distance_mse_m2=mse,
                       success_completion_time_s=s['duration_s'] if s['success'] else None,
                       arm_window_mean_distance_m=None, arm_window_distance_mse_m2=None)
            if target in eligible:
                curve = np.interp(grid + s['arm_start_time_s'], t, d)
                row['arm_window_mean_distance_m'], row['arm_window_distance_mse_m2'] = window_metrics(grid, curve)
                model_curves.append(curve)
            rows.append(row)
            model_rows.append(row)
        avg_field = lambda key: float(np.mean([r[key] for r in model_rows if r[key] is not None])) if any(r[key] is not None for r in model_rows) else None
        final_mse = float(np.mean([r['final_distance_m']**2 for r in model_rows]))
        aggregates[name] = dict(model=settings['model'], episodes=len(model_rows),
            success_rate=avg_field('success'), mean_final_distance_m=avg_field('final_distance_m'),
            final_distance_mse_m2=final_mse, final_distance_rmse_m=float(np.sqrt(final_mse)),
            mean_min_distance_m=avg_field('min_distance_m'),
            mean_trajectory_distance_m=avg_field('trajectory_mean_distance_m'),
            mean_trajectory_distance_mse_m2=avg_field('trajectory_distance_mse_m2'),
            mean_success_completion_time_s=avg_field('success_completion_time_s'),
            arm_window_episodes=len(eligible), arm_window_s=args.window,
            arm_window_mean_distance_m=avg_field('arm_window_mean_distance_m'),
            arm_window_distance_mse_m2=avg_field('arm_window_distance_mse_m2'),
            expert_ee_error_mse=None,
            expert_unavailable_reason='Logs contain expert shadow TORQUES, not an independent same-target expert EE rollout; expert EE-error MSE cannot be inferred.')
        if model_curves:
            curves[name] = np.asarray(model_curves)
    output = args.output or ROOT / 'outputs' / ('ee_comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    notes = [
        'Distances are 3D Euclidean EE-target distances in metres; distance MSE is mean squared Euclidean distance (not coordinate-averaged MSE).',
        'Trajectory averages are time-weighted within each episode, then equally weighted across episodes. Different stop times can affect whole-trajectory comparisons.',
        'Arm-window metrics and curves use an identical target subset with a complete fixed-duration trace in ALL runs; time is relative to each run Arm activation. No terminal padding.',
        'Successful completion times exclude failures and include the original minimum-time gate; report success rate alongside time.',
        'Initial observable states and recorded settings checked; unrecorded historical simulator/configuration changes cannot be excluded.',
    ]
    (output / 'metrics.json').write_text(json.dumps(dict(metrics=aggregates, notes=notes), indent=2)+'\n')
    for filename, records in [('episodes.csv', rows), ('comparison.csv', [dict(name=k, **v) for k,v in aggregates.items()])]:
        with (output / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, values in curves.items():
        axes[0].plot(grid, values.mean(axis=0)*1000, label=name)
        axes[1].plot(grid, np.mean(values**2, axis=0), label=name)
    axes[0].set_ylabel('Mean EE-target distance (mm)')
    axes[1].set_ylabel('Mean squared EE-target distance (m²)')
    for ax in axes:
        ax.set_xlabel('Time since Arm activation (s)'); ax.grid(alpha=.3)
        if curves:
            ax.legend()
    fig.suptitle(f'Paired targets with complete {args.window:g}s Arm traces: n={len(eligible)}')
    fig.savefig(output / 'ee_distance_comparison.png', dpi=200)
    plt.close(fig)
    print(json.dumps(aggregates, indent=2)); print(f'Saved: {output}')


if __name__ == '__main__':
    main()
