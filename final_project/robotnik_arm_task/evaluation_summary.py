"""Summarize closest recorded EE-target distances and successful completion times."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EVALUATION_PATH = ROOT / 'evaluation/final_20261029T194112_590396Z'


def summarize(folder):
    results = json.loads((folder / 'results.json').read_text())
    rows, missing = [], []
    for ep in results['episodes']:
        episode_id = ep['episode_id']
        log = folder / f'episode_{episode_id:04d}.npz'
        minimum, first_entry = None, None
        if log.is_file():
            with np.load(log, allow_pickle=False) as data:
                if 'ee_error' in data:
                    distances = np.asarray(data['ee_error'], dtype=float)
                elif 'ee_pos' in data:
                    distances = np.linalg.norm(data['ee_pos'] - np.asarray(ep['target']), axis=1)
                else:
                    distances = np.array([])
                if distances.ndim != 1 or np.any(~np.isfinite(distances)) or np.any(distances < 0):
                    raise ValueError(f'Invalid distance trace: {log}')
                if len(distances):
                    minimum = float(distances.min())
                    if 'time' in data:
                        settings = json.loads((folder / 'settings.json').read_text())
                        hits = np.flatnonzero(distances < settings['reach_tolerance_m'])
                        if len(hits):
                            first_entry = float(data['time'][hits[0]])
        if minimum is None:
            missing.append(episode_id)
        else:
            # The terminal post-action state is measured separately from trace rows.
            final = ep.get('final_error_m')
            if final is not None and np.isfinite(final):
                minimum = min(minimum, float(final))
        rows.append(dict(episode_id=episode_id, success=ep['success'],
                         min_distance_m=minimum,
                         completion_time_s=ep['duration_s'] if ep['success'] else None,
                         first_recorded_tolerance_entry_s=first_entry,
                         episode_duration_s=ep['duration_s']))
    minima = [r['min_distance_m'] for r in rows if r['min_distance_m'] is not None]
    times = [r['completion_time_s'] for r in rows if r['completion_time_s'] is not None]
    mean = lambda values: float(np.mean(values)) if values else None
    summary = dict(evaluation=str(folder.resolve()), episodes=len(rows),
                   successes=len(times), success_rate=len(times) / len(rows) if rows else None,
                   distance_episodes=len(minima), missing_distance_episode_ids=missing,
                   mean_min_distance_m=mean(minima),
                   mean_min_distance_mm=mean([x * 1000 for x in minima]),
                   mean_success_completion_time_s=mean(times),
                   mean_all_episode_duration_s=mean([r['episode_duration_s'] for r in rows]),
                   notes=[
                       'Mean minimum distance: equal weight per episode; minimum over saved ee_error trace and final measured error. Sampling can miss between-sample minima.',
                       'Missing traces are excluded and listed; final error alone is not substituted for a trajectory minimum.',
                       'Completion time: duration_s for success=true episodes only, simulation seconds; includes base navigation and the evaluator minimum-time gate.',
                       'First tolerance entry is a separate diagnostic and does not imply official success or stable arrival.',
                       'No expert stop-message best error is used: its improvement threshold does not track the exact minimum.',
                   ])
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=EVALUATION_PATH,
                        help='One evaluation folder, or evaluation root to summarize all runs')
    args = parser.parse_args()
    folders = ([args.evaluation] if (args.evaluation / 'results.json').is_file()
               else sorted(p.parent for p in args.evaluation.glob('*/results.json')))
    if not folders:
        raise FileNotFoundError('No evaluation results.json found')
    for folder in folders:
        summary, rows = summarize(folder)
        (folder / 'distance_time_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        with (folder / 'distance_time_episodes.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode_id', 'success', 'min_distance_m',
                'completion_time_s', 'first_recorded_tolerance_entry_s', 'episode_duration_s'])
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
