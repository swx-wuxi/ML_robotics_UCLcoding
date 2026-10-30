"""Evaluate saved single-frame staged NNs without training or simulation.

Train curves describe ORIGINAL successful training rows, not the stochastic
augmentation stream used by the optimizer. No historical epoch losses are inferred.
"""
import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# MODEL_PATH = ROOT / 'trained_model_weights/neural_network_bo_recovery_20261028T190338_972438Z.npz'
MODEL_PATH = ROOT / 'trained_model_weights/neural_network_model.npz'

DATASET_PATH = None  # None uses dataset_path embedded in the weights.
BATCH_SIZE = 8192

def training_helpers():
    """Reuse exact pure helpers without importing sklearn or simulator modules."""
    source = ROOT / 'neural_network.py'
    tree = ast.parse(source.read_text())
    wanted = {'load_dataset', 'split_indices', 'base_features', 'validate_nn', 'predict_nn'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('Required original training helpers are missing')
    scope = dict(np=np, DATA_BATCH_SIZE=50000,
                 SPLIT_NAMES={0: 'train', 1: 'validation', 2: 'test'},
                 BASE_COLUMNS=[0, 1, 2, 3, 18, 19, 20])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


def summarize(sse, sst, count):
    r2 = np.full(len(sse), np.nan)
    np.divide(sse, sst, out=r2, where=sst > 1e-12)
    r2 = 1 - r2
    return dict(rmse=np.sqrt(sse / count).tolist(),
                r2=[float(v) if np.isfinite(v) else None for v in r2],
                overall_rmse=float(np.sqrt(sse.mean() / count)),
                overall_r2=float(r2.mean()) if np.isfinite(r2).all() else None)


def evaluate(network, data, indices, stage, codes, limit, helpers):
    width = 4 if stage == 'base' else 7
    sse, mean, sst = (np.zeros(width, dtype=np.float64) for _ in range(3))
    count = 0
    if not len(indices):
        raise ValueError(f'Empty {stage} evaluation set')
    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]
        x = data['X'][idx]
        if stage == 'base':
            x = helpers['base_features'](x, codes[idx])
        truth = np.asarray(data['y'][idx, :4] if stage == 'base' else data['y'][idx, 4:11], dtype=np.float64)
        pred = helpers['predict_nn'](network, x)
        error = pred.astype(np.float64) - truth
        if not np.isfinite(error).all():
            raise ValueError('Non-finite prediction/error')
        sse += np.square(error).sum(axis=0)
        # Parallel variance update avoids cancellation in sum(y^2)-sum(y)^2/N.
        batch_mean = truth.mean(axis=0)
        delta = batch_mean - mean
        n = len(truth)
        sst += np.square(truth - batch_mean).sum(axis=0) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    return dict(samples=count, episodes=len(np.unique(data['episode_id'][indices])),
                **summarize(sse, sst, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--dataset', type=Path, default=DATASET_PATH)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    helpers = training_helpers()
    with np.load(args.model, allow_pickle=False) as archive:
        model = {k: archive[k] for k in archive.files}
    if str(model['model_type'].item()) != 'staged_neural_network':
        raise ValueError('Requires staged_neural_network with TWO neural networks')
    if int(model.get('arm_window_steps', 1)) != 1:
        raise ValueError('Temporal feature reconstruction is unsupported; refusing different features')
    base = {k[len('base_'):]: v for k, v in model.items() if k.startswith('base_nn_')}
    helpers['validate_nn'](base, n_features=8, n_outputs=4)
    helpers['validate_nn'](model, n_features=21, n_outputs=7)
    path = args.dataset or Path(str(model['dataset_path'].item()))
    data = helpers['load_dataset'](path)
    for key in ('feature_names', 'action_names'):
        if not np.array_equal(model[key], data[key]):
            raise ValueError(f'{key} differs between model and dataset')
    groups = helpers['split_indices'](data)
    with np.load(path, allow_pickle=False) as archive:
        done, phases = archive['nav_done_after'], archive['phase_after']
    if done.shape != (len(data['X']),) or done.dtype.kind != 'b':
        raise ValueError('Invalid/missing recorded stage labels')
    phase_map = {'align': 0, 'drive': 1, 'orient': 2, 'done': 0}
    if phases.shape != done.shape or not np.isin(phases, list(phase_map)).all():
        raise ValueError('Invalid/missing phase labels')
    codes = np.zeros(len(done), dtype=np.int8)
    for name, code in phase_map.items():
        codes[phases == name] = code
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    nav = metadata['base_params']
    wheel_limit = (nav['max_lin_vel'] + nav['max_ang_vel'] * nav['wheel_base_width'] / 2) / nav['wheel_radius']
    notes = [
        'All splits use recorded assignments, successful episodes only, and recorded post-controller stage labels.',
        'Train metrics cover original rows ONLY. Full augmented optimizer-stream training MSE is unavailable.',
        'Velocity/angle/recovery samples were mixed stochastically per epoch/batch. Weights do not store the realized row stream or RNG state; no replacement stream is generated.',
        'Validation/test contain original held-out rows only. Recovery validation is a separate dataset and is not merged here.',
        'No random split, fitting, simulation, augmentation generation or scaler refitting is performed.',
        'Two ReLU hidden layers, linear output, saved output inverse standardization; exact training predict_nn helper reused.',
        'Metrics use inverse-standardized, unclipped predictions, matching offline training report logic; no execution clipping.',
        'Overall RMSE=sqrt(mean squared error over all samples/dimensions); overall R2 is the unweighted mean of per-output R2. Constant-label R2 is null. Longer episodes have more weight.',
        'Plots compare final-model metrics across splits/output dimensions, never epochs.',
    ]
    result = dict(model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  dataset=str(path.resolve()), collection_seed=metadata.get('seed'),
                  notes=notes, augmented_training_metrics=dict(status='unavailable', reason=notes[2]),
                  units={'base': {'mse': '(rad/s)^2', 'rmse': 'rad/s'},
                         'arm': {'mse': '(N*m)^2', 'rmse': 'N*m'},
                         'standardized_raw': 'dimensionless'}, metrics={})
    for stage, network, mask, limit in (
            ('base', base, ~done, wheel_limit),
            ('arm', model, done, metadata['arm_params']['max_torque'])):
        result['metrics'][stage] = {}
        for split, idx in groups.items():
            print(f'Evaluating {stage}/{split} (original rows)', flush=True)
            result['metrics'][stage][split] = evaluate(network, data, idx[mask[idx]], stage, codes, limit, helpers)
    output = args.output or ROOT / 'outputs' / ('offline_metrics_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['network', 'split', 'scope', 'channel', 'samples', 'rmse', 'r2'])
        for stage, splits in result['metrics'].items():
            names = data['action_names'][:4] if stage == 'base' else data['action_names'][4:11]
            for split, item in splits.items():
                writer.writerow([stage, split, 'original_rows', 'overall', item['samples'], item['overall_rmse'], item['overall_r2']])
                for name, rmse, r2 in zip(names, item['rmse'], item['r2']):
                    writer.writerow([stage, split, 'original_rows', name, item['samples'], rmse, r2])
    plot(result, data['action_names'], output)
    print(f'Saved: {output}\nWARNING: augmented training-stream MSE unavailable; train=original rows only.')


def plot(result, action_names, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage, splits in result['metrics'].items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        names = action_names[:4] if stage == 'base' else action_names[4:11]
        for col, metric in enumerate(('rmse', 'r2')):
            ax = axes[col]
            for split, item in splits.items():
                ax.plot(names, item[metric], marker='o', label={'validation': 'valid'}.get(split, split))
            ax.set_title(f'{stage.upper()} {metric.upper()}')
            ax.set_ylabel(result['units'][stage]['rmse'] if metric == 'rmse' else 'R² (dimensionless)')
            ax.tick_params(axis='x', rotation=30)
            ax.legend()
            ax.grid(alpha=.3)
        fig.suptitle(f'{stage.upper()} final-model metrics (original rows; x = output channel)')
        fig.savefig(output / f'{stage}_rmse_r2.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()
