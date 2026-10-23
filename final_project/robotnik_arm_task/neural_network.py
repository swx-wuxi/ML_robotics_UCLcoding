"""Lecture 6 style MLP baseline: keep the base tree, replace arm torque tree.

Run: python neural_network.py, then select neural_network_model.npz in the evaluator.
Uses sklearn's Linear/ReLU MLP and Adam; evaluation requires only NumPy.
"""
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import linear
from nn_policy import predict_nn, validate_nn
from tree_policy import TREE_KEYS, validate_tree

ROOT = Path(__file__).resolve().parent
DATA_PATH = None
BASE_MODEL_PATH = ROOT / 'trained_model_weights' / 'decision_tree_model.npz'
MODEL_PATH = ROOT / 'trained_model_weights' / 'neural_network_model.npz'
HIDDEN_LAYERS = (128, 128)
BATCH_SIZE = 4096
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 10
EARLY_STOPPING = True  # Run all epochs while retaining the best validation weights.
MIN_DELTA = 1e-5
RANDOM_STATE = 42


def export_nn(estimator, xs, ys):
    model = dict(nn_x_mean=xs.mean_.astype(np.float32),
                 nn_x_scale=xs.scale_.astype(np.float32),
                 nn_y_mean=ys.mean_.astype(np.float32),
                 nn_y_scale=ys.scale_.astype(np.float32))
    for i, (w, b) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
        model[f'nn_w{i}'], model[f'nn_b{i}'] = w.copy(), b.copy()
    return model


def fit_nn(X, y, groups):
    # CORE CHANGE: fit normalization on TRAIN ONLY, never on validation/test.
    xs, ys = StandardScaler(), StandardScaler()
    for idx in linear.batches(groups['train']):
        xs.partial_fit(X[idx])
        ys.partial_fit(y[idx])
    net = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation='relu',
                       solver='adam', learning_rate_init=LEARNING_RATE,
                       alpha=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                       shuffle=False, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    best, best_loss, stale, history = None, np.inf, 0, []
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(groups['train'])
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            net.batch_size = len(idx)
            net.partial_fit(xs.transform(X[idx]).astype(np.float32),
                            ys.transform(y[idx]).astype(np.float32))
        # CORE CHANGE: validation chooses the checkpoint; test is used only afterward.
        total, count = 0.0, 0
        for idx in linear.batches(groups['validation']):
            error = net.predict(xs.transform(X[idx]).astype(np.float32)) - ys.transform(y[idx])
            total += np.square(error).sum()
            count += error.size
        loss = float(total / count)
        if not np.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        history.append(dict(epoch=epoch + 1, validation_standardized_mse=loss))
        print(f'Epoch {epoch + 1}/{MAX_EPOCHS}: validation MSE={loss:.6g}', flush=True)
        if loss < best_loss - MIN_DELTA:
            best, best_loss, stale = copy.deepcopy(net), loss, 0
        else:
            stale += 1
        if EARLY_STOPPING and stale >= PATIENCE:
            print('Early stopping; restoring best validation checkpoint.')
            break
    if best is None:
        raise ValueError('No fitted NN checkpoint')
    print(f'Training finished after {len(history)} epochs; restoring best validation checkpoint '
          f'(standardized MSE={best_loss:.6g}).')
    return export_nn(best, xs, ys), history


def report(model, X, y, indices, names):
    sse, absolute = np.zeros(7), np.zeros(7)
    scaler = StandardScaler()
    for idx in linear.batches(indices):
        truth = y[idx]
        error = predict_nn(model, X[idx]) - truth
        sse += np.square(error).sum(0)
        absolute += np.abs(error).sum(0)
        scaler.partial_fit(truth)
    rmse, mae = np.sqrt(sse / len(indices)), absolute / len(indices)
    sst = scaler.var_ * len(indices)
    ratio = np.full(7, np.nan)
    np.divide(sse, sst, out=ratio, where=sst > 1e-12)
    r2 = 1 - ratio
    for name, a, b, c in zip(names, rmse, mae, r2):
        print(f'{name:16s} RMSE={a:.5g} MAE={b:.5g} R2={c:.5g}')
    return dict(samples=len(indices), rmse=rmse.tolist(), mae=mae.tolist(),
                r2=[float(v) if np.isfinite(v) else None for v in r2])


def main():
    # Reuse exactly the trained navigation tree and its dataset for fair comparison.
    with np.load(BASE_MODEL_PATH, allow_pickle=False) as archive:
        if str(archive['model_type'].item()) != 'staged_decision_tree':
            raise ValueError('Train a staged base tree first')
        source = {key: archive[f'base_{key}'] for key in TREE_KEYS}
        source_dataset = Path(str(archive['dataset_path'].item()))
        feature_names, action_names = archive['feature_names'], archive['action_names']
    validate_tree(source, 8, 4)
    path = Path(DATA_PATH) if DATA_PATH is not None else source_dataset
    if path.resolve() != source_dataset.resolve():
        raise ValueError('Use the same dataset as the base tree to preserve target splits')
    print(f'Dataset: {path}', flush=True)
    data = linear.load_dataset(path)
    if not (np.array_equal(data['feature_names'], feature_names) and
            np.array_equal(data['action_names'], action_names)):
        raise ValueError('Dataset/model feature or action order differs')
    groups = linear.split_indices(data)
    with np.load(path, allow_pickle=False) as archive:
        active = archive['nav_done_after']
    if active.shape != (len(data['X']),) or active.dtype.kind != 'b':
        raise ValueError('Invalid arm stage labels')
    groups = {name: idx[active[idx]] for name, idx in groups.items()}
    for name, idx in groups.items():
        if not len(idx):
            raise ValueError(f'No reaching samples in {name}')
        print(f'arm/{name}: {len(idx)} samples', flush=True)
    # Baseline: current 21-dimensional state -> current seven joint torques.
    model, history = fit_nn(data['X'], data['y'][:, 4:], groups)
    model.update(model_type=np.asarray('staged_neural_network'),
                 feature_names=feature_names, action_names=action_names,
                 dataset_path=np.asarray(str(path.resolve())),
                 arm_window_steps=np.asarray(1), arm_window_stride=np.asarray(1))
    model.update({f'base_{key}': value for key, value in source.items()})
    validate_nn(model)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, **model)
    metrics = {}
    for name, idx in groups.items():
        print(f'arm/{name}')
        metrics[name] = report(model, data['X'], data['y'][:, 4:], idx, action_names[4:])
    output = ROOT / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'neural_network_metrics.json').write_text(json.dumps(dict(
        model=str(MODEL_PATH), dataset=str(path.resolve()), architecture=[21, *HIDDEN_LAYERS, 7],
        activation='relu', optimizer='adam', learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        early_stopping=EARLY_STOPPING,
        weight_decay=WEIGHT_DECAY, seed=RANDOM_STATE, history=history, metrics=metrics), indent=2)+'\n')
    print(f'Model saved: {MODEL_PATH}')


if __name__ == '__main__':
    main()
