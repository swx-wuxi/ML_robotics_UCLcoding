"""Train a multi-output decision tree on the same dataset as linear.py.

Run: python decision_tree.py. No simulator is started.
Load: model = dict(np.load("decision_tree_model.npz", allow_pickle=False))
Infer: actions = predict(model, state)  # (21,) -> (11,)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import linear
from tree_policy import export_tree, predict_tree


ROOT = Path(__file__).resolve().parent
# None selects the latest completed batch, just as in linear.py.
DATA_PATH = None
MODEL_PATH = ROOT / "trained_model_weights"/"decision_tree_model.npz"
MAX_DEPTH = 32
MIN_SAMPLES_LEAF = 1
RANDOM_STATE = 42

def predict(model, X):
    """Map one state (21,) or a batch (N, 21) to 11 action commands."""
    if "estimator" not in model:
        return predict_tree(model, X)
    X = np.asarray(X, dtype=float)
    if X.ndim not in (1, 2) or X.shape[-1] != 21:
        raise ValueError("Expected state shape (21,) or (N, 21)")
    single = X.ndim == 1
    prediction = model["estimator"].predict(X.reshape(-1, 21))
    return prediction[0] if single else prediction

def fit_decision_tree(X, y, indices):
    """Fit all 11 outputs jointly, using only training samples.

    Trees need no input standardization. Unlike the incremental linear fit,
    sklearn's decision tree requires the training arrays to fit in memory.
    """
    estimator = DecisionTreeRegressor(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    estimator.fit(X[indices], y[indices])
    action_sum = np.zeros(11)
    for idx in linear.batches(indices):
        action_sum += y[idx].sum(axis=0)
    print(f"Tree depth: {estimator.get_depth()}, leaves: {estimator.get_n_leaves()}")
    return dict(estimator=estimator, action_mean=action_sum / len(indices))

def report_metrics(label, data, model, indices, constant=False):
    """Accumulate per-action errors without mixing wheel and arm units."""
    count = 0
    mean = np.zeros(11)
    sst = np.zeros(11)
    sse = np.zeros(11)
    absolute_error = np.zeros(11)
    for idx in linear.batches(indices):
        truth = data["y"][idx]
        prediction = model["action_mean"] if constant else predict(model, data["X"][idx])
        error = prediction - truth
        sse += (error**2).sum(axis=0)
        absolute_error += np.abs(error).sum(axis=0)
        batch_mean = truth.mean(axis=0)
        delta = batch_mean - mean
        total = count + len(idx)
        sst += ((truth - batch_mean)**2).sum(axis=0)
        sst += delta**2 * count * len(idx) / total
        mean += delta * len(idx) / total
        count = total
    rmse = np.sqrt(sse / count)
    mae = absolute_error / count
    r2 = np.full(11, np.nan)
    np.divide(sse, sst, out=r2, where=sst > 1e-12)
    r2 = 1 - r2
    print(f"\n{label}: {count} samples")
    print(f"{'Action':16s} {'RMSE':>12s} {'MAE':>12s} {'R2':>12s}")
    for name, a, b, c in zip(data["action_names"], rmse, mae, r2):
        print(f"{name:16s} {a:12.5g} {b:12.5g} {c:12.5g}")
    print("Wheel RMSE (rad/s):", np.sqrt(sse[:4].sum() / (count * 4)))
    print("Arm RMSE (N*m):", np.sqrt(sse[4:].sum() / (count * 7)))
    return dict(samples=count, rmse=rmse.tolist(), mae=mae.tolist(),
                r2=[float(v) if np.isfinite(v) else None for v in r2])

def main():
    path = Path(DATA_PATH) if DATA_PATH is not None else linear.find_dataset()
    print(f"Dataset: {path.resolve()}")
    data = linear.load_dataset(path)
    groups = linear.split_indices(data)
    model = fit_decision_tree(data["X"], data["y"], groups["train"])
    model.update(feature_names=data["feature_names"], action_names=data["action_names"],
                 dataset_path=str(path.resolve()))
    np.savez_compressed(MODEL_PATH, **export_tree(model))
    print(f"Model saved: {MODEL_PATH}")

    metrics = {}
    for label, indices in groups.items():
        metrics[label] = report_metrics(label, data, model, indices)
        if label != "train":
            metrics[f"{label}_constant"] = report_metrics(
                f"{label} / training-mean baseline", data, model, indices, constant=True)
    results = dict(dataset=str(path.resolve()), model=str(MODEL_PATH),
                   parameters=model["estimator"].get_params(),
                   evaluation="Offline action prediction on successful expert episodes only",
                   weighting="Per timestep; longer episodes contribute more samples",
                   metrics=metrics)
    (ROOT / "decision_tree_metrics.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
