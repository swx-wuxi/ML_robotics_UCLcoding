"""Train a linear behaviour-cloning baseline on a completed multi-episode dataset.

Run: python linear.py. No simulator is started.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
# Set a specific dataset path, or leave None to select the latest completed batch.
DATA_PATH = None
MODEL_PATH = ROOT / "linear_model.npz"
BATCH_SIZE = 50000
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


def find_dataset():
    if DATA_PATH is not None:
        return Path(DATA_PATH)
    for folder in sorted((ROOT / "data").glob("expert_*"), reverse=True):
        metadata = folder / "metadata.json"
        dataset = folder / "dataset.npz"
        if metadata.is_file() and dataset.is_file():
            if json.loads(metadata.read_text())["episodes"] > 1:
                return dataset
    raise FileNotFoundError("No completed multi-episode dataset. Wait for collection "
                            "to finish or set DATA_PATH explicitly.")


def load_dataset(path):
    # Load only training and plotting fields, not all diagnostic joint arrays.
    keys = ("X", "y", "episode_id", "split", "episode_success", "time",
            "feature_names", "action_names")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    n = len(data["X"])
    if data["X"].shape != (n, 21) or data["y"].shape != (n, 11):
        raise ValueError("Expected X=(N, 21) and y=(N, 11)")
    for key in ("episode_id", "split", "episode_success", "time"):
        if data[key].shape != (n,):
            raise ValueError(f"Invalid shape for {key}")
    if data["feature_names"].shape != (21,) or data["action_names"].shape != (11,):
        raise ValueError("Invalid feature or action names")
    if not np.isin(data["split"], [0, 1, 2]).all():
        raise ValueError("Unknown split code")
    for start in range(0, n, BATCH_SIZE):
        if not (np.isfinite(data["X"][start:start + BATCH_SIZE]).all()
                and np.isfinite(data["y"][start:start + BATCH_SIZE]).all()):
            raise ValueError("Non-finite state or action")
    return data


def split_indices(data):
    """Keep successful episodes and preserve the collector's target-based split."""
    groups = {}
    episode_sets = []
    for code, name in SPLIT_NAMES.items():
        all_mask = data["split"] == code
        idx = np.flatnonzero(all_mask & data["episode_success"])
        episodes = np.unique(data["episode_id"][idx])
        attempted = len(np.unique(data["episode_id"][all_mask]))
        print(f"{name}: {len(episodes)}/{attempted} successful episodes, "
              f"{len(idx)} usable samples")
        if not len(idx):
            raise ValueError(f"No successful samples in {name}; check collection results")
        groups[name] = idx
        episode_sets.append(set(episodes.tolist()))
    for i in range(3):
        for j in range(i):
            if episode_sets[i] & episode_sets[j]:
                raise ValueError("An episode appears in more than one split")
    # Fixed targets are repeated per timestep; compare unique targets across splits.
    target_sets = [set(map(tuple, np.unique(data["X"][idx, 18:21], axis=0)))
                   for idx in groups.values()]
    for i in range(3):
        for j in range(i):
            if target_sets[i] & target_sets[j]:
                raise ValueError("A target appears in more than one split")
    return groups


def batches(indices):
    for start in range(0, len(indices), BATCH_SIZE):
        yield indices[start:start + BATCH_SIZE]


def fit_linear_regression(X, y, indices):
    """Fit standardized multi-output OLS using small incremental QR factorizations."""
    # Combine batch statistics without constructing a full training-data copy.
    count = 0
    mean = np.zeros(21)
    m2 = np.zeros(21)
    action_sum = np.zeros(11)
    for idx in batches(indices):
        block = X[idx]
        batch_mean = block.mean(axis=0)
        delta = batch_mean - mean
        total = count + len(idx)
        m2 += ((block - batch_mean)**2).sum(axis=0)
        m2 += delta**2 * count * len(idx) / total
        mean += delta * len(idx) / total
        count = total
        action_sum += y[idx].sum(axis=0)
    scale = np.sqrt(m2 / count)
    scale = np.where(scale < 1e-12, 1.0, scale)

    # QR compression preserves the least-squares minimizer without normal equations.
    R = np.empty((0, 22))
    projected_y = np.empty((0, 11))
    for idx in batches(indices):
        design = np.column_stack(((X[idx] - mean) / scale, np.ones(len(idx))))
        Q, R = np.linalg.qr(np.vstack((R, design)), mode="reduced")
        projected_y = Q.T @ np.vstack((projected_y, y[idx]))
    coefficients, _, rank, _ = np.linalg.lstsq(R, projected_y, rcond=None)
    print(f"Training design rank: {rank}/22")
    return dict(mean=mean, scale=scale, weights=coefficients[:-1],
                bias=coefficients[-1], action_mean=action_sum / count)


def predict(model, X):
    """Map one state (21,) or a batch (N, 21) to 11 action commands."""
    X = np.asarray(X, dtype=float)
    if X.ndim not in (1, 2) or X.shape[-1] != 21:
        raise ValueError("Expected state shape (21,) or (N, 21)")
    return ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]


def report_metrics(label, data, model, indices, constant=False):
    """Accumulate sample-weighted metrics in batches; do not mix action units."""
    count = 0
    mean = np.zeros(11)
    sst = np.zeros(11)
    sse = np.zeros(11)
    absolute_error = np.zeros(11)
    for idx in batches(indices):
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


def plot_predictions(data, model, groups):
    import matplotlib.pyplot as plt

    # Show one complete successful episode per split, not thousands of overlapping lines.
    fig, axes = plt.subplots(11, 3, figsize=(16, 22), sharex="col")
    for column, (label, indices) in enumerate(groups.items()):
        episode = data["episode_id"][indices[0]]
        idx = indices[data["episode_id"][indices] == episode]
        prediction = predict(model, data["X"][idx])
        time = data["time"][idx]
        for j in range(11):
            ax = axes[j, column]
            ax.plot(time, data["y"][idx, j], label="expert", color="black")
            ax.plot(time, prediction[:, j], label="linear", alpha=0.8)
            if column == 0:
                unit = "rad/s" if j < 4 else "N*m"
                ax.set_ylabel(f"{data['action_names'][j]} ({unit})")
            if j == 0:
                ax.set_title(f"{label}: episode {episode}")
            ax.grid(alpha=0.3)
        axes[-1, column].set_xlabel("Time (s)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "linear_predictions.png", dpi=150)
    plt.show()


def main():
    path = find_dataset()
    print(f"Dataset: {path.resolve()}")
    data = load_dataset(path)
    groups = split_indices(data)
    model = fit_linear_regression(data["X"], data["y"], groups["train"])
    metrics = {}
    for label, indices in groups.items():
        metrics[label] = report_metrics(label, data, model, indices)
        if label != "train":
            metrics[f"{label}_constant"] = report_metrics(
                f"{label} / training-mean baseline", data, model, indices, constant=True)
    np.savez_compressed(MODEL_PATH, **model, feature_names=data["feature_names"],
                        action_names=data["action_names"], dataset_path=str(path.resolve()))
    results = dict(dataset=str(path.resolve()),
                   evaluation="Offline action prediction on successful expert episodes only",
                   weighting="Per timestep; longer episodes contribute more samples",
                   metrics=metrics)
    (ROOT / "linear_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Model saved: {MODEL_PATH}")
    plot_predictions(data, model, groups)


if __name__ == "__main__":
    main()
