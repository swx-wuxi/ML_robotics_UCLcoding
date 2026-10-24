"""Train separate navigation and reaching decision trees on the same dataset as py.

Run: python decision_tree.py. No simulator is started.
Load: model = dict(np.load("decision_tree_model.npz", allow_pickle=False))
Evaluate: python evaluation_final.py (stage supervisor + arm posture hold).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor



ROOT = Path(__file__).resolve().parent
# None selects the latest completed batch, just as in py.
DATA_PATH = None
MODEL_PATH = ROOT / "trained_model_weights"/"decision_tree_model.npz"
MAX_DEPTH = 32
MIN_SAMPLES_LEAF = 1
RANDOM_STATE = 42
# CORE CHANGE: six states sampled every 20 steps, including the current state.
# At dt=0.001 s this covers 0.100 s; the target is the CURRENT action, not a future action.
ARM_WINDOW_STEPS = 6
ARM_WINDOW_STRIDE = 20

# Self-contained data/inference helpers; no auxiliary policy module imports.
DATA_BATCH_SIZE = 50000
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}

PHASE_CODES = {"align": 0, "drive": 1, "orient": 2}
BASE_COLUMNS = [0, 1, 2, 3, 18, 19, 20]
TREE_KEYS = ("children_left", "children_right", "feature", "threshold", "values")

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
    for start in range(0, n, DATA_BATCH_SIZE):
        if not (np.isfinite(data["X"][start:start + DATA_BATCH_SIZE]).all()
                and np.isfinite(data["y"][start:start + DATA_BATCH_SIZE]).all()):
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
    for start in range(0, len(indices), DATA_BATCH_SIZE):
        yield indices[start:start + DATA_BATCH_SIZE]

def export_tree(model):
    tree = model["estimator"].tree_
    return dict(model_type=np.asarray("decision_tree"),
                children_left=tree.children_left, children_right=tree.children_right,
                feature=tree.feature, threshold=tree.threshold,
                values=tree.value[:, :, 0], action_mean=model["action_mean"],
                feature_names=model["feature_names"], action_names=model["action_names"],
                dataset_path=np.asarray(model["dataset_path"]))

def base_features(X, phase):
    """Exclude arm states so arm deviations cannot change navigation predictions."""
    X = np.asarray(X)
    return np.concatenate((X[..., BASE_COLUMNS],
                           np.broadcast_to(np.asarray(phase), X.shape[:-1])[..., None]), axis=-1)

class TemporalFeatures:
    """Build requested causal windows without expanding the entire dataset."""

    def __init__(self, X, episode_id, active, steps, stride):
        if steps < 1 or stride < 1:
            raise ValueError("Window steps and stride must be positive")
        self.X, self.steps, self.stride = X, steps, stride
        # CORE CHANGE: reset at episode AND stage boundaries; never use future data.
        reset = np.r_[True, (episode_id[1:] != episode_id[:-1]) |
                      (active[1:] != active[:-1])]
        self.starts = np.maximum.accumulate(np.where(reset, np.arange(len(X)), 0))

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        offsets = np.arange(self.steps - 1, -1, -1) * self.stride
        rows = np.maximum(indices[..., None] - offsets, self.starts[indices][..., None])
        return self.X[rows].astype(np.float32).reshape(*indices.shape, self.steps * 21)


def predict(model, X):
    """Predict one stage's outputs; the runtime supervisor selects the stage."""
    X = np.asarray(X)
    single = X.ndim == 1
    result = model["estimator"].predict(np.atleast_2d(X))
    return result[0] if single else result


def fit_decision_tree(X, y, indices):
    estimator = DecisionTreeRegressor(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    estimator.fit(X[indices], y[indices])
    action_sum = np.zeros(y.shape[1])
    for idx in batches(indices):
        action_sum += y[idx].sum(axis=0)
    print(f"Tree depth: {estimator.get_depth()}, leaves: {estimator.get_n_leaves()}")
    return dict(estimator=estimator, action_mean=action_sum / len(indices))


def report_metrics(label, data, model, indices, constant=False):
    """Accumulate per-action errors without mixing wheel and arm units."""
    count = 0
    mean = np.zeros(data["y"].shape[1])
    sst = np.zeros(data["y"].shape[1])
    sse = np.zeros(data["y"].shape[1])
    absolute_error = np.zeros(data["y"].shape[1])
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
    r2 = np.full(data["y"].shape[1], np.nan)
    np.divide(sse, sst, out=r2, where=sst > 1e-12)
    r2 = 1 - r2
    print(f"\n{label}: {count} samples")
    print(f"{'Action':16s} {'RMSE':>12s} {'MAE':>12s} {'R2':>12s}")
    for name, a, b, c in zip(data["action_names"], rmse, mae, r2):
        print(f"{name:16s} {a:12.5g} {b:12.5g} {c:12.5g}")
    return dict(samples=count, rmse=rmse.tolist(), mae=mae.tolist(),
                r2=[float(v) if np.isfinite(v) else None for v in r2])

def main():
    path = Path(DATA_PATH) if DATA_PATH is not None else find_dataset()
    print(f"Dataset: {path.resolve()}")
    data = load_dataset(path)
    groups = split_indices(data)
    # CORE CHANGE: use the recorded post-controller flag, which is exactly
    # reach_active for this action. Zero wheel speed is NOT a phase label.
    with np.load(path, allow_pickle=False) as archive:
        done = archive["nav_done_after"]
        phases = archive["phase_after"]
    if done.shape != (len(data["X"]),) or done.dtype.kind != "b":
        raise ValueError("Invalid nav_done_after labels")
    if phases.shape != done.shape or not np.isin(phases, list(PHASE_CODES)).all():
        raise ValueError("Invalid navigation phase labels")
    codes = np.zeros(len(done), dtype=np.int8)
    for name, code in PHASE_CODES.items():
        codes[phases == name] = code

    # CORE CHANGE: two independently fitted trees share the original target split.
    # Base: only navigation samples, seven base/target fields + phase -> 4 wheels.
    # Arm: reaching samples, causal history of 21 fields -> current 7 joint torques.
    saved = dict(model_type=np.asarray("staged_decision_tree"),
                 feature_names=data["feature_names"], action_names=data["action_names"],
                 dataset_path=np.asarray(str(path.resolve())))
    consecutive = data["episode_id"][1:] == data["episode_id"][:-1]
    intervals = np.diff(data["time"])[consecutive]
    if (not len(intervals) or np.any(intervals <= 0) or
            not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-9)):
        raise ValueError("Temporal windows require a consistent positive timestep")
    saved.update(arm_window_steps=np.asarray(ARM_WINDOW_STEPS),
                 arm_window_stride=np.asarray(ARM_WINDOW_STRIDE),
                 sample_time_step=np.asarray(float(intervals[0])))
    print(f"Arm history: {ARM_WINDOW_STEPS} states, "
          f"{(ARM_WINDOW_STEPS-1)*ARM_WINDOW_STRIDE*intervals[0]:.3f} s span")
    metrics = {}
    parameters = None
    for stage, mask, outputs in (("base", ~done, slice(0, 4)),
                                 ("arm", done, slice(4, 11))):
        stage_data = dict(X=(base_features(data["X"], codes) if stage == "base" else TemporalFeatures(
                              data["X"], data["episode_id"], done, ARM_WINDOW_STEPS, ARM_WINDOW_STRIDE)),
                          y=data["y"][:, outputs], action_names=data["action_names"][outputs])
        stage_groups = {name: idx[mask[idx]] for name, idx in groups.items()}
        for name, idx in stage_groups.items():
            print(f"{stage}/{name}: {len(idx)} samples")
            if not len(idx):
                raise ValueError(f"No {stage} samples in {name}")
        model = fit_decision_tree(stage_data["X"], stage_data["y"], stage_groups["train"])
        parameters = model["estimator"].get_params()
        model.update(feature_names=data["feature_names"], action_names=stage_data["action_names"],
                     dataset_path=str(path.resolve()))
        exported = export_tree(model)
        saved.update({f"{stage}_{key}": exported[key] for key in TREE_KEYS})
        for name, idx in stage_groups.items():
            metrics[f"{stage}_{name}"] = report_metrics(f"{stage}/{name}", stage_data, model, idx)
            if name != "train":
                metrics[f"{stage}_{name}_constant"] = report_metrics(
                    f"{stage}/{name} baseline", stage_data, model, idx, constant=True)
        del model, stage_data
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, **saved)
    results = dict(dataset=str(path.resolve()), model=str(MODEL_PATH), parameters=parameters,
                   evaluation="Stage-specific offline errors on successful expert episodes",
                   arm_window_steps=ARM_WINDOW_STEPS, arm_window_stride=ARM_WINDOW_STRIDE,
                   window_order="oldest to newest; pad with first reaching state",
                   phase_label="nav_done_after; base phase input uses phase_after",
                   metrics=metrics)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs"/"decision_tree_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Two-tree model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
