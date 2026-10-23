"""NumPy-only storage and inference for an exported sklearn decision tree."""

from collections import deque

import numpy as np


def export_tree(model):
    tree = model["estimator"].tree_
    return dict(model_type=np.asarray("decision_tree"),
                children_left=tree.children_left, children_right=tree.children_right,
                feature=tree.feature, threshold=tree.threshold,
                values=tree.value[:, :, 0], action_mean=model["action_mean"],
                feature_names=model["feature_names"], action_names=model["action_names"],
                dataset_path=np.asarray(model["dataset_path"]))


def validate_tree(model, n_features=21, n_outputs=11):
    n = len(model["feature"])
    if n == 0:
        raise ValueError("Empty decision tree")
    for key in ("children_left", "children_right", "feature", "threshold"):
        if model[key].shape != (n,) or not np.isfinite(model[key]).all():
            raise ValueError(f"Invalid tree parameter: {key}")
    for key in ("children_left", "children_right", "feature"):
        if model[key].dtype.kind not in "iu":
            raise ValueError(f"Tree indices must be integers: {key}")
    if model["values"].shape != (n, n_outputs) or not np.isfinite(model["values"]).all():
        raise ValueError("Invalid tree values")
    left, right = model["children_left"], model["children_right"]
    leaf = (left == -1) & (right == -1)
    internal = ~leaf
    # sklearn numbers every child after its parent; this also rules out cycles.
    parents = np.arange(n)[internal]
    for children in (left[internal], right[internal]):
        if np.any(children <= parents) or np.any(children >= n):
            raise ValueError("Invalid tree child index")
    if np.any(model["feature"][internal] < 0) or np.any(model["feature"][internal] >= n_features):
        raise ValueError("Invalid tree feature index")


def predict_tree(model, X, n_features=21):
    # sklearn converts input features to float32 before testing thresholds.
    X = np.asarray(X, dtype=np.float32)
    if X.ndim not in (1, 2) or X.shape[-1] != n_features:
        raise ValueError(f"Expected {n_features} input features")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite state")
    single = X.ndim == 1
    X = X.reshape(-1, n_features)
    nodes = np.zeros(len(X), dtype=np.intp)
    left, right = model["children_left"], model["children_right"]
    while True:
        rows = np.flatnonzero(left[nodes] != -1)
        if not len(rows):
            break
        current = nodes[rows]
        go_left = X[rows, model["feature"][current]] <= model["threshold"][current]
        nodes[rows] = np.where(go_left, left[current], right[current])
    result = model["values"][nodes]
    return result[0] if single else result


PHASE_CODES = {"align": 0, "drive": 1, "orient": 2}
BASE_COLUMNS = [0, 1, 2, 3, 18, 19, 20]
TREE_KEYS = ("children_left", "children_right", "feature", "threshold", "values")


def base_features(X, phase):
    """Exclude arm states so arm deviations cannot change navigation predictions."""
    X = np.asarray(X)
    return np.concatenate((X[..., BASE_COLUMNS],
                           np.broadcast_to(np.asarray(phase), X.shape[:-1])[..., None]), axis=-1)


def stage_tree(model, stage):
    return {key: model[f"{stage}_{key}"] for key in TREE_KEYS}


def update_navigation(phase, position, yaw, goal, bearing, pos_tol, ang_tol):
    """Geometry-only phase supervisor; it never computes expert wheel commands."""
    delta = goal - np.asarray(position)[:2]
    distance = np.linalg.norm(delta)
    heading_error = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2*np.pi) - np.pi
    if phase == "align" and (distance < pos_tol or abs(heading_error) < ang_tol):
        phase = "drive" if distance >= pos_tol else "orient"
    if phase == "drive" and distance < pos_tol:
        phase = "orient"
    # Unlike the expert timeout, only actual arrival permits arm activation.
    yaw_error = (bearing - yaw + np.pi) % (2*np.pi) - np.pi
    arrived = phase == "orient" and distance < pos_tol and abs(yaw_error) < ang_tol
    if phase == "orient" and distance >= pos_tol:
        phase = "align"
    return phase, bool(arrived)


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


class ArmHistory:
    """Online equivalent: oldest-to-newest states, padded with the first state."""

    def __init__(self, steps, stride):
        if steps < 1 or stride < 1:
            raise ValueError("Window steps and stride must be positive")
        self.stride = stride
        self.history = deque(maxlen=1 + (steps - 1) * stride)

    def append(self, state):
        if not self.history:
            self.history.extend(np.asarray(state).copy() for _ in range(self.history.maxlen))
        else:
            self.history.append(np.asarray(state).copy())
        return np.asarray(self.history, dtype=np.float32)[::self.stride].reshape(-1)
