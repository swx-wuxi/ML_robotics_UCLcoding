"""NumPy-only storage and inference for an exported sklearn decision tree."""

import numpy as np


def export_tree(model):
    tree = model["estimator"].tree_
    return dict(model_type=np.asarray("decision_tree"),
                children_left=tree.children_left, children_right=tree.children_right,
                feature=tree.feature, threshold=tree.threshold,
                values=tree.value[:, :, 0], action_mean=model["action_mean"],
                feature_names=model["feature_names"], action_names=model["action_names"],
                dataset_path=np.asarray(model["dataset_path"]))


def validate_tree(model):
    n = len(model["feature"])
    if n == 0:
        raise ValueError("Empty decision tree")
    for key in ("children_left", "children_right", "feature", "threshold"):
        if model[key].shape != (n,) or not np.isfinite(model[key]).all():
            raise ValueError(f"Invalid tree parameter: {key}")
    for key in ("children_left", "children_right", "feature"):
        if model[key].dtype.kind not in "iu":
            raise ValueError(f"Tree indices must be integers: {key}")
    if model["values"].shape != (n, 11) or not np.isfinite(model["values"]).all():
        raise ValueError("Invalid tree values")
    left, right = model["children_left"], model["children_right"]
    leaf = (left == -1) & (right == -1)
    internal = ~leaf
    # sklearn numbers every child after its parent; this also rules out cycles.
    parents = np.arange(n)[internal]
    for children in (left[internal], right[internal]):
        if np.any(children <= parents) or np.any(children >= n):
            raise ValueError("Invalid tree child index")
    if np.any(model["feature"][internal] < 0) or np.any(model["feature"][internal] >= 21):
        raise ValueError("Invalid tree feature index")


def predict_tree(model, X):
    # sklearn converts input features to float32 before testing thresholds.
    X = np.asarray(X, dtype=np.float32)
    if X.ndim not in (1, 2) or X.shape[-1] != 21:
        raise ValueError("Expected state shape (21,) or (N, 21)")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite state")
    single = X.ndim == 1
    X = X.reshape(-1, 21)
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
