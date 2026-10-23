"""Linear behaviour-cloning baseline: state X -> expert action y.

Use least squares as in week 1, but predict actions instead of inertial parameters.
Run: python linear.py. This script does not start the simulator.
"""

from pathlib import Path

import numpy as np


DEFAULT_DATA = (Path(__file__).resolve().parent / "data" /
                "expert_20261021T150137_884730Z" / "dataset.npz")
MODEL_PATH = Path(__file__).resolve().parent / "linear_model.npz"


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    X, y = data["X"], data["y"]
    if X.ndim != 2 or X.shape[1] != 21 or y.shape != (len(X), 11):
        raise ValueError("需要 X=(N, 21)、y=(N, 11) 的 Robotnik 数据集")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X/y 包含非有限值")
    return data


def split_indices(data, train_ratio=0.7):
    """Use saved splits, otherwise hold out entire episodes or a time segment."""
    n = len(data["X"])
    success = data.get("episode_success", np.ones(n, dtype=bool))
    eligible = np.flatnonzero(success)
    split = data.get("split", np.zeros(n, dtype=np.int8))
    train = np.flatnonzero(success & (split == 0))
    heldout = {name: np.flatnonzero(success & (split == code))
               for name, code in (("validation", 1), ("test", 2))}
    if any(len(idx) for idx in heldout.values()):
        if not len(train):
            raise ValueError("已有验证/测试划分，但没有成功的训练样本")
        return train, {k: v for k, v in heldout.items() if len(v)}

    episodes = data.get("episode_id", np.zeros(n, dtype=int))
    ids = np.unique(episodes[eligible])
    if len(ids) > 1:
        cut = min(len(ids) - 1, max(1, int(len(ids) * train_ratio)))
        train = eligible[np.isin(episodes[eligible], ids[:cut])]
        evaluation = eligible[np.isin(episodes[eligible], ids[cut:])]
        print("按完整轨迹留出评估；相同目标仍可能出现在两组中。")
    else:
        cut = int(len(eligible) * train_ratio)
        train, evaluation = eligible[:cut], eligible[cut:]
        print("仅一条成功轨迹：前 70% 拟合，后 30% 时间留出。")
        print("该评估不代表新目标或新轨迹的泛化能力，且两段可能处于不同控制阶段。")
    if not len(train) or not len(evaluation):
        raise ValueError("成功样本不足，无法划分训练和评估数据")
    return train, {"holdout": evaluation}


def fit_linear_regression(X, y):
    """Fit y = ((X - mean) / scale) @ W + b using training statistics only."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    # Avoid division by zero for constant features such as a fixed target.
    scale = np.where(scale < 1e-12, 1.0, scale)
    design = np.column_stack(((X - mean) / scale, np.ones(len(X))))
    # Like pinv in the reference, lstsq supports multiple outputs and rank deficiency.
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    print(f"训练设计矩阵秩：{rank}/{design.shape[1]}")
    return dict(mean=mean, scale=scale,
                weights=coefficients[:-1], bias=coefficients[-1])


def predict(model, X):
    """Predict 11 actions for a batch of shape (N, 21) or one state of shape (21,)."""
    X = np.asarray(X, dtype=float)
    if X.ndim not in (1, 2) or X.shape[-1] != len(model["mean"]):
        raise ValueError("预测输入应为 (21,) 或 (N, 21)")
    return ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]


def report_metrics(label, truth, prediction, action_names):
    error = prediction - truth
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    sse = np.sum(error**2, axis=0)
    sst = np.sum((truth - truth.mean(axis=0))**2, axis=0)
    # R-squared is undefined for constant targets, such as zero wheel speeds.
    r2 = np.full(truth.shape[1], np.nan)
    np.divide(sse, sst, out=r2, where=sst > 1e-12)
    r2 = 1.0 - r2
    print(f"\n{label}: {len(truth)} samples")
    print(f"{'Action':16s} {'RMSE':>12s} {'MAE':>12s} {'R2':>12s}")
    for name, a, b, c in zip(action_names, rmse, mae, r2):
        print(f"{name:16s} {a:12.5g} {b:12.5g} {c:12.5g}")
    print("Wheel RMSE (rad/s):", np.sqrt(np.mean(error[:, :4]**2)))
    print("Arm RMSE (N*m):    ", np.sqrt(np.mean(error[:, 4:]**2)))


def plot_predictions(data, prediction, train, heldout):
    import matplotlib.pyplot as plt

    time = data.get("time", np.arange(len(prediction)))
    names = data.get("action_names", [f"action_{i}" for i in range(11)])
    fig, axes = plt.subplots(6, 2, figsize=(14, 14))
    for j, ax in enumerate(axes.flat):
        if j >= 11:
            ax.axis("off")
            continue
        # Plot episodes separately to avoid connecting lines across time resets.
        episode = data.get("episode_id", np.zeros(len(prediction), dtype=int))
        for number in np.unique(episode):
            idx = np.flatnonzero(episode == number)
            ax.plot(time[idx], data["y"][idx, j], color="black", alpha=0.5)
        ax.scatter(time[train], prediction[train, j], s=2, label="train prediction")
        for label, idx in heldout.items():
            ax.scatter(time[idx], prediction[idx, j], s=3, label=f"{label} prediction")
        ax.set_title(f"{names[j]} ({'rad/s' if j < 4 else 'N*m'})")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
    axes.flat[0].plot([], [], color="black", label="expert")
    axes.flat[0].legend()
    fig.tight_layout()
    plt.show()


def main():
    data = load_dataset(DEFAULT_DATA)
    X, y = data["X"], data["y"]
    train, heldout = split_indices(data)
    model = fit_linear_regression(X[train], y[train])
    prediction = predict(model, X)   # Core:Get the results
    names = data.get("action_names", [f"action_{i}" for i in range(11)])
    report_metrics("train", y[train], prediction[train], names)
    for label, idx in heldout.items():
        report_metrics(label, y[idx], prediction[idx], names)
        # Compare against a constant prediction equal to the training action mean.
        constant = np.broadcast_to(y[train].mean(axis=0), y[idx].shape)
        report_metrics(f"{label} / training-mean baseline", y[idx], constant, names)

    np.savez_compressed(MODEL_PATH, **model,
                        feature_names=data["feature_names"],
                        action_names=np.asarray(names))
    print(f"Model saved: {MODEL_PATH}")
    plot_predictions(data, prediction, train, heldout)


if __name__ == "__main__":
    main()
