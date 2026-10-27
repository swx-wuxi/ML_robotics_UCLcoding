"""Lecture 6 style MLP baseline: independently train base and arm networks.

Run: python neural_network.py, then select neural_network_near_target_augmented.npz.
Uses sklearn's Linear/ReLU MLP and Adam; evaluation requires only NumPy.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = None
MODEL_PATH = ROOT / 'trained_model_weights' / 'neural_network_near_target_augmented.npz'
HIDDEN_LAYERS = (128, 128)
BATCH_SIZE = 4096
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 10
EARLY_STOPPING = True  # Stop on validation plateau when enabled.
MIN_DELTA = 1e-5
RANDOM_STATE = 42
# Train-only augmentation: keep originals and add +/- velocity samples for one
# randomly selected joint per row. Inputs stay 21D; the network stays unchanged.
ARM_VELOCITY_AUGMENTATION = True
ARM_VELOCITY_DELTAS = (0.01, 0.05)  # rad/s; choose one magnitude per original row.
ARM_ANGLE_AUGMENTATION = True
ARM_ANGLE_DELTAS = (0.002, 0.005)  # rad; labels recomputed using the full expert.
ARM_ANGLE_EPISODE_LIMIT = 128  # Random successful TRAIN episodes to replay/cache.
ARM_ANGLE_SAMPLES_PER_EPISODE = 64
ARM_ANGLE_ROWS_PER_ORIGINAL = 0.5  # Extra angle rows mixed into each training epoch.
ARM_NEAR_TARGET_AUGMENTATION = True
ARM_NEAR_TARGET_ROWS_PER_ORIGINAL = 0.5  # Additional rows; keep baseline angle weight.
ARM_NEAR_TARGET_SETTINGS = dict(
    max_distance_m=0.05, max_joint_speed=0.1,
    min_shift_m=0.005, max_shift_m=0.03,
    min_angle_rad=0.005, max_angle_rad=0.02,
    samples_per_episode=64, attempts=20)


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

def base_features(X, phase):
    """Exclude arm states so arm deviations cannot change navigation predictions."""
    X = np.asarray(X)
    return np.concatenate((X[..., BASE_COLUMNS],
                           np.broadcast_to(np.asarray(phase), X.shape[:-1])[..., None]), axis=-1)

def validate_nn(model, n_features=None, n_outputs=7):
    steps = np.asarray(model.get('arm_window_steps', 1))
    if steps.shape != () or steps.dtype.kind not in 'iu' or steps.item() < 1:
        raise ValueError('Invalid NN history length')
    inputs = n_features if n_features is not None else 21 * int(steps.item())
    for key, shape in {'nn_x_mean': (inputs,), 'nn_x_scale': (inputs,),
                       'nn_y_mean': (n_outputs,), 'nn_y_scale': (n_outputs,)}.items():
        if model[key].shape != shape or not np.isfinite(model[key]).all():
            raise ValueError(f'Invalid NN parameter: {key}')
    if np.any(model['nn_x_scale'] <= 0) or np.any(model['nn_y_scale'] <= 0):
        raise ValueError('NN scales must be positive')
    width = inputs
    for i in range(3):
        w, b = model[f'nn_w{i}'], model[f'nn_b{i}']
        if (w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],)
                or not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError(f'Invalid NN layer {i}')
        width = w.shape[1]
    if width != n_outputs:
        raise ValueError(f'Expected {n_outputs} NN outputs')

def predict_nn(model, X):
    X = np.asarray(X, dtype=np.float32)
    inputs = len(model['nn_x_mean'])
    if X.ndim not in (1, 2) or X.shape[-1] != inputs or not np.isfinite(X).all():
        raise ValueError(f'Expected finite input ({inputs},) or (N, {inputs})')
    h = (X - model['nn_x_mean']) / model['nn_x_scale']
    # CORE CHANGE: two Linear+ReLU hidden layers, linear regression output.
    for i in range(3):
        h = h @ model[f'nn_w{i}'] + model[f'nn_b{i}']
        if i < 2:
            h = np.maximum(h, 0)
    return h * model['nn_y_scale'] + model['nn_y_mean']


def export_nn(estimator, xs, ys):
    model = dict(nn_x_mean=xs.mean_.astype(np.float32),
                 nn_x_scale=xs.scale_.astype(np.float32),
                 nn_y_mean=ys.mean_.astype(np.float32),
                 nn_y_scale=ys.scale_.astype(np.float32))
    for i, (w, b) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
        model[f'nn_w{i}'], model[f'nn_b{i}'] = w.copy(), b.copy()
    return model


def velocity_augmentation_settings(path):
    if not ARM_VELOCITY_AUGMENTATION:
        return None
    metadata = json.loads((Path(path).parent / 'metadata.json').read_text())
    source = ROOT / 'robotnik_arm_control.py'
    expected = metadata.get('source_sha256', {}).get(source.name)
    if expected is None or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Expert controller differs from collection; verify the velocity-label law first')
    kd = float(metadata['arm_params']['kd'])
    limit = float(metadata['arm_params']['max_torque'])
    deltas = np.asarray(ARM_VELOCITY_DELTAS, dtype=float)
    if (not np.isfinite(kd) or kd <= 0 or not np.isfinite(limit) or limit <= 0
            or deltas.ndim != 1 or not len(deltas)
            or not np.isfinite(deltas).all() or np.any(deltas <= 0)):
        raise ValueError('Invalid velocity augmentation settings')
    return dict(kd=kd, max_torque=limit, deltas=deltas.tolist(),
                seed=RANDOM_STATE + 1, original_samples_retained=True,
                label_rule='tau_new = clip(tau_original - kd * delta_qd); '
                           'only perturb a joint whose original torque is not saturated',
                reference_policy='Same pose, target and pre-action expert reference; '
                                 'velocity alone does not change expert IK/reference integration',
                scaler_policy='Fit on original training rows only',
                validation_test_policy='Original rows only; no augmentation')


def augment_arm_velocity(X, y, rng, settings):
    """Exact velocity-only expert labels without requiring FK or a simulator.

    In the official expert, IK and q_ref integration do not depend on measured
    qd. At fixed pose/reference, delta_tau = -kd*delta_qd before clipping.
    A clipped source torque hides its pre-clip value, so NEVER perturb that
    joint. Other joints may remain clipped because their inputs do not change.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 21 or y.shape != (len(X), 7):
        raise ValueError('Velocity augmentation expects X=(N,21), y=(N,7)')
    limit = settings['max_torque']
    eligible = np.abs(y) < limit - 1e-5
    rows = np.flatnonzero(eligible.any(axis=1))
    if not len(rows):
        return X.copy(), y.copy()
    # Uniformly choose one eligible joint for each row; both signs use the same
    # magnitude. No dense (dataset x 14) augmented array is ever allocated.
    scores = rng.random((len(rows), 7))
    scores[~eligible[rows]] = -1
    joints = scores.argmax(axis=1)
    delta = rng.choice(settings['deltas'], size=len(rows))
    local = np.arange(len(rows))
    parts_x, parts_y = [X], [y]
    for sign in (1, -1):
        xx, yy = X[rows].copy(), y[rows].copy()
        xx[local, 11 + joints] += sign * delta
        yy[local, joints] = np.clip(
            yy[local, joints] - settings['kd'] * sign * delta, -limit, limit)
        parts_x.append(xx)
        parts_y.append(yy)
    xx, yy = np.concatenate(parts_x), np.concatenate(parts_y)
    order = rng.permutation(len(xx))
    return xx[order], yy[order]


def prepare_angle_pool(path, data, groups):
    if not ARM_ANGLE_AUGMENTATION:
        return None, None
    if not np.isfinite(ARM_ANGLE_ROWS_PER_ORIGINAL) or ARM_ANGLE_ROWS_PER_ORIGINAL <= 0:
        raise ValueError('Angle sample ratio must be finite and positive')
    if ARM_NEAR_TARGET_AUGMENTATION and (not np.isfinite(ARM_NEAR_TARGET_ROWS_PER_ORIGINAL)
                                        or ARM_NEAR_TARGET_ROWS_PER_ORIGINAL <= 0):
        raise ValueError('Near-target sample ratio must be finite and positive')
    from arm_angle_augmentation import load_or_build_angle_pool
    pool, info = load_or_build_angle_pool(
        path, np.unique(data['episode_id'][groups['train']]),
        deltas=ARM_ANGLE_DELTAS, episode_limit=ARM_ANGLE_EPISODE_LIMIT,
        samples_per_episode=ARM_ANGLE_SAMPLES_PER_EPISODE, seed=RANDOM_STATE + 2,
        near_target=ARM_NEAR_TARGET_SETTINGS if ARM_NEAR_TARGET_AUGMENTATION else None)
    info = dict(info, rows_per_original=ARM_ANGLE_ROWS_PER_ORIGINAL,
                near_target_rows_per_original=ARM_NEAR_TARGET_ROWS_PER_ORIGINAL
                if ARM_NEAR_TARGET_AUGMENTATION else 0.)
    print(f"Angle pool sample counts: {info.get('sample_counts')}; "
          f"accepted near-target pairs: {info.get('accepted_near_pairs', 0)}", flush=True)
    return pool, info


def mix_angle_samples(X, y, original_count, pool, rng):
    if pool is None:
        return X, y, 0
    count = max(1, int(round(original_count * ARM_ANGLE_ROWS_PER_ORIGINAL)))
    kinds = pool.get('sample_kind', np.zeros(len(pool['X']), dtype=np.int8))
    single = np.flatnonzero(kinds == 0)
    if not len(single):
        raise ValueError('Angle pool is missing baseline single-joint samples')
    idx = rng.choice(single, size=count, replace=True)
    if ARM_NEAR_TARGET_AUGMENTATION:
        # Balance original-velocity and zero-velocity multi-joint labels so the
        # stationary-but-off-target states cannot be drowned out by other rows.
        near = [np.flatnonzero(kinds == kind) for kind in (1, 2)]
        if any(not len(indices) for indices in near):
            raise ValueError('Near-target augmentation requires both velocity modes in the pool')
        extra = max(2, int(round(original_count * ARM_NEAR_TARGET_ROWS_PER_ORIGINAL)))
        idx = np.concatenate((idx, rng.choice(near[0], size=extra//2, replace=True),
                              rng.choice(near[1], size=extra-extra//2, replace=True)))
        count += extra
    xx = np.concatenate((X, pool['X'][idx]))
    yy = np.concatenate((y, pool['y'][idx]))
    order = rng.permutation(len(xx))
    return xx[order], yy[order], count


def fit_nn(X, y, groups, augmentation=None, angle_pool=None):
    # CORE CHANGE: fit normalization on TRAIN ONLY, never on validation/test.
    xs, ys = StandardScaler(), StandardScaler()
    for idx in batches(groups['train']):
        xs.partial_fit(X[idx])
        ys.partial_fit(y[idx])
    net = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation='relu',
                       solver='adam', learning_rate_init=LEARNING_RATE,
                       alpha=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                       shuffle=False, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    augmentation_rng = np.random.default_rng(RANDOM_STATE + 1)
    angle_rng = np.random.default_rng(RANDOM_STATE + 2)
    best, best_loss, stale, history = None, np.inf, 0, []
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(groups['train'])
        training_rows = 0
        angle_rows = 0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            if augmentation is not None:
                xb, yb = augment_arm_velocity(xb, yb, augmentation_rng, augmentation)
            xb, yb, added = mix_angle_samples(xb, yb, len(idx), angle_pool, angle_rng)
            angle_rows += added
            # Preserve the actual optimizer minibatch size after adding rows.
            for offset in range(0, len(xb), BATCH_SIZE):
                xx, yy = xb[offset:offset+BATCH_SIZE], yb[offset:offset+BATCH_SIZE]
                net.batch_size = len(xx)
                net.partial_fit(xs.transform(xx).astype(np.float32),
                                ys.transform(yy).astype(np.float32))
            training_rows += len(xb)
        # CORE CHANGE: validation chooses the checkpoint; test is used only afterward.
        total, count = 0.0, 0
        for idx in batches(groups['validation']):
            error = net.predict(xs.transform(X[idx]).astype(np.float32)) - ys.transform(y[idx])
            total += np.square(error).sum()
            count += error.size
        loss = float(total / count)
        if not np.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        history.append(dict(epoch=epoch + 1, validation_standardized_mse=loss,
                            training_rows=training_rows,
                            angle_augmented_rows=angle_rows,
                            velocity_augmented_rows=training_rows - len(order) - angle_rows,
                            augmented_rows=training_rows - len(order)))
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
    sse, absolute = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    scaler = StandardScaler()
    for idx in batches(indices):
        truth = y[idx]
        error = predict_nn(model, X[idx]) - truth
        sse += np.square(error).sum(0)
        absolute += np.abs(error).sum(0)
        scaler.partial_fit(truth)
    rmse, mae = np.sqrt(sse / len(indices)), absolute / len(indices)
    sst = scaler.var_ * len(indices)
    ratio = np.full(y.shape[1], np.nan)
    np.divide(sse, sst, out=ratio, where=sst > 1e-12)
    r2 = 1 - ratio
    for name, a, b, c in zip(names, rmse, mae, r2):
        print(f'{name:16s} RMSE={a:.5g} MAE={b:.5g} R2={c:.5g}')
    return dict(samples=len(indices), rmse=rmse.tolist(), mae=mae.tolist(),
                r2=[float(v) if np.isfinite(v) else None for v in r2])


def bayesian_optimize(stage='arm', n_calls=15, trial_epochs=30,
                      train_limit=100000, validation_limit=30000):
    """Optional GP Bayesian search; never changes baseline globals or model files.

    Search uses fixed train/validation subsets, train-only normalization and
    standardized validation MSE. Test samples are never used for selection.
    Returns/saves best *observed* parameters; retrain and evaluate separately.
    Example: python neural_network.py --bayes --stage arm --trials 15
    """
    from skopt import gp_minimize
    from skopt.space import Categorical, Real

    if stage not in ('base', 'arm'):
        raise ValueError('stage must be base or arm')
    if min(n_calls, trial_epochs, train_limit, validation_limit) < 1:
        raise ValueError('Trial counts, epochs and sample limits must be positive')
    path = Path(DATA_PATH) if DATA_PATH is not None else find_dataset()
    data = load_dataset(path)
    groups = split_indices(data)
    with np.load(path, allow_pickle=False) as archive:
        active = archive['nav_done_after']
        phases = archive['phase_after'] if stage == 'base' else None
    if active.shape != (len(data['X']),) or active.dtype.kind != 'b':
        raise ValueError('Invalid stage labels')
    mask = active if stage == 'arm' else ~active
    rng = np.random.default_rng(RANDOM_STATE)
    selected = {}
    for name, limit in (('train', train_limit), ('validation', validation_limit)):
        indices = groups[name][mask[groups[name]]]
        if not len(indices):
            raise ValueError(f'No {stage}/{name} samples')
        # The same subset is reused across trials for comparable scores.
        selected[name] = np.sort(rng.choice(indices, min(limit, len(indices)), replace=False))
    X = data['X']
    if stage == 'base':
        if phases.shape != active.shape or not np.isin(phases, list(PHASE_CODES)).all():
            raise ValueError('Invalid navigation phases')
        codes = np.zeros(len(active), dtype=np.int8)
        for name, code in PHASE_CODES.items():
            codes[phases == name] = code
        X = base_features(X, codes)
    y = data['y'][:, 4:] if stage == 'arm' else data['y'][:, :4]
    augmentation = velocity_augmentation_settings(path) if stage == 'arm' else None
    angle_pool, angle_info = prepare_angle_pool(path, data, groups) if stage == 'arm' else (None, None)
    xs, ys = StandardScaler(), StandardScaler()
    xt, yt = X[selected['train']], y[selected['train']]
    xs.fit(xt)
    ys.fit(yt)
    original_train_samples = len(xt)
    if augmentation is not None:
        xt, yt = augment_arm_velocity(xt, yt, np.random.default_rng(RANDOM_STATE + 1), augmentation)
    xt, yt, _ = mix_angle_samples(xt, yt, original_train_samples, angle_pool,
                                np.random.default_rng(RANDOM_STATE + 2))
    xt, yt = xs.transform(xt).astype(np.float32), ys.transform(yt).astype(np.float32)
    xv = xs.transform(X[selected['validation']]).astype(np.float32)
    yv = ys.transform(y[selected['validation']]).astype(np.float32)
    del data, X, y

    # CORE CHANGE: GP surrogate + Expected Improvement chooses new trials.
    # Two hidden layers and ReLU/Adam stay fixed to preserve baseline inference.
    space = [Categorical(['64,64', '128,64', '128,128', '256,128', '256,256'], name='hidden_layers'),
             Real(1e-4, 3e-3, prior='log-uniform', name='learning_rate'),
             Real(1e-6, 1e-2, prior='log-uniform', name='weight_decay'),
             Categorical([512, 1024, 2048, 4096], name='batch_size')]
    suffix = '_velocity_augmented' if augmentation is not None else ''
    if angle_pool is not None:
        suffix += '_angle_augmented'
        if ARM_NEAR_TARGET_AUGMENTATION:
            suffix += '_near_target'
    output = ROOT / 'outputs' / f'bayesian_optimization_{stage}{suffix}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = []

    def objective(values):
        layers, lr, alpha, batch = values
        params = dict(hidden_layers=[int(v) for v in layers.split(',')],
                      learning_rate=float(lr), weight_decay=float(alpha), batch_size=int(batch))
        net = MLPRegressor(hidden_layer_sizes=tuple(params['hidden_layers']),
                           activation='relu', solver='adam', learning_rate_init=float(lr),
                           alpha=float(alpha), batch_size=int(batch), shuffle=False,
                           random_state=RANDOM_STATE)
        order_rng = np.random.default_rng(RANDOM_STATE)
        best, stale, best_epoch = np.inf, 0, 0
        print(f'Bayes trial {len(trials)+1}/{n_calls}: {params}', flush=True)
        for epoch in range(trial_epochs):
            order = order_rng.permutation(len(xt))
            for start in range(0, len(order), int(batch)):
                idx = order[start:start+int(batch)]
                net.batch_size = len(idx)
                net.partial_fit(xt[idx], yt[idx])
            score = float(np.mean((net.predict(xv) - yv)**2))
            if not np.isfinite(score):
                raise ValueError('Non-finite Bayesian trial loss')
            if score < best - MIN_DELTA:
                best, stale, best_epoch = score, 0, epoch + 1
            else:
                stale += 1
            print(f'  epoch {epoch+1}/{trial_epochs}: validation MSE={score:.6g}', flush=True)
            if EARLY_STOPPING and stale >= PATIENCE:
                break
        trials.append(dict(parameters=params, validation_standardized_mse=best,
                           best_epoch=best_epoch, epochs_run=epoch+1))
        winner = min(trials, key=lambda trial: trial['validation_standardized_mse'])
        # Persist every completed trial so a long search remains reviewable.
        result = dict(stage=stage, dataset=str(path.resolve()), seed=RANDOM_STATE,
                      objective='standardized validation MSE; no test selection',
                      train_samples=len(xt), validation_samples=len(xv),
                      original_train_samples=original_train_samples,
                      velocity_augmentation=augmentation,
                      angle_augmentation=angle_info,
                      trial_epochs=trial_epochs, early_stopping=EARLY_STOPPING,
                      best_parameters=winner['parameters'],
                      best_validation_mse=winner['validation_standardized_mse'], trials=trials)
        output.write_text(json.dumps(result, indent=2)+'\n')
        return best

    baseline = ['128,128', 0.001, 0.0001, 4096]
    gp_minimize(objective, space, x0=baseline, n_calls=n_calls,
                n_initial_points=min(4, n_calls-1), acq_func='EI',
                random_state=RANDOM_STATE)
    result = json.loads(output.read_text())
    print(f"Best observed parameters: {result['best_parameters']}\nResults: {output}")
    return result


def main():
    # CORE CHANGE: load expert data directly; no decision-tree model is required.
    path = Path(DATA_PATH) if DATA_PATH is not None else find_dataset()
    print(f'Dataset: {path}', flush=True)
    data = load_dataset(path)
    feature_names, action_names = data['feature_names'], data['action_names']
    groups = split_indices(data)
    augmentation = velocity_augmentation_settings(path)
    # Prepare once before training; future runs reuse the source-keyed cache.
    angle_pool, angle_info = prepare_angle_pool(path, data, groups)
    if augmentation is not None:
        print('Arm training: originals + paired velocity perturbations '
              f"{augmentation['deltas']} rad/s; kd={augmentation['kd']}; "
              'validation/test unchanged.', flush=True)
    with np.load(path, allow_pickle=False) as archive:
        active = archive['nav_done_after']
    if active.shape != (len(data['X']),) or active.dtype.kind != 'b':
        raise ValueError('Invalid arm stage labels')
    with np.load(path, allow_pickle=False) as archive:
        phases = archive['phase_after']
    if phases.shape != active.shape or not np.isin(phases, list(PHASE_CODES)).all():
        raise ValueError('Invalid navigation phase labels')
    codes = np.zeros(len(active), dtype=np.int8)
    for name, code in PHASE_CODES.items():
        codes[phases == name] = code
    model = dict(model_type=np.asarray('staged_neural_network'),
                 feature_names=feature_names, action_names=action_names,
                 dataset_path=np.asarray(str(path.resolve())),
                 arm_window_steps=np.asarray(1), arm_window_stride=np.asarray(1))
    metrics, histories = {}, {}
    # CORE CHANGE: train two independent MLPs using the original target split.
    # Navigation uses eight base/target/phase inputs; reaching uses 21 state inputs.
    for stage, mask, outputs in (('base', ~active, slice(0, 4)),
                                 ('arm', active, slice(4, 11))):
        stage_groups = {name: idx[mask[idx]] for name, idx in groups.items()}
        for name, idx in stage_groups.items():
            if not len(idx):
                raise ValueError(f'No {stage} samples in {name}')
            print(f'{stage}/{name}: {len(idx)} samples', flush=True)
        X = base_features(data['X'], codes) if stage == 'base' else data['X']
        y = data['y'][:, outputs]
        network, history = fit_nn(X, y, stage_groups,
                                 augmentation=augmentation if stage == 'arm' else None,
                                 angle_pool=angle_pool if stage == 'arm' else None)
        validate_nn(network, n_features=8 if stage == 'base' else 21,
                    n_outputs=4 if stage == 'base' else 7)
        # Keep arm keys compatible with older NN archives; prefix base NN keys.
        model.update({('base_' + key if stage == 'base' else key): value
                      for key, value in network.items()})
        histories[stage] = history
        for name, idx in stage_groups.items():
            print(f'{stage}/{name}')
            metrics[f'{stage}_{name}'] = report(network, X, y, idx, action_names[outputs])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model['arm_velocity_augmentation'] = np.asarray(json.dumps(augmentation))
    model['arm_angle_augmentation'] = np.asarray(json.dumps(angle_info))
    np.savez_compressed(MODEL_PATH, **model)
    output = ROOT / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{MODEL_PATH.stem}_metrics.json').write_text(json.dumps(dict(
        model=str(MODEL_PATH), dataset=str(path.resolve()), architecture={'base': [8, *HIDDEN_LAYERS, 4], 'arm': [21, *HIDDEN_LAYERS, 7]},
        activation='relu', optimizer='adam', learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        early_stopping=EARLY_STOPPING,
        arm_velocity_augmentation=augmentation,
        arm_angle_augmentation=angle_info,
        weight_decay=WEIGHT_DECAY, seed=RANDOM_STATE, history=histories, metrics=metrics), indent=2)+'\n')
    print(f'Model saved: {MODEL_PATH}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bayes', action='store_true', help='Search only; do not overwrite model weights')
    parser.add_argument('--stage', choices=['base', 'arm'], default='arm')
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--trial-epochs', type=int, default=30)
    parser.add_argument('--train-limit', type=int, default=100000)
    parser.add_argument('--validation-limit', type=int, default=30000)
    args = parser.parse_args()
    if args.bayes:
        bayesian_optimize(args.stage, args.trials, args.trial_epochs,
                          args.train_limit, args.validation_limit)
    else:
        main()
