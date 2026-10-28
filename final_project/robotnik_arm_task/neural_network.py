"""Fine-tune the saved arm MLP while retaining the parent base network.

Run: python neural_network.py to fine-tune the saved near-target model.
Use --bayes to optimize recovery fine-tuning from the same parent weights.
Uses sklearn's Linear/ReLU MLP and Adam; evaluation requires only NumPy.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = None
BASELINE_MODEL_PATH = ROOT / 'trained_model_weights' / 'neural_network_near_target_augmented.npz'

# Legacy fitting defaults; fine-tuning reads layer widths from the parent model.
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
FINETUNE_LEARNING_RATE = 5e-5
FINETUNE_EPOCHS = 15
FINETUNE_PATIENCE = 5
FINETUNE_ORIGINAL_ROWS_PER_EPOCH = 100000
FINETUNE_RECOVERY_ROWS_PER_ORIGINAL = 1.0
FINETUNE_NOMINAL_MSE_TOLERANCE = 0.10


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


def bayesian_optimize(stage='arm', n_calls=15, trial_epochs=15,
                      train_limit=100000, validation_limit=30000, init_model=None):
    """BO over the existing recovery fine-tuning workflow, not random training.

    Same parent, frozen scalers/base, data pools and random seeds in every trial.
    Save a separate best candidate and JSON; never overwrite the parent.
    """
    from skopt import gp_minimize
    from skopt.space import Real
    from arm_recovery_data import load_or_build_recovery_pool, DEFAULT_SETTINGS

    global FINETUNE_LEARNING_RATE, WEIGHT_DECAY
    global FINETUNE_RECOVERY_ROWS_PER_ORIGINAL, FINETUNE_EPOCHS
    global FINETUNE_ORIGINAL_ROWS_PER_EPOCH

    if stage != 'arm':
        raise ValueError('Recovery BO supports --stage arm only; base stays frozen')
    if min(n_calls, trial_epochs, train_limit, validation_limit) < 1:
        raise ValueError('Trial counts and sample limits must be positive')
    source = Path(init_model or BASELINE_MODEL_PATH).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as archive:
        parent = {k: archive[k] for k in archive.files}
    if (str(parent['model_type'].item()) != 'staged_neural_network'
            or int(parent.get('arm_window_steps', 1)) != 1):
        raise ValueError('BO requires a single-frame staged MLP parent')
    validate_nn(parent, n_features=21, n_outputs=7)
    path = Path(str(parent['dataset_path'].item()))
    if DATA_PATH is not None and Path(DATA_PATH).resolve() != path.resolve():
        raise ValueError('BO dataset must match parent')
    data = load_dataset(path)
    for key in ('feature_names', 'action_names'):
        if not np.array_equal(data[key], parent[key]):
            raise ValueError('Parent/dataset feature or action order mismatch')
    groups = split_indices(data)
    with np.load(path, allow_pickle=False) as archive:
        active = archive['nav_done_after']
    if active.shape != (len(data['X']),) or active.dtype.kind != 'b':
        raise ValueError('Invalid arm stage labels')
    train_ids = groups['train'][active[groups['train']]]
    validation_ids = groups['validation'][active[groups['validation']]]
    if not len(train_ids) or not len(validation_ids):
        raise ValueError('Empty arm train/validation split')
    rng = np.random.default_rng(RANDOM_STATE)
    validation_ids = np.sort(rng.choice(validation_ids,
        min(validation_limit, len(validation_ids)), replace=False))
    recovery_train, train_info = load_or_build_recovery_pool(
        path, np.unique(data['episode_id'][groups['train']]), 0)
    recovery_validation, val_info = load_or_build_recovery_pool(
        path, np.unique(data['episode_id'][groups['validation']]), 1,
        settings=dict(DEFAULT_SETTINGS, episodes=8), seed=RANDOM_STATE+4)
    if set(recovery_train['parent_episode']) & set(recovery_validation['parent_episode']):
        raise ValueError('Recovery train/validation overlap')
    angle_pool, angle_info = prepare_angle_pool(path, data, groups)
    augmentation = velocity_augmentation_settings(path)
    # Small search: step size, regularization, and recovery-vs-nominal balance.
    space = [Real(1e-5, 2e-4, prior='log-uniform', name='finetune_learning_rate'),
             Real(1e-6, 1e-3, prior='log-uniform', name='weight_decay'),
             Real(.25, 2., prior='log-uniform', name='recovery_rows_per_original')]
    original = (FINETUNE_LEARNING_RATE, WEIGHT_DECAY,
                FINETUNE_RECOVERY_ROWS_PER_ORIGINAL, FINETUNE_EPOCHS,
                FINETUNE_ORIGINAL_ROWS_PER_EPOCH)
    baseline = list(original[:3])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    model_output = source.parent / f'neural_network_bo_recovery_{stamp}.npz'
    output = ROOT / 'outputs' / f'bayesian_optimization_recovery_{stamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    best_score = 1.
    best_parameters = None
    best_epoch = 0
    # Epoch zero remains a valid fallback if every trial fails to improve.
    np.savez_compressed(model_output, **parent)
    print(f'BO parent: {source}\nBO candidate: {model_output}\nBO results: {output}', flush=True)

    def objective(values):
        nonlocal best_score, best_parameters, best_epoch
        global FINETUNE_LEARNING_RATE, WEIGHT_DECAY
        global FINETUNE_RECOVERY_ROWS_PER_ORIGINAL
        FINETUNE_LEARNING_RATE, WEIGHT_DECAY, FINETUNE_RECOVERY_ROWS_PER_ORIGINAL = map(float, values)
        params = dict(zip((dimension.name for dimension in space), map(float, values)))
        print(f'BO trial {len(trials)+1}/{n_calls}: {params}', flush=True)
        # finetune_arm initializes a fresh Adam state from the SAME parent each time.
        candidate, metrics = finetune_arm(parent, data['X'], data['y'][:, 4:11],
            train_ids, validation_ids, recovery_train, recovery_validation,
            augmentation=augmentation, angle_pool=angle_pool)
        selected = [row for row in metrics['history'] if row['selected']][-1]
        score = float(selected['score'])
        trials.append(dict(parameters=params, score=score,
                           best_epoch=metrics['best_epoch'], history=metrics['history']))
        if score < best_score - MIN_DELTA:
            best_score, best_parameters, best_epoch = score, params, metrics['best_epoch']
            for key in parent:
                if key.startswith('base_') or key in ('nn_x_mean', 'nn_x_scale', 'nn_y_mean', 'nn_y_scale'):
                    np.testing.assert_array_equal(candidate[key], parent[key])
            candidate['bo_config'] = np.asarray(json.dumps(dict(
                parent_model=str(source), parent_sha256=digest, parameters=params,
                best_epoch=best_epoch, score=score, seed=RANDOM_STATE)))
            np.savez_compressed(model_output, **candidate)
        result = dict(parent_model=str(source), parent_sha256=digest,
            model=str(model_output), dataset=str(path), seed=RANDOM_STATE,
            objective='Equal nominal/recovery validation MSE ratios; nominal degradation <=10%; parent score=1. Not closed-loop success.',
            best_score=best_score, best_parameters=best_parameters, best_epoch=best_epoch,
            parent_retained=best_parameters is None, baseline_parameters=baseline,
            trial_epochs=trial_epochs, batch_size=BATCH_SIZE,
            original_rows_per_epoch=min(train_limit, len(train_ids)),
            validation_samples=len(validation_ids), recovery_training=train_info,
            recovery_validation=val_info, angle_augmentation=angle_info,
            velocity_augmentation=augmentation, trials=trials)
        output.write_text(json.dumps(result, indent=2)+'\n')
        return score

    try:
        FINETUNE_EPOCHS = trial_epochs
        FINETUNE_ORIGINAL_ROWS_PER_EPOCH = min(train_limit, len(train_ids))
        gp_minimize(objective, space, x0=baseline, n_calls=n_calls,
                    n_initial_points=min(4, n_calls-1), acq_func='EI',
                    random_state=RANDOM_STATE)
    finally:
        (FINETUNE_LEARNING_RATE, WEIGHT_DECAY,
         FINETUNE_RECOVERY_ROWS_PER_ORIGINAL, FINETUNE_EPOCHS,
         FINETUNE_ORIGINAL_ROWS_PER_EPOCH) = original
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    result = json.loads(output.read_text())
    print(f"BO complete: score={best_score:.6g}; parameters={best_parameters}\nModel: {model_output}\nResults: {output}")
    return result



def initialize_finetune_arm(model):
    """Load inference weights; initialize a NEW Adam optimizer, not a resume."""
    validate_nn(model, n_features=21, n_outputs=7)
    widths = tuple(model[f'nn_w{i}'].shape[1] for i in (0, 1))
    net = MLPRegressor(hidden_layer_sizes=widths, activation='relu', solver='adam',
        learning_rate_init=FINETUNE_LEARNING_RATE, alpha=WEIGHT_DECAY,
        batch_size=2, shuffle=False, random_state=RANDOM_STATE)
    # Public initialization establishes sklearn's version-specific fitted attrs.
    # Discard this dummy update entirely before loading the actual parameters.
    net.partial_fit(np.zeros((2,21), dtype=np.float32), np.zeros((2,7), dtype=np.float32))
    net.coefs_ = [model[f'nn_w{i}'].copy() for i in range(3)]
    net.intercepts_ = [model[f'nn_b{i}'].copy() for i in range(3)]
    del net._optimizer
    net.t_ = net.n_iter_ = net._no_improvement_count = 0
    net.loss_curve_ = []
    net.best_loss_ = np.inf
    return net


def frozen_x(model, X):
    # Exactly the exported inference normalization, including float32 rounding.
    return (np.asarray(X,dtype=np.float32)-model['nn_x_mean'])/model['nn_x_scale']


def frozen_y(model, y):
    return (np.asarray(y,dtype=np.float32)-model['nn_y_mean'])/model['nn_y_scale']


def finetune_arm(model, X, y, train_ids, validation_ids, recovery_train,
                 recovery_validation, augmentation=None, angle_pool=None, on_best=None):
    net = initialize_finetune_arm(model)
    probe = X[train_ids[:min(32,len(train_ids))]]
    np.testing.assert_allclose(net.predict(frozen_x(model,probe))*model['nn_y_scale']+model['nn_y_mean'],
                               predict_nn(model,probe),rtol=1e-5,atol=1e-5)

    def loss(xx,yy,indices=None):
        indices = np.arange(len(xx)) if indices is None else indices
        total = count = 0
        for idx in batches(indices):
            delta = net.predict(frozen_x(model,xx[idx]))-frozen_y(model,yy[idx])
            total += float(np.square(delta).sum()); count += delta.size
        value = total/count
        if not np.isfinite(value):
            raise ValueError('Non-finite fine-tuning validation loss')
        return value

    nominal_baseline = loss(X,y,validation_ids)
    recovery_baseline = loss(recovery_validation['X'],recovery_validation['y'])
    best, best_score, best_epoch, stale = copy.deepcopy(net), 1., 0, 0
    history = [dict(epoch=0, nominal_validation_mse=nominal_baseline,
                    recovery_validation_mse=recovery_baseline, score=1., selected=True)]
    rng = np.random.default_rng(RANDOM_STATE+20)
    for epoch in range(FINETUNE_EPOCHS):
        order = rng.choice(train_ids, min(FINETUNE_ORIGINAL_ROWS_PER_EPOCH,len(train_ids)),replace=False)
        rows = recovery_rows = 0
        for start in range(0,len(order),BATCH_SIZE):
            idx = order[start:start+BATCH_SIZE]
            xb,yb = X[idx],y[idx]
            if augmentation is not None:
                xb,yb = augment_arm_velocity(xb,yb,rng,augmentation)
            xb,yb,_ = mix_angle_samples(xb,yb,len(idx),angle_pool,rng)
            count = max(1,int(round(len(idx)*FINETUNE_RECOVERY_ROWS_PER_ORIGINAL)))
            ri = rng.integers(len(recovery_train['X']),size=count)
            xb = np.concatenate((xb,recovery_train['X'][ri]))
            yb = np.concatenate((yb,recovery_train['y'][ri]))
            perm = rng.permutation(len(xb));xb,yb=xb[perm],yb[perm]
            for offset in range(0,len(xb),BATCH_SIZE):
                xx,yy=xb[offset:offset+BATCH_SIZE],yb[offset:offset+BATCH_SIZE]
                net.batch_size=len(xx)
                net.partial_fit(frozen_x(model,xx),frozen_y(model,yy))
            rows+=len(xb);recovery_rows+=count
        nominal=loss(X,y,validation_ids)
        recovery=loss(recovery_validation['X'],recovery_validation['y'])
        score=.5*nominal/max(nominal_baseline,1e-12)+.5*recovery/max(recovery_baseline,1e-12)
        eligible=nominal <= nominal_baseline*(1+FINETUNE_NOMINAL_MSE_TOLERANCE)+1e-12
        selected=bool(eligible and score<best_score-MIN_DELTA)
        history.append(dict(epoch=epoch+1,nominal_validation_mse=nominal,
            recovery_validation_mse=recovery,score=score,eligible=bool(eligible),selected=selected,
            training_rows=rows,recovery_training_rows=recovery_rows))
        print(f'Fine-tune {epoch+1}/{FINETUNE_EPOCHS}: nominal={nominal:.6g}, '
              f'recovery={recovery:.6g}, score={score:.5g}, selected={selected}',flush=True)
        if selected:
            best,best_score,best_epoch,stale=copy.deepcopy(net),score,epoch+1,0
            if on_best is not None:
                on_best(best,history)
        else:
            stale+=1
        if stale>=FINETUNE_PATIENCE:
            break
    result={k:v.copy() for k,v in model.items()}
    for i in range(3):
        result[f'nn_w{i}']=best.coefs_[i].copy()
        result[f'nn_b{i}']=best.intercepts_[i].copy()
    # All base weights and scalers are copied byte-for-byte from the parent.
    return result,dict(best_epoch=best_epoch,history=history,
        nominal_baseline_mse=nominal_baseline,recovery_baseline_mse=recovery_baseline,
        selection='Equal mean of nominal/recovery validation MSE ratios to parent; '
                  'nominal degradation <=10%; epoch zero remains eligible. '
                  'Offline proxy only: closed-loop success/holding must still be evaluated.')


def finetune_main(init_model=None):
    source = Path(init_model or BASELINE_MODEL_PATH).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source,allow_pickle=False) as a:
        model={k:a[k] for k in a.files}
    if str(model['model_type'].item())!='staged_neural_network':
        raise ValueError('Fine-tuning requires a staged MLP archive')
    if int(model.get('arm_window_steps',1))!=1:
        raise ValueError('Fine-tuning requires the current single-frame baseline')
    validate_nn(model,n_features=21,n_outputs=7)
    path=Path(str(model['dataset_path'].item()))
    if DATA_PATH is not None and Path(DATA_PATH).resolve()!=path.resolve():
        raise ValueError('Fine-tuning dataset must match parent model')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=source.parent/f'neural_network_recovery_finetuned_{stamp}.npz'
    report_dir=ROOT/'outputs'/f'recovery_finetune_{stamp}'
    report_dir.mkdir(parents=True,exist_ok=False)
    provenance=dict(parent_model=str(source),parent_sha256=digest,
        output=str(output),learning_rate=FINETUNE_LEARNING_RATE,max_epochs=FINETUNE_EPOCHS,
        patience=FINETUNE_PATIENCE,original_rows_per_epoch=FINETUNE_ORIGINAL_ROWS_PER_EPOCH,
        recovery_rows_per_original=FINETUNE_RECOVERY_ROWS_PER_ORIGINAL,
        optimizer='New Adam state initialized from saved weights; not exact optimizer resume',
        base_policy='Frozen',normalization='Frozen parent input/output means and scales',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('neural_network.py','arm_recovery_data.py')})
    (report_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Parent retained: {source}\nCandidate: {output}',flush=True)
    data=load_dataset(path)
    if (not np.array_equal(data['feature_names'],model['feature_names']) or
            not np.array_equal(data['action_names'],model['action_names'])):
        raise ValueError('Dataset feature/action order differs from parent model')
    groups=split_indices(data)
    with np.load(path,allow_pickle=False) as a:
        active=a['nav_done_after']
    if active.shape!=(len(data['X']),) or active.dtype.kind!='b':
        raise ValueError('Invalid arm stage labels')
    arm_groups={name:idx[active[idx]] for name,idx in groups.items()}
    if any(not len(idx) for idx in arm_groups.values()):
        raise ValueError('Empty arm split')
    from arm_recovery_data import load_or_build_recovery_pool,DEFAULT_SETTINGS
    recovery_train,train_info=load_or_build_recovery_pool(path,
        np.unique(data['episode_id'][groups['train']]),0)
    validation_settings=dict(DEFAULT_SETTINGS,episodes=8)
    recovery_validation,val_info=load_or_build_recovery_pool(path,
        np.unique(data['episode_id'][groups['validation']]),1,
        settings=validation_settings,seed=RANDOM_STATE+4)
    assert not set(recovery_train['parent_episode']) & set(recovery_validation['parent_episode'])
    angle_pool,angle_info=prepare_angle_pool(path,data,groups)
    augmentation=velocity_augmentation_settings(path)
    model['recovery_finetune_parent_sha256']=np.asarray(digest)
    model['recovery_finetune_config']=np.asarray(json.dumps(provenance))

    def save_best(net,history):
        candidate={k:v.copy() for k,v in model.items()}
        for i in range(3):
            candidate[f'nn_w{i}']=net.coefs_[i].copy();candidate[f'nn_b{i}']=net.intercepts_[i].copy()
        np.savez_compressed(output,**candidate)
        (report_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')

    candidate,metrics=finetune_arm(model,data['X'],data['y'][:,4:11],
        arm_groups['train'],arm_groups['validation'],recovery_train,recovery_validation,
        augmentation=augmentation,angle_pool=angle_pool,on_best=save_best)
    for key in model:
        if key.startswith('base_') or key in ('nn_x_mean','nn_x_scale','nn_y_mean','nn_y_scale'):
            np.testing.assert_array_equal(candidate[key],model[key])
    candidate['recovery_finetune_best_epoch']=np.asarray(metrics['best_epoch'])
    np.savez_compressed(output,**candidate)
    metrics.update(recovery_training=train_info,recovery_validation=val_info,angle_augmentation=angle_info,
                   velocity_augmentation=augmentation,provenance=provenance)
    (report_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    print(f'Fine-tuning complete; best epoch={metrics["best_epoch"]}.\nModel: {output}\nMetrics: {report_dir}',flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bayes', action='store_true', help='BO of arm recovery fine-tuning; save separate best candidate')
    parser.add_argument('--init-model', type=Path, default=BASELINE_MODEL_PATH, help='Parent NPZ for default arm-only fine-tuning')
    parser.add_argument('--stage', choices=['base', 'arm'], default='arm')
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--trial-epochs', type=int, default=15)
    parser.add_argument('--train-limit', type=int, default=100000)
    parser.add_argument('--validation-limit', type=int, default=30000)
    args = parser.parse_args()
    if args.bayes:
        bayesian_optimize(args.stage, args.trials, args.trial_epochs,
                          args.train_limit, args.validation_limit, args.init_model)
    else:
        finetune_main(args.init_model)
