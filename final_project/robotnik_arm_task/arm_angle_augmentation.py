"""Generate train-only angle perturbation labels using the unchanged expert.

Old datasets lack the full base quaternion. Replay selected successful training
experts and query perturbations while the full Pinocchio state is available.
No noisy actions are executed: these are one-step local corrective labels.
"""
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def quiet_replay():
    sys.stdout.flush()
    ctypes.CDLL(None).fflush(None)
    with tempfile.TemporaryFile() as log:
        saved = os.dup(1)
        failed = True
        try:
            os.dup2(log.fileno(), 1)
            yield
            failed = False
        finally:
            sys.stdout.flush()
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            if failed:
                log.seek(0)
                sys.stdout.write(log.read().decode(errors='replace'))
                sys.stdout.flush()


def load_or_build_angle_pool(dataset, training_episodes, deltas=(.002, .005),
                             episode_limit=128, samples_per_episode=64, seed=44,
                             cache_root=None):
    dataset = Path(dataset)
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or not len(deltas) or not np.isfinite(deltas).all() or np.any(deltas <= 0):
        raise ValueError('Angle deltas must be finite positive radians')
    if episode_limit < 1 or samples_per_episode < 1:
        raise ValueError('Angle episode/sample counts must be positive')
    allowed = set(int(e) for e in training_episodes)
    manifest = json.loads((dataset.parent/'episodes.json').read_text())
    candidates = sorted(e['episode_id'] for e in manifest
                        if e['episode_id'] in allowed and e['split']==0 and e['success'])
    if not candidates:
        raise ValueError('No successful training episodes available for angle augmentation')
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(candidates, min(episode_limit, len(candidates)), replace=False).tolist())
    source_names = ('robotnik_arm_task.py', 'robotnik_arm_control.py',
                    'collect_expert_data.py', 'configs/robotnik_arm.json', 'arm_angle_augmentation.py')
    settings = dict(dataset=str(dataset.resolve()), dataset_bytes=dataset.stat().st_size,
        dataset_mtime_ns=dataset.stat().st_mtime_ns, episodes=selected, deltas_rad=deltas.tolist(),
        samples_per_episode=samples_per_episode, seed=seed,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},
        metadata_sha256=hashlib.sha256((dataset.parent/'metadata.json').read_bytes()).hexdigest(),
        label_method='Replayed full expert state; perturb one joint +/-; recompute FK, IK, reference integration and gravity with the same pre-action q_ref. No perturbed action is executed.',
        selection='Successful training episodes only; uniform trajectory-time centers; skip limit-margin crossings',
        reproduction_tolerance=1e-6)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = Path(cache_root) if cache_root is not None else ROOT/'data'/'augmentation_cache'
    cache = cache_dir/f'arm_angle_{key}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as a:
            pool = {k:a[k] for k in ('X', 'y', 'parent_episode', 'parent_row')}
            info = json.loads(str(a['info'].item()))
        validate_pool(pool, allowed)
        print(f'Angle augmentation cache: {cache} ({len(pool["X"])} rows)', flush=True)
        return pool, info

    # Imports are deferred: ordinary inference and velocity-only training need
    # neither PyBullet nor Pinocchio.
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/robotnik_matplotlib')
    import collect_expert_data as collector
    task = collector.task
    metadata = json.loads((dataset.parent/'metadata.json').read_text())
    if settings['source_sha256']['robotnik_arm_control.py'] != metadata['source_sha256']['robotnik_arm_control.py']:
        raise ValueError('Official expert differs from collection; cannot generate matching angle labels')
    official = task.arm_controller
    xs, ys, parents, parent_rows = [], [], [], []
    max_state_error = max_torque_error = 0.
    for number, episode_id in enumerate(selected, 1):
        episode_path = dataset.parent/f'episode_{episode_id:04d}.npz'
        with np.load(episode_path, allow_pickle=False) as a:
            expected_X, expected_y = a['X'], a['y'][:, 4:11]
            active, expected_ref = a['nav_done_after'], a['q_ref_before']
            if not np.all(a['split']==0) or not np.all(a['episode_success']):
                raise ValueError('Angle augmentation attempted to use a holdout/failed episode')
        indices = np.flatnonzero(active)
        if not len(indices):
            continue
        centers = set(indices[np.unique(np.linspace(0, len(indices)-1,
                      min(samples_per_episode, len(indices)), dtype=int))].tolist())
        row = 0
        found = set()

        def probe(*args, **kwargs):
            nonlocal row, max_state_error, max_torque_error
            original = official(*args, **kwargs)
            current = row
            row += 1
            if current not in centers:
                return original
            dyn, qp, q, qd, reference, hold, target, ee, frame, shoulder, lower, upper, dt, params = args[:14]
            reach = kwargs.get('reach_active', args[14] if len(args)>14 else None)
            if not reach:
                raise ValueError('Replayed arm stage no longer matches collection')
            # Floating base position/quaternion then arm coordinates: assert the
            # actual Pinocchio ordering instead of silently assuming joint names.
            np.testing.assert_allclose(qp[7:14], q, rtol=0, atol=1e-10)
            yaw = task.quaternion2bearing(qp[6], qp[3], qp[4], qp[5])
            x = np.concatenate((qp[:3], [yaw], q, qd, target))
            if current >= len(expected_X):
                raise ValueError('Expert replay exceeds source episode')
            state_error = float(np.max(np.abs(x-expected_X[current])))
            torque_error = float(np.max(np.abs(original[0]-expected_y[current])))
            max_state_error = max(max_state_error, state_error)
            max_torque_error = max(max_torque_error, torque_error)
            if state_error > 1e-6 or torque_error > 1e-6 or not np.allclose(reference, expected_ref[current], rtol=0, atol=1e-6):
                raise ValueError(f'Expert episode {episode_id}, row {current} cannot be reproduced: '
                                 f'state error={state_error:g}, torque error={torque_error:g}. '
                                 'Refusing approximate angle labels.')
            found.add(current)
            delta = float(rng.choice(deltas))
            limited = np.asarray(lower) < np.asarray(upper)
            eligible = (~limited | ((q-delta >= np.asarray(lower)+params.limit_margin) &
                                     (q+delta <= np.asarray(upper)-params.limit_margin)))
            joints = np.flatnonzero(eligible)
            if not len(joints):
                return original
            joint = int(rng.choice(joints))
            try:
                for sign in (1, -1):
                    qq = q.copy(); qq[joint] += sign*delta
                    perturbed = qp.copy(); perturbed[7+joint] = qq[joint]
                    task.pin.forwardKinematics(dyn.pin_model, dyn.pin_data, perturbed)
                    task.pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
                    ee_new = dyn.pin_data.oMf[frame].translation.copy()
                    torque, _, _ = official(dyn, perturbed, qq, qd.copy(), reference.copy(), hold.copy(),
                        target.copy(), ee_new, frame, shoulder, lower, upper, dt, params, reach_active=True)
                    xx = x.copy(); xx[4+joint] = qq[joint]
                    xs.append(xx); ys.append(torque.copy())
                    parents.append(episode_id); parent_rows.append(current)
            finally:
                # Queries never change the running expert's reference or physics.
                task.pin.forwardKinematics(dyn.pin_model, dyn.pin_data, qp)
                task.pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
            return original

        print(f'Preparing angle labels: episode {number}/{len(selected)} (id={episode_id})', flush=True)
        try:
            task.arm_controller = probe
            target = expected_X[0, 18:21].copy()
            with quiet_replay():
                replay_data, summary = collector.collect_episode(target, metadata['max_steps'], gui=False)
            del replay_data
        finally:
            task.arm_controller = official
        if found != centers or not summary['success']:
            raise ValueError(f'Expert replay {episode_id} did not reproduce all selected successful arm states')

    pool = dict(X=np.asarray(xs), y=np.asarray(ys), parent_episode=np.asarray(parents, dtype=np.int64),
                parent_row=np.asarray(parent_rows, dtype=np.int64))
    validate_pool(pool, allowed)
    info = dict(settings, cache=str(cache), samples=len(xs),
                max_replay_state_error=max_state_error, max_replay_torque_error=max_torque_error)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write atomically so interruption cannot leave a seemingly valid cache.
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **pool, info=np.asarray(json.dumps(info)))
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return pool, info


def validate_pool(pool, allowed):
    n = len(pool['X'])
    if not n or pool['X'].shape != (n, 21) or pool['y'].shape != (n, 7):
        raise ValueError('Invalid/empty angle augmentation pool')
    if not np.isfinite(pool['X']).all() or not np.isfinite(pool['y']).all():
        raise ValueError('Non-finite angle augmentation')
    if not set(pool['parent_episode'].tolist()) <= allowed:
        raise ValueError('Angle augmentation pool contains holdout episodes')
