"""Read-only dataset audit and comparison; no training or robot simulation.

Example: python check_data_quality.py --dataset old=data/expert_old --dataset new=data/expert_new
Paths are relative to the working directory. See README_DATA_QUALITY.md.
"""
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robotnik_matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = (['base_x', 'base_y', 'base_z', 'base_yaw']
            + [f'arm_q_{i}' for i in range(1, 8)]
            + [f'arm_qd_{i}' for i in range(1, 8)] + ['target_x', 'target_y', 'target_z'])
ACTIONS = ['wheel_FR', 'wheel_FL', 'wheel_BL', 'wheel_BR'] + [f'arm_torque_{i}' for i in range(1, 8)]
CORE = ('X', 'y', 'time', 'episode_id', 'split', 'episode_success', 'nav_done_after',
        'phase_after', 'feature_names', 'action_names')
DIAGNOSTIC = ('ee_pos', 'ee_error', 'q_mes', 'qd_mes', 'q_ref_before',
              'nav_done_before', 'phase_before')
SPLITS = {0: 'train', 1: 'validation', 2: 'test'}
# Fixed physical scales make neighbor distances comparable across runs. They are
# diagnostic choices, not learned scalers, robot limits, or success thresholds.
STATE_SCALES = np.array([.1, .1, .05, .2] + [.1] * 14 + [.1] * 3)
REPEAT_TOLERANCES = np.array([1e-4] * 4 + [1e-4] * 7 + [1e-3] * 7 + [1e-6] * 3)


def read_json(path):
    # Collector rewrites its manifest during collection. Never silently replace
    # an unreadable manifest with an empty one and call the dataset healthy.
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{path} is incomplete/being written; rerun after the write finishes') from error


def issue(issues, level, code, message, episode_id=None):
    issues.append(dict(level=level, code=code, episode_id=episode_id, message=message))


def ratio(n, d):
    return float(n / d) if d else None


def percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, median=None, p95=None, maximum=None)
    return dict(n=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def success_interval(successes, count):
    """Wilson 95% interval; descriptive, not a guarantee for selected targets."""
    if not count:
        return [None, None]
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [center - half, center + half]


def load_manifest(folder):
    manifest_path = folder / 'episodes.json'
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f'No completed episodes in {manifest_path}')
    metadata = read_json(folder / 'metadata.json') if (folder / 'metadata.json').exists() else {}
    planned = read_json(folder / 'targets.json') if (folder / 'targets.json').exists() else manifest
    issues, ids, target_splits = [], set(), {}
    for entry in manifest:
        eid = entry.get('episode_id')
        if not isinstance(eid, int) or eid < 0 or eid in ids:
            raise ValueError(f'Invalid/duplicate episode ID in {manifest_path}: {eid}')
        ids.add(eid)
        target = np.asarray(entry.get('target', []), dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or entry.get('split') not in SPLITS:
            raise ValueError(f'Invalid target/split in {manifest_path}, episode {eid}')
        key = tuple(target)
        target_splits.setdefault(key, set()).add(entry['split'])
    leaking = [key for key, splits in target_splits.items() if len(splits) > 1]
    if leaking:
        issue(issues, 'error', 'target_split_leakage', f'{len(leaking)} exact targets occur in different splits')
    if len(target_splits) < len(manifest):
        issue(issues, 'warning', 'duplicate_targets', f'{len(manifest) - len(target_splits)} repeated episode targets')
    plan_by_id = {e['episode_id']: e for e in planned}
    if len(plan_by_id) != len(planned):
        issue(issues, 'error', 'duplicate_plan_ids', 'targets.json contains duplicate episode IDs')
    for e in manifest:
        p = plan_by_id.get(e['episode_id'])
        if p is None or p['split'] != e['split'] or not np.array_equal(p['target'], e['target']):
            issue(issues, 'error', 'plan_mismatch', 'Recorded target/split differs from targets.json', e['episode_id'])
    region = metadata.get('target_sampling', {})
    for e in manifest:
        if region.get('mode') in ('uniform_area_disk', 'uniform_area_annulus'):
            target = np.asarray(e['target'])
            radius = np.linalg.norm(target[:2] - region['center_xy'])
            low, high = ((region['height'], region['height']) if 'height' in region
                         else (region['min_height'], region['max_height']))
            if not (region['min_radius'] - 1e-7 <= radius <= region['max_radius'] + 1e-7
                    and low - 1e-7 <= target[2] <= high + 1e-7):
                issue(issues, 'error', 'target_outside_region', 'Target lies outside recorded sampling bounds', e['episode_id'])
    return manifest, metadata, planned, issues


def audit_episode(path, entry, metadata, args):
    eid = entry['episode_id']
    issues = []
    with np.load(path, allow_pickle=False) as archive:
        missing = [k for k in CORE if k not in archive.files]
        if missing:
            raise ValueError('Missing required fields: ' + ', '.join(missing))
        data = {k: archive[k] for k in CORE + DIAGNOSTIC if k in archive.files}
    x, y, t = data['X'], data['y'], data['time']
    n = x.shape[0] if x.ndim else 0
    if not n or x.shape != (n, 21) or y.shape != (n, 11):
        raise ValueError('Expected nonempty X=(N,21), y=(N,11)')
    for k in ('time', 'episode_id', 'split', 'episode_success', 'nav_done_after'):
        if data[k].shape != (n,):
            raise ValueError(f'Wrong shape for {k}')
    for k in ('episode_id', 'split'):
        if data[k].dtype.kind not in 'iu':
            raise ValueError(f'{k} must be an integer array')
    for k in ('episode_success', 'nav_done_after'):
        if data[k].dtype.kind != 'b':
            raise ValueError(f'{k} must be boolean')
    if data['feature_names'].tolist() != FEATURES or data['action_names'].tolist() != ACTIONS:
        raise ValueError('Feature/action order differs from the training contract')
    for k, value in data.items():
        if k in ('X', 'y', 'time', 'ee_pos', 'ee_error', 'q_mes', 'qd_mes', 'q_ref_before') and value.dtype.kind not in 'fiu':
            raise ValueError(f'{k} must contain real numeric values')
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite values in {k}')
    checks = {
        'episode_id_mismatch': np.all(data['episode_id'] == eid),
        'split_mismatch': np.all(data['split'] == entry['split']),
        'success_flag_mismatch': np.all(data['episode_success'] == entry['success']),
        'sample_count_mismatch': n == entry['samples'],
        'target_row_mismatch': np.allclose(x[:, 18:21], entry['target'], atol=1e-7, rtol=0),
        'success_message_mismatch': bool(entry['success']) == entry['stop_message'].startswith('Target reached'),
    }
    dt = float(entry.get('time_step', metadata.get('time_step', 0)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Missing/invalid recorded time step')
    checks['time_grid_mismatch'] = np.allclose(t, np.arange(n) * dt, rtol=0, atol=1e-8)
    active = data['nav_done_after']
    checks['stage_reversal'] = not np.any(active[:-1] & ~active[1:])
    missing_diagnostics = [k for k in DIAGNOSTIC if k not in data]
    if missing_diagnostics:
        issue(issues, 'warning', 'missing_diagnostics', ', '.join(missing_diagnostics), eid)
    for key, cols in (('q_mes', slice(4, 11)), ('qd_mes', slice(11, 18))):
        if key in data:
            if data[key].ndim != 2 or data[key].shape[0] != n or data[key].shape[1] < 11:
                raise ValueError(f'Invalid {key} shape')
            checks[key + '_alignment'] = np.allclose(x[:, cols], data[key][:, 4:11], rtol=0, atol=1e-8)
    for key in ('nav_done_before', 'phase_before', 'phase_after'):
        if key in data and data[key].shape != (n,):
            raise ValueError(f'Invalid {key} shape')
    if 'nav_done_before' in data:
        checks['stage_before_after_alignment'] = np.array_equal(data['nav_done_before'][1:], active[:-1])
    for key in ('phase_before', 'phase_after'):
        if key in data:
            checks[key + '_values'] = np.isin(data[key], ['align', 'drive', 'orient']).all()
    if 'phase_before' in data and 'phase_after' in data:
        checks['phase_before_after_alignment'] = np.array_equal(data['phase_before'][1:], data['phase_after'][:-1])
    distance = None
    if 'ee_error' in data:
        if data['ee_error'].shape != (n,):
            raise ValueError('Invalid ee_error shape')
        distance = data['ee_error']
        checks['negative_ee_error'] = np.all(distance >= 0)
    if 'ee_pos' in data:
        if data['ee_pos'].shape != (n, 3):
            raise ValueError('Invalid ee_pos shape')
        computed = np.linalg.norm(data['ee_pos'] - entry['target'], axis=1)
        if distance is not None:
            checks['ee_error_alignment'] = np.allclose(distance, computed, rtol=0, atol=1e-8)
        else:
            distance = computed
    if 'q_ref_before' in data and data['q_ref_before'].shape != (n, 7):
        raise ValueError('Invalid q_ref_before shape')
    final_error = entry.get('final_error')
    if final_error is None or not np.isfinite(final_error) or final_error < 0:
        checks['invalid_final_error'] = False
    if entry.get('final_ee_pos') is not None and final_error is not None:
        checks['final_position_error_mismatch'] = np.isclose(
            np.linalg.norm(np.asarray(entry['final_ee_pos']) - entry['target']), final_error, rtol=0, atol=1e-8)
    stop = entry.get('stop_observation')
    if stop is not None:
        checks['stop_time_alignment'] = np.isclose(stop['time'], t[-1], rtol=0, atol=1e-8)
        checks['stop_stage_alignment'] = bool(stop['nav_done']) == bool(active[-1])
        if distance is not None:
            checks['stop_error_alignment'] = np.isclose(stop['ee_error'], distance[-1], rtol=0, atol=1e-8)
    tolerance = metadata.get('stop_criteria', {}).get('reach_tolerance', args.reach_tolerance)
    if entry['success'] and distance is not None:
        checks['success_pre_error_outside_tolerance'] = bool(active[-1] and distance[-1] < tolerance)
    # The template checks a PRE-action error. A post-action discrepancy is worth
    # inspecting but must not be mislabeled as corruption of its official label.
    post_discrepancy = bool(entry['success'] and final_error is not None and final_error >= tolerance)
    if post_discrepancy:
        issue(issues, 'warning', 'post_action_success_discrepancy',
              'Template success, but post-action endpoint is outside tolerance', eid)
    torque_limit = metadata.get('arm_params', {}).get('max_torque')
    saturation = None
    if torque_limit is not None:
        checks['torque_outside_limit'] = np.all(np.abs(y[:, 4:]) <= torque_limit + 1e-5)
        saturation = int(np.count_nonzero(np.abs(y[active, 4:]) >= torque_limit - 1e-5))
    nav = metadata.get('base_params')
    if nav:
        wheel_limit = (nav['max_lin_vel'] + nav['max_ang_vel'] * nav['wheel_base_width'] / 2) / nav['wheel_radius']
        checks['wheel_outside_limit'] = np.all(np.abs(y[:, :4]) <= wheel_limit + 1e-5)
    for code, passed in checks.items():
        if not passed:
            issue(issues, 'error', code, 'Recorded arrays/manifest failed consistency check', eid)
    arm_rows = np.flatnonzero(active)
    first = int(arm_rows[0]) if len(arm_rows) else None
    speed = np.max(np.abs(x[:, 11:18]), axis=1)
    low_speed = speed < args.low_speed
    critical = active & low_speed & (distance >= args.reach_tolerance) & (distance < args.near_distance) if distance is not None else None
    state_counts = None
    if distance is not None:
        state_counts = [int(np.count_nonzero(active & condition)) for condition in (
            (distance < args.reach_tolerance) & low_speed,
            (distance < args.reach_tolerance) & ~low_speed,
            (distance >= args.reach_tolerance) & (distance < args.near_distance) & low_speed,
            (distance >= args.reach_tolerance) & (distance < args.near_distance) & ~low_speed,
            distance >= args.near_distance)]
    adjacent = active[:-1] & active[1:]
    repeated = np.all(np.abs(np.diff(x, axis=0)) <= REPEAT_TOLERANCES, axis=1)
    ref_rms = float(np.sqrt(np.mean((data['q_ref_before'][active] - x[active, 4:11]) ** 2))) if 'q_ref_before' in data and len(arm_rows) else None
    record = dict(episode_id=eid, split=entry['split'], success=bool(entry['success']),
                  target=entry['target'], samples=n, arm_rows=int(active.sum()), duration_s=n * dt,
                  final_error_m=final_error, stop_message=entry['stop_message'],
                  critical_rows=int(critical.sum()) if critical is not None else None,
                  critical_seconds=float(critical.sum() * dt) if critical is not None else None,
                  state_counts=state_counts, adjacent_pairs=int(adjacent.sum()),
                  near_duplicate_pairs=int(np.count_nonzero(adjacent & repeated)),
                  saturated_arm_channels=saturation, reference_gap_rms_rad=ref_rms,
                  arm_entry_linear_speed=None, arm_entry_angular_speed=None,
                  post_action_success_discrepancy=post_discrepancy,
                  valid=not any(i['level'] == 'error' for i in issues))
    if first is not None and first > 0:
        record['arm_entry_linear_speed'] = float(np.linalg.norm(x[first, :2] - x[first - 1, :2]) / dt)
        record['arm_entry_angular_speed'] = float(abs((x[first, 3] - x[first - 1, 3] + np.pi) % (2 * np.pi) - np.pi) / dt)
    # Bounded, deterministic, episode-balanced sample for geometric diagnostics.
    take = np.random.default_rng(np.random.SeedSequence([args.seed, eid])).choice(
        arm_rows, min(args.rows_per_episode, len(arm_rows)), replace=False)
    sample = dict(X=x[take], y=y[take, 4:11], episode_id=np.full(len(take), eid))
    return record, issues, sample


def merged_audit(folder, manifest, metadata, full, issues):
    path = folder / 'dataset.npz'
    if not path.exists():
        return dict(status='missing', mode='none')
    expected = sum(e['samples'] for e in manifest)
    checked = []
    try:
        with zipfile.ZipFile(path) as archive:
            for key in CORE:
                with archive.open(key + '.npy') as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError(f'Unsupported NPY version {version}')
                    wanted = ((21,) if key == 'feature_names' else (11,) if key == 'action_names'
                              else (expected, 21) if key == 'X' else (expected, 11) if key == 'y' else (expected,))
                    if shape != wanted or dtype.hasobject or fortran:
                        raise ValueError(f'Invalid merged {key} header: {shape}, {dtype}, fortran={fortran}')
                    if key in ('feature_names', 'action_names'):
                        actual = np.frombuffer(member.read(math.prod(shape) * dtype.itemsize), dtype=dtype)
                        if actual.tolist() != (FEATURES if key == 'feature_names' else ACTIONS):
                            raise ValueError(f'Merged {key} order mismatch')
                    elif full:
                        for e in manifest:
                            count = e['samples'] * math.prod(shape[1:])
                            raw = member.read(count * dtype.itemsize)
                            if len(raw) != count * dtype.itemsize:
                                raise ValueError(f'Truncated merged {key}')
                            actual = np.frombuffer(raw, dtype=dtype).reshape((e['samples'],) + shape[1:])
                            with np.load(folder / f"episode_{e['episode_id']:04d}.npz", allow_pickle=False) as source:
                                if not np.array_equal(actual, source[key]):
                                    raise ValueError(f'Merged {key} differs from episode {e["episode_id"]}')
                        if member.read(1):
                            raise ValueError(f'Unexpected trailing bytes in {key}')
                    checked.append(key)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        issue(issues, 'error', 'merged_dataset_invalid', str(error))
        return dict(status='failed', mode='full_core_fields' if full else 'headers_only', checked=checked)
    if metadata and metadata.get('samples') != expected:
        issue(issues, 'error', 'metadata_sample_count', 'metadata.samples differs from episodes.json')
    return dict(status='passed', mode='full_core_fields' if full else 'headers_and_field_order', checked=checked)


def neighbor_diagnostic(parts, args):
    if not parts:
        return dict(status='unavailable', reason='No valid successful TRAIN Arm samples')
    x = np.concatenate([p['X'] for p in parts])
    y = np.concatenate([p['y'] for p in parts])
    episode = np.concatenate([p['episode_id'] for p in parts])
    rng = np.random.default_rng(args.seed)
    if len(x) > args.neighbor_rows:
        keep = rng.choice(len(x), args.neighbor_rows, replace=False)
        x, y, episode = x[keep], y[keep], episode[keep]
    if len(np.unique(episode)) < 2:
        return dict(status='unavailable', reason='Need at least two successful TRAIN Arm trajectories')
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return dict(status='unavailable', reason='Install scipy to enable cross-episode neighbor diagnostics')
    k = min(args.rows_per_episode + 1, len(x))
    # At most rows_per_episode samples came from one trajectory, so k contains
    # a sample from another trajectory. Do not confuse same-episode continuity
    # with cross-episode consistency.
    distances, indices = cKDTree(x / STATE_SCALES).query(x / STATE_SCALES, k=k, workers=1)
    other = episode[indices] != episode[:, None]
    valid = other.any(axis=1)
    first = np.argmax(other, axis=1)
    neighbor = indices[np.arange(len(x)), first]
    distance = distances[np.arange(len(x)), first] / np.sqrt(21)
    delta = np.sqrt(np.mean((y - y[neighbor]) ** 2, axis=1))
    close = valid & (distance <= args.neighbor_radius)
    return dict(status='computed', rows=len(x), trajectories=len(np.unique(episode)),
                close_pairs=int(close.sum()), close_fraction=ratio(int(close.sum()), int(valid.sum())),
                normalized_neighbor_distance=percentiles(distance[valid]),
                close_pair_torque_rms_nm=percentiles(delta[close]),
                meaning='Directed nearest cross-episode pairs on sampled successful TRAIN rows. '
                        'Large action differences are a diagnostic, not proof of erroneous labels. '
                        'Distance uses fixed physical scales; raw yaw matches the current MLP input.')


def audit_dataset(name, folder, args):
    manifest, metadata, planned, issues = load_manifest(folder)
    available_snapshot = len(manifest)
    selected = manifest
    if args.max_episodes and len(manifest) > args.max_episodes:
        selected_ids = set(np.random.default_rng(args.seed).choice(len(manifest), args.max_episodes, replace=False).tolist())
        selected = [e for i, e in enumerate(manifest) if i in selected_ids]
    collection_complete = bool(metadata and (folder / 'dataset.npz').is_file()
                               and metadata.get('episodes') == len(manifest) == len(planned))
    if not collection_complete:
        issue(issues, 'warning', 'incomplete_collection', 'Collection lacks final metadata/merged archive or not all planned episodes are present')
    if len(selected) != len(manifest):
        issue(issues, 'warning', 'sampled_audit', f'Only {len(selected)}/{len(manifest)} episode files scanned')
    if not metadata:
        issue(issues, 'warning', 'metadata_unavailable', 'Recorded limits/source hashes unavailable; those checks are not performed')
    records, parts = [], []
    for number, entry in enumerate(selected, 1):
        try:
            record, local, sample = audit_episode(folder / f"episode_{entry['episode_id']:04d}.npz", entry, metadata, args)
            records.append(record)
            issues.extend(local)
            if record['valid'] and record['split'] == 0 and record['success'] and len(sample['X']):
                parts.append(sample)
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as error:
            issue(issues, 'error', 'episode_unreadable_or_invalid', str(error), entry['episode_id'])
            records.append(dict(episode_id=entry['episode_id'], valid=False, target=entry['target'],
                                split=entry['split'], success=bool(entry['success'])))
        if number % 50 == 0 or number == len(selected):
            print(f'{name}: checked {number}/{len(selected)} episodes', flush=True)
    merged = (merged_audit(folder, manifest, metadata, args.check_merged, issues)
              if collection_complete else dict(status='not_checked_incomplete_collection', mode='none'))
    if args.check_merged and not collection_complete:
        issue(issues, 'warning', 'merged_full_check_skipped', 'Full merge comparison requires completed collection')
    source_status = {}
    for key, expected in metadata.get('source_sha256', {}).items():
        candidate = ROOT / key
        # Hash mismatch describes provenance drift, not evidence of bad samples.
        source_status[key] = ('match' if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected
                              else 'different_or_missing')
    if any(value != 'match' for value in source_status.values()):
        issue(issues, 'warning', 'source_drift', 'Current source/config hashes differ from collection; comments can also change hashes')
    valid = [e for e in records if e['valid']]
    train = [e for e in valid if e['split'] == 0 and e['success']]
    total_arm = sum(e['arm_rows'] for e in train)
    known = [e for e in train if e['critical_rows'] is not None]
    known_arm = sum(e['arm_rows'] for e in known)
    saturated = [e for e in train if e['saturated_arm_channels'] is not None]
    weights = np.array([e['arm_rows'] for e in train], dtype=float)
    critical_rows = sum(e['critical_rows'] for e in known)
    successes = sum(e['success'] for e in valid)
    state_counts = np.sum([e['state_counts'] for e in known], axis=0).astype(int).tolist() if known else [0] * 5
    summary = dict(valid_episodes=len(valid), invalid_episodes=len(records) - len(valid),
                   expert_success_rate=ratio(successes, len(valid)), expert_successes=successes,
                   success_wilson95=success_interval(successes, len(valid)),
                   rows=sum(e['samples'] for e in valid), successful_train_episodes=len(train),
                   train_arm_rows=total_arm, critical_train_rows=critical_rows,
                   critical_train_fraction=ratio(critical_rows, known_arm),
                   critical_train_seconds=sum(e['critical_seconds'] for e in known),
                   critical_covered_train_episodes=sum(e['critical_rows'] > 0 for e in known),
                   critical_episode_coverage=ratio(sum(e['critical_rows'] > 0 for e in known), len(known)),
                   critical_macro_fraction=float(np.mean([e['critical_rows'] / e['arm_rows'] for e in known if e['arm_rows']])) if known_arm else None,
                   state_counts=state_counts,
                   near_duplicate_fraction=ratio(sum(e['near_duplicate_pairs'] for e in train), sum(e['adjacent_pairs'] for e in train)),
                   saturated_arm_channel_fraction=ratio(sum(e['saturated_arm_channels'] for e in saturated), 7 * sum(e['arm_rows'] for e in saturated)),
                   row_weight_effective_trajectories=float(weights.sum() ** 2 / (weights @ weights)) if weights.sum() else None,
                   post_action_success_discrepancies=sum(e['post_action_success_discrepancy'] for e in valid),
                   final_error_m=percentiles([e['final_error_m'] for e in valid]),
                   arm_entry_angular_speed=percentiles([e['arm_entry_angular_speed'] for e in valid if e['arm_entry_angular_speed'] is not None]),
                   reference_gap_rms_rad=percentiles([e['reference_gap_rms_rad'] for e in train if e['reference_gap_rms_rad'] is not None]))
    invalid_count = sum(i['level'] == 'error' for i in issues)
    status = 'ERROR' if invalid_count else ('PASS_SCANNED' if collection_complete and len(selected) == len(manifest) else 'PRELIMINARY')
    return dict(name=name, folder=str(folder), status=status, collection_complete=collection_complete,
                audit_complete=len(selected) == len(manifest), planned_episodes=len(planned),
                available_episodes=available_snapshot, scanned_episodes=len(selected),
                region=metadata.get('target_sampling'), limits_available=bool(metadata),
                summary=summary, issues=issues, issue_counts=dict(Counter(i['code'] for i in issues)),
                merged_check=merged, source_hash_status=source_status,
                neighbor=neighbor_diagnostic(parts, args), episodes=records,
                split_summary={SPLITS[s]: dict(episodes=sum(e['split'] == s for e in valid),
                    successes=sum(e['split'] == s and e['success'] for e in valid)) for s in SPLITS})


def reference_coverage(report, targets):
    train = [e['target'] for e in report['episodes'] if e['valid'] and e['split'] == 0 and e['success']]
    if not train:
        return dict(status='unavailable')
    train = np.unique(np.asarray(train), axis=0)
    distances = []
    for start in range(0, len(targets), 1000):
        distances.extend(np.linalg.norm(targets[start:start + 1000, None, :] - train[None, :, :], axis=2).min(axis=1))
    distances = np.asarray(distances)
    unseen = distances > 1e-9
    return dict(status='computed', reference_targets=len(targets), unique_successful_train_targets=len(train),
                nearest_target_distance_m=percentiles(distances),
                fraction_within_5cm=float(np.mean(distances <= .05)),
                overlapping_training_targets=int(np.count_nonzero(~unseen)),
                unseen_reference_distance_m=percentiles(distances[unseen]),
                unseen_fraction_within_5cm=ratio(int(np.count_nonzero(distances[unseen] <= .05)), int(unseen.sum())),
                note='Target-position coverage only, not state coverage or predicted success')


def load_reference(path):
    obj = read_json(path)
    if isinstance(obj, dict):
        obj = obj.get('targets', obj.get('episodes'))
    if not isinstance(obj, list) or not obj:
        raise ValueError('Reference must contain targets or episode targets')
    targets = np.asarray([e['target'] if isinstance(e, dict) else e for e in obj], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError('Reference targets must be finite (N,3)')
    return targets


def plot_reports(reports, output, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = []
    all_targets = np.asarray([e['target'] for r in reports for e in r['episodes']])
    xy_limit = max(.1, float(np.abs(all_targets[:, :2]).max()) * 1.05)
    zlo, zhi = all_targets[:, 2].min(), all_targets[:, 2].max()
    if zhi - zlo < 1e-6:
        zlo, zhi = zlo - .05, zhi + .05
    z_edges = np.linspace(zlo, zhi + 1e-9, 6)
    for index, report in enumerate(reports):
        valid = [e for e in report['episodes'] if e['valid']]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        if valid:
            targets = np.asarray([e['target'] for e in valid])
            ok = np.asarray([e['success'] for e in valid])
            radius = np.linalg.norm(targets[:, :2], axis=1)
            for mask, label, color in ((ok, 'Expert success', '#24766b'), (~ok, 'Expert failure', '#c34c44')):
                axes[0, 0].scatter(targets[mask, 0], targets[mask, 1], s=12, alpha=.65, color=color, label=label)
                axes[0, 1].scatter(radius[mask], targets[mask, 2], s=12, alpha=.65, color=color)
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].set(xlabel='Target X (m)', ylabel='Target Y (m)', xlim=(-xy_limit, xy_limit), ylim=(-xy_limit, xy_limit), aspect='equal')
            axes[0, 1].set(xlabel='Target radius from origin (m)', ylabel='Target Z (m)', ylim=(zlo, zhi))
            errors_mm = np.asarray([e['final_error_m'] for e in valid]) * 1000
            bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, max(3000, float(errors_mm.max()) + 1)]
            axes[0, 2].hist(errors_mm, bins=bins)
            axes[0, 2].set_xscale('symlog', linthresh=10)
            axes[0, 2].axvline(args.reach_tolerance * 1000, color='red', linestyle='--')
            axes[0, 2].set(xlabel='Post-action final error (mm; >10 mm log scale)', ylabel='Episodes')
            for ax, values, edges, xlabel in ((axes[1, 1], np.arctan2(targets[:, 1], targets[:, 0]), np.linspace(-np.pi, np.pi, 9), 'Target angle (deg)'),
                                                (axes[1, 2], targets[:, 2], z_edges, 'Target height (m)')):
                totals, _ = np.histogram(values, bins=edges)
                successes, _ = np.histogram(values[ok], bins=edges)
                rates = np.divide(successes * 100., totals, out=np.full(len(totals), np.nan), where=totals > 0)
                positions = np.arange(len(totals))
                ax.bar(positions, rates, color='#24766b')
                for j, (n, rate) in enumerate(zip(totals, rates)):
                    ax.text(j, min(105, rate + 3) if n else 5, f'n={n}', ha='center', fontsize=8)
                factor = 180 / np.pi if xlabel.startswith('Target angle') else 1
                labels = ([f'{edges[j] * factor:.0f}–{edges[j+1] * factor:.0f}' for j in positions]
                          if factor != 1 else [f'{edges[j]:.2f}–{edges[j+1]:.2f}' for j in positions])
                ax.set_xticks(positions, labels, rotation=35)
                ax.set(xlabel=xlabel, ylabel='Expert success (%)', ylim=(0, 115))
        counts = np.asarray(report['summary']['state_counts'])
        axes[1, 0].bar(np.arange(5), counts / max(counts.sum(), 1) * 100, color=['#24766b', '#619b8a', '#d98c32', '#e7b76b', '#71839b'])
        axes[1, 0].set_xticks(np.arange(5), ['Inside\nslow', 'Inside\nmoving', 'Near outside\nslow', 'Near outside\nmoving', 'Farther'], rotation=15)
        axes[1, 0].set(ylabel='Successful TRAIN Arm rows (%)')
        titles = ['Recorded target coverage', 'Radius / height coverage', 'Endpoint precision', 'Training state composition', 'Success by direction', 'Success by height']
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.grid(alpha=.15)
        fig.suptitle(f"{report['name']} | {report['status']} | scanned {report['scanned_episodes']}/{report['planned_episodes']} planned episodes", fontsize=15)
        path = output / f'dataset_{index + 1:02d}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        images.append(path)
    return images


def fmt(value, percent=False):
    if value is None:
        return 'N/A'
    return f'{value * 100:.2f}%' if percent else f'{value:.4g}' if isinstance(value, float) else str(value)


def json_safe(value):
    """Keep corrupt manifest NaNs from preventing a readable error report."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def comparison_rows(reports):
    rows = []
    for r in reports:
        s = r['summary']
        rows.append(dict(dataset=r['name'], status=r['status'], complete=r['collection_complete'],
            scanned_episodes=r['scanned_episodes'], planned_episodes=r['planned_episodes'],
            invalid_episodes=s['invalid_episodes'], error_checks=sum(i['level'] == 'error' for i in r['issues']),
            expert_success_rate=s['expert_success_rate'], successful_train_episodes=s['successful_train_episodes'],
            train_arm_rows=s['train_arm_rows'], critical_train_fraction=s['critical_train_fraction'],
            critical_episode_coverage=s['critical_episode_coverage'], near_duplicate_fraction=s['near_duplicate_fraction'],
            endpoint_p95_mm=None if s['final_error_m']['p95'] is None else s['final_error_m']['p95'] * 1000,
            neighbor_close_pairs=r['neighbor'].get('close_pairs'),
            neighbor_torque_rms_median_nm=r['neighbor'].get('close_pair_torque_rms_nm', {}).get('median'),
            merged_check=r['merged_check']['mode']))
    return rows


def write_outputs(reports, output, args):
    output.mkdir(parents=True, exist_ok=False)
    notes = [
        'This report checks internal record consistency and training-state coverage; it does not replace simulation replay or closed-loop evaluation on a fixed target set.',
        'There is no overall quality score: datasets with different target regions, trajectory counts, or completion levels cannot be ranked by expert success rate alone.',
        'Coverage, adjacent-duplicate, and neighboring-label metrics use only valid, successful TRAIN trajectories; expert success rate uses valid scanned trajectories.',
        'PRELIMINARY means collection or scanning is incomplete. PASS_SCANNED only means the checks performed found no hard errors; it does not mean all physical labels have been verified.',
        'By default, only core-field headers and field order in the merged file are checked; --check-merged additionally streams all core training arrays and compares them against individual trajectories.',
        'Test-split success rates are for dataset diagnostics only; do not use them to select training samples or repeatedly tune parameters.',
        'The near-duplicate rate is not a count of independent samples; torque differences between neighboring states may also reflect normal dynamics, internal expert references, or the distance definition.',
    ]
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        state_distance_scales=STATE_SCALES.tolist(), adjacent_repeat_tolerances=REPEAT_TOLERANCES.tolist(),
        notes=notes, datasets=reports)
    (output / 'metrics.json').write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    rows = comparison_rows(reports)
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'episodes.csv').open('w', newline='') as handle:
        fields = ['dataset', 'episode_id', 'split', 'success', 'valid', 'samples', 'arm_rows',
                  'duration_s', 'final_error_m', 'critical_rows', 'critical_seconds',
                  'adjacent_pairs', 'near_duplicate_pairs', 'arm_entry_linear_speed',
                  'arm_entry_angular_speed', 'reference_gap_rms_rad', 'stop_message']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in reports:
            writer.writerows(dict(dataset=r['name'], **e) for e in r['episodes'])
    images = plot_reports(reports, output, args)
    labels = [('Dataset', 'dataset', False), ('Check status', 'status', False), ('Scanned trajectories', 'scanned_episodes', False),
              ('Hard-error checks', 'error_checks', False), ('Expert success rate', 'expert_success_rate', True),
              ('Successful TRAIN trajectories', 'successful_train_episodes', False), ('Arm training frames', 'train_arm_rows', False),
              ('Near-target low-speed fraction', 'critical_train_fraction', True), ('Critical-state trajectory coverage', 'critical_episode_coverage', True),
              ('Adjacent near-duplicate rate', 'near_duplicate_fraction', True), ('Endpoint error P95 / mm', 'endpoint_p95_mm', False),
              ('Neighbor pairs', 'neighbor_close_pairs', False), ('Median neighboring torque RMS difference / N m', 'neighbor_torque_rms_median_nm', False)]
    header = ''.join(f'<th>{label}</th>' for label, _, _ in labels)
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(fmt(row[key], pct))}</td>' for _, key, pct in labels) + '</tr>' for row in rows)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Robot Dataset Quality Report</title>',
            '<style>body{font:15px system-ui;margin:32px;color:#23313c;max-width:1500px}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccd4da;padding:8px;text-align:left}th{background:#edf3f5}img{width:100%;height:auto}li{margin:8px 0}pre{white-space:pre-wrap;background:#f4f6f7;padding:16px}h2{margin-top:36px}.scroll{overflow:auto}</style>',
            '<h1>Robot Dataset Quality and Training Suitability Diagnostics</h1>',
            '<ul>' + ''.join('<li>' + html.escape(n) + '</li>' for n in notes) + '</ul>',
            f'<p>Critical-state definition: end-effector distance [{args.reach_tolerance:g}, {args.near_distance:g}) m, maximum absolute joint speed &lt; {args.low_speed:g} rad/s. This definition is for diagnostics and does not modify data labels.</p>',
            f'<div class="scroll"><table><tr>{header}</tr>{table}</table></div>']
    for report, image_path in zip(reports, images):
        body.append(f'<h2>{html.escape(report["name"])} — {report["status"]}</h2>')
        body.append(f'<p>{html.escape(report["folder"])}<br>Completed trajectory snapshot: {report["available_episodes"]}/{report["planned_episodes"]}; scanned: {report["scanned_episodes"]}; merged-file check: {report["merged_check"]["mode"]}</p>')
        body.append('<p>Target region: ' + html.escape(json.dumps(report['region'], ensure_ascii=False)) + '</p>')
        body.append('<img alt="Dataset diagnostics" src="data:image/png;base64,' + base64.b64encode(image_path.read_bytes()).decode() + '">')
        for key in ('issue_counts', 'neighbor', 'reference_coverage', 'source_hash_status'):
            if key in report:
                body.append('<details><summary>' + key + '</summary><pre>' + html.escape(json.dumps(report[key], indent=2, ensure_ascii=False)) + '</pre></details>')
        body.append('<details><summary>Findings (first 50; see metrics.json for the complete list)</summary><pre>' + html.escape(json.dumps(report['issues'][:50], indent=2, ensure_ascii=False)) + '</pre></details>')
    body.append('<h2>How to Select Data</h2><ol><li>Address hard errors and target-split leakage first.</li><li>Compare coverage and critical-state fractions only for completed datasets with the same task region and similar trajectory counts.</li><li>Optionally pass --reference-targets to measure distances from a common target set to the nearest successful training targets; this measures geometric coverage, not predicted success rate.</li><li>Finally, confirm with closed-loop comparisons using the same model, training budget, and fixed validation targets.</li></ol></html>')
    (output / 'report.html').write_text('\n'.join(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', metavar='NAME=PATH', help='Repeat to compare; PATH is a dataset directory or dataset.npz')
    parser.add_argument('--output', type=Path, help='New report directory; never overwrites an existing report')
    parser.add_argument('--max-episodes', type=int, default=0, help='0 scans all completed trajectories; positive means a seeded sample')
    parser.add_argument('--check-merged', action='store_true', help='Also stream-compare core training arrays in dataset.npz against every episode')
    parser.add_argument('--reference-targets', type=Path, help='Optional common targets.json/settings.json/results.json for geometric coverage comparison')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rows-per-episode', type=int, default=24, help='Max sampled TRAIN Arm rows per trajectory for neighbor checks')
    parser.add_argument('--neighbor-rows', type=int, default=6000, help='Maximum total rows for cross-trajectory neighbor checks')
    parser.add_argument('--neighbor-radius', type=float, default=.1, help='Close-pair RMS distance in fixed scaled 21D state space')
    parser.add_argument('--reach-tolerance', type=float, default=.01, help='Diagnostic reach distance in m; used for old datasets lacking recorded tolerance')
    parser.add_argument('--near-distance', type=float, default=.03)
    parser.add_argument('--low-speed', type=float, default=.03, help='Diagnostic maximum absolute joint speed in rad/s')
    args = parser.parse_args()
    if args.max_episodes < 0 or args.rows_per_episode < 1 or args.neighbor_rows < 2 or args.seed < 0:
        parser.error('Invalid sample counts or seed')
    if not (0 < args.reach_tolerance < args.near_distance < np.inf and 0 < args.low_speed < np.inf and 0 < args.neighbor_radius < np.inf):
        parser.error('Require finite positive thresholds, reach-tolerance < near-distance')
    if args.dataset is None:
        candidates = sorted((ROOT / 'data').glob('expert_*'), reverse=True)
        candidates = [p for p in candidates if (p / 'episodes.json').exists()]
        if not candidates:
            parser.error('No expert dataset found; specify --dataset NAME=PATH')
        inputs = [(candidates[0].name, candidates[0])]
    else:
        inputs = []
        for spec in args.dataset:
            name, path = spec.split('=', 1) if '=' in spec else ('', spec)
            folder = Path(path).resolve()
            if folder.name == 'dataset.npz':
                folder = folder.parent
            inputs.append((name or folder.name, folder))
    if len({name for name, _ in inputs}) != len(inputs):
        parser.error('Dataset display names must be unique')
    try:
        reports = [audit_dataset(name, folder, args) for name, folder in inputs]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.reference_targets:
        targets = load_reference(args.reference_targets)
        for report in reports:
            report['reference_coverage'] = reference_coverage(report, targets)
    output = args.output or ROOT / 'outputs' / ('data_quality_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    write_outputs(reports, output, args)
    for row in comparison_rows(reports):
        print(f"{row['dataset']}: {row['status']}; expert success={fmt(row['expert_success_rate'], True)}; "
              f"successful TRAIN={row['successful_train_episodes']}; critical rows={fmt(row['critical_train_fraction'], True)}; "
              f"hard errors={row['error_checks']}", flush=True)
    print(f'Report: {output.resolve() / "report.html"}\nMetrics: {output.resolve() / "metrics.json"}')
    if any(r['status'] == 'ERROR' for r in reports):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
