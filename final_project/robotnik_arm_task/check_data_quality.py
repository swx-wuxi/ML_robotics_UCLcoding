"""Audit saved expert data without importing controllers or starting simulation.

Run: python check_data_quality.py
Set DATA_DIR to audit a specific completed collection. Source data are read only.
"""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = None
# Full merged-file verification reads all uncompressed array bytes and may take time.
CHECK_MERGED_VALUES = True
REACH_TOLERANCE = 0.01
FEATURE_NAMES = (["base_x", "base_y", "base_z", "base_yaw"]
                 + [f"arm_q_{i}" for i in range(1, 8)]
                 + [f"arm_qd_{i}" for i in range(1, 8)]
                 + ["target_x", "target_y", "target_z"])
ACTION_NAMES = (["wheel_FR", "wheel_FL", "wheel_BL", "wheel_BR"]
                + [f"arm_torque_{i}" for i in range(1, 8)])
SHAPES = {"X": (21,), "y": (11,), "time": (), "ee_pos": (3,),
          "q_mes": (17,), "qd_mes": (17,), "phase_before": (),
          "phase_after": (), "nav_done_before": (), "nav_done_after": (),
          "q_ref_before": (7,), "ee_error": (), "episode_id": (),
          "split": (), "episode_success": ()}


def find_folder():
    if DATA_DIR is not None:
        return Path(DATA_DIR)
    for folder in sorted((ROOT / "data").glob("expert_*"), reverse=True):
        if all((folder / name).is_file() for name in
               ("metadata.json", "episodes.json", "dataset.npz")):
            return folder
    raise FileNotFoundError("No completed collection found; set DATA_DIR explicitly")


def main():
    folder = find_folder()
    print(f"Auditing: {folder}", flush=True)
    metadata = json.loads((folder / "metadata.json").read_text())
    summaries = json.loads((folder / "episodes.json").read_text())
    errors, warnings = [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(summaries) == metadata["episodes"], "Manifest episode count mismatch")
    ids = [item["episode_id"] for item in summaries]
    check(len(set(ids)) == len(ids), "Duplicate manifest episode IDs")
    paths = [folder / f"episode_{i:04d}.npz" for i in ids]
    check(len(list(folder.glob("episode_*.npz"))) == len(paths),
          "Episode file count differs from manifest")
    hashes = {key: hashlib.sha256() for key in SHAPES}
    dtypes = {}
    split_stats = {str(i): dict(episodes=0, successes=0, samples=0) for i in range(3)}
    targets_by_split = [set(), set(), set()]
    initial = None
    total = 0
    failures = []
    final_errors = []
    action_min = np.full(11, np.inf)
    action_max = np.full(11, -np.inf)
    torque_saturated = 0
    phases = {}
    durations = []
    for number, (path, item) in enumerate(zip(paths, summaries)):
        prefix = path.name
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in SHAPES}
                check(archive["feature_names"].tolist() == FEATURE_NAMES,
                      f"{prefix}: feature names/order mismatch")
                check(archive["action_names"].tolist() == ACTION_NAMES,
                      f"{prefix}: action names/order mismatch")
            n = len(data["X"])
            check(n > 0 and n == item["samples"], f"{prefix}: sample count mismatch")
            valid_shapes = all(data[k].shape == (n, *tail) for k, tail in SHAPES.items())
            check(valid_shapes, f"{prefix}: invalid array shapes")
            if not valid_shapes or not n:
                continue
            for key, a in data.items():
                if a.dtype.kind in "fiu":
                    check(np.isfinite(a).all(), f"{prefix}: non-finite {key}")
                if key in dtypes:
                    check(a.dtype == dtypes[key], f"{prefix}: dtype changed for {key}")
                else:
                    dtypes[key] = a.dtype
                hashes[key].update(a.tobytes(order="C"))
            X, y, time = data["X"], data["y"], data["time"]
            split = int(item["split"])
            check(split in (0, 1, 2), f"{prefix}: invalid split code")
            check(np.all(data["split"] == split), f"{prefix}: mixed/wrong split")
            check(np.all(data["episode_id"] == item["episode_id"]), f"{prefix}: wrong ID")
            check(np.all(data["episode_success"] == item["success"]),
                  f"{prefix}: inconsistent success flags")
            check(np.isclose(time[0], 0, atol=1e-9), f"{prefix}: time does not start at zero")
            check(np.allclose(np.diff(time), item["time_step"], rtol=1e-6, atol=1e-9),
                  f"{prefix}: non-contiguous timestamps")
            check(np.allclose(X[:, 18:21], item["target"], atol=1e-10, rtol=0),
                  f"{prefix}: target changes or disagrees with manifest")
            check(np.allclose(X[:, 4:11], data["q_mes"][:, 4:11]),
                  f"{prefix}: arm position columns disagree")
            check(np.allclose(X[:, 11:18], data["qd_mes"][:, 4:11]),
                  f"{prefix}: arm velocity columns disagree")
            check(np.allclose(data["ee_error"],
                              np.linalg.norm(data["ee_pos"] - X[:, 18:21], axis=1)),
                  f"{prefix}: stored EE errors disagree")
            state = np.concatenate((X[0, :18], data["q_mes"][0], data["qd_mes"][0]))
            if initial is None:
                initial = state
            check(np.allclose(state, initial, atol=1e-7, rtol=0),
                  f"{prefix}: initial measured state differs from first episode")
            check(np.allclose(y[:, 0], y[:, 3]) and np.allclose(y[:, 1], y[:, 2]),
                  f"{prefix}: paired wheel commands disagree")
            torque_limit = metadata["arm_params"]["max_torque"]
            check(np.all(np.abs(y[:, 4:]) <= torque_limit + 1e-8),
                  f"{prefix}: expert arm torque limit exceeded")
            final_error = float(np.linalg.norm(np.asarray(item["final_ee_pos"])
                                               - np.asarray(item["target"])))
            check(np.isclose(final_error, item["final_error"]),
                  f"{prefix}: final error disagrees with final EE position")
            check(bool(item["success"]) == item["stop_message"].startswith("Target reached"),
                  f"{prefix}: success disagrees with stop message")
            if item["success"] and final_error > REACH_TOLERANCE:
                warnings.append(f"{prefix}: post-action final error exceeds tolerance")
            if not item["success"]:
                failures.append(dict(episode_id=item["episode_id"],
                                     final_error=final_error, reason=item["stop_message"]))
            if split in (0, 1, 2):
                stats = split_stats[str(split)]
                stats["episodes"] += 1
                stats["successes"] += int(item["success"])
                stats["samples"] += n
                targets_by_split[split].add(tuple(item["target"]))
            region = metadata.get("target_sampling", {})
            if region.get("mode") == "uniform_area_annulus":
                target = np.asarray(item["target"])
                radius = np.linalg.norm(target[:2] - region["center_xy"])
                check(region["min_radius"] - 1e-9 <= radius <= region["max_radius"] + 1e-9
                      and np.isclose(target[2], region["height"]),
                      f"{prefix}: target outside declared sampling region")
            for phase, count in zip(*np.unique(data["phase_before"], return_counts=True)):
                phases[str(phase)] = phases.get(str(phase), 0) + int(count)
            action_min = np.minimum(action_min, y.min(axis=0))
            action_max = np.maximum(action_max, y.max(axis=0))
            torque_saturated += int(np.count_nonzero(np.abs(y[:, 4:]) >= torque_limit - 1e-8))
            final_errors.append(final_error)
            durations.append(float(time[-1] + item["time_step"]))
            total += n
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            errors.append(f"{prefix}: cannot fully inspect: {exc}")
        if (number + 1) % 50 == 0 or number + 1 == len(paths):
            print(f"Checked {number + 1}/{len(paths)} episode files", flush=True)

    check(total == metadata["samples"], "Total sample count disagrees with metadata")
    for i in range(3):
        if not split_stats[str(i)]["successes"]:
            warnings.append(f"Split {i} has no successful episodes")
        for j in range(i):
            check(not targets_by_split[i] & targets_by_split[j],
                  f"Target leakage between splits {i} and {j}")
    # Stream merged arrays one field at a time; verify exact concatenation hashes.
    try:
        with zipfile.ZipFile(folder / "dataset.npz") as archive:
            for key, tail in SHAPES.items():
                print(f"Checking merged field: {key}", flush=True)
                with archive.open(f"{key}.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported NPY version: {version}")
                    check(shape == (total, *tail), f"Merged {key}: wrong shape")
                    check(dtype == dtypes.get(key) and not fortran,
                          f"Merged {key}: wrong dtype or storage order")
                    if CHECK_MERGED_VALUES:
                        digest = hashlib.sha256()
                        while True:
                            chunk = stream.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                        check(digest.digest() == hashes[key].digest(),
                              f"Merged {key}: values differ from episode concatenation")
        with np.load(folder / "dataset.npz", allow_pickle=False) as archive:
            check(archive["feature_names"].tolist() == FEATURE_NAMES, "Merged feature names mismatch")
            check(archive["action_names"].tolist() == ACTION_NAMES, "Merged action names mismatch")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        errors.append(f"Merged archive: {exc}")

    report = dict(dataset=str(folder), errors=errors, warnings=warnings,
                  merged_values_checked=CHECK_MERGED_VALUES,
                  checked_samples=total, checked_episodes=len(final_errors),
                  successful_episodes=len(final_errors) - len(failures),
                  success_rate=(len(final_errors) - len(failures)) / len(final_errors) if final_errors else None,
                  splits=split_stats, failures=failures, phase_samples=phases,
                  final_error_quantiles_m=np.quantile(final_errors, [0, .5, .95, 1]).tolist() if final_errors else [],
                  duration_range_s=[min(durations), max(durations)] if durations else [],
                  action_names=ACTION_NAMES,
                  action_min=action_min.tolist() if final_errors else [],
                  action_max=action_max.tolist() if final_errors else [],
                  arm_torque_saturation_fraction=torque_saturated / (total * 7) if total else None,
                  limitations="Checks stored consistency, not physical correctness, actual state-action timing, "
                              "full orientation equality, or closed-loop ML performance.")
    output = folder / "quality_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nErrors: {len(errors)}; warnings: {len(warnings)}")
    print(f"Successful episodes: {report['successful_episodes']}/{len(final_errors)}")
    print("Final EE error [min, median, p95, max] (m):", report["final_error_quantiles_m"])
    print("Split counts:", split_stats)
    for message in (errors + warnings)[:20]:
        print(message)
    print(f"Full report: {output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
