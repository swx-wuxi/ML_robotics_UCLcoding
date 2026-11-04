"""Run with RoboEnv Python: python -m unittest discover -s tests -v."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final_project/robotnik_arm_task"))
import performance_evaluation as perf


class PerformanceEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Keep pulse timing assertions independent of the user's duration setting.
        duration = patch.object(perf, "PERTURBATION_DURATION_S", 0.20)
        duration.start()
        self.addCleanup(duration.stop)

    def test_targets_within_absolute_radius_and_random_height_ranges(self):
        metadata = {"target_sampling": dict(mode="uniform_area_annulus", center_xy=[3, -2],
                                             min_radius=1.2, max_radius=2.4, height=0.7)}
        targets = perf.sample_generalization_targets(metadata, 101, 123)
        radii = np.linalg.norm(targets[:, :2] - [3, -2], axis=1)
        low, high = perf.GENERALIZATION_RADIUS_RANGE_M
        self.assertTrue(np.all((radii >= low) & (radii <= high)))
        z_low, z_high = perf.GENERALIZATION_HEIGHT_OFFSETS_M
        self.assertTrue(np.all((targets[:, 2] >= 0.7 + z_low)
                               & (targets[:, 2] <= 0.7 + z_high)))
        self.assertGreater(len(np.unique(targets[:, 2])), 2)
        np.testing.assert_array_equal(targets, perf.sample_generalization_targets(metadata, 101, 123))

    def test_noise_is_arm_relative_and_only_changes_policy_joint_inputs(self):
        pulse = perf.ArmPerturbation(0)
        original = np.arange(21, dtype=float)
        clean = original.copy()
        np.testing.assert_array_equal(pulse.observe(original, 10.24, 10.0, 0.01), original)
        noisy = pulse.observe(original, 10.25, 10.0, 0.01)
        self.assertEqual(pulse.event_id, 0)
        np.testing.assert_array_equal(original, clean)
        np.testing.assert_array_equal(noisy[:4], original[:4])
        np.testing.assert_array_equal(noisy[18:], original[18:])
        self.assertTrue(np.any(noisy[4:18] != original[4:18]))
        np.testing.assert_array_equal(pulse.observe(original, 10.45, 10.0, 0.01), original)
        self.assertEqual(pulse.event_id, -1)

    def test_reproducible_noise_and_distinct_episode_seeds(self):
        state = np.zeros(21)
        a, b, c = [perf.ArmPerturbation(i) for i in (3, 3, 4)]
        x = a.observe(state, 0.25, 0.0, 0.01)
        np.testing.assert_array_equal(x, b.observe(state, 0.25, 0.0, 0.01))
        self.assertFalse(np.array_equal(x, c.observe(state, 0.25, 0.0, 0.01)))

    def test_force_reapplied_to_end_effector_each_active_step(self):
        client = Mock(WORLD_FRAME=2)
        client.getNumJoints.return_value = 1
        info = [None] * 13
        info[12] = b"robot_j2s7s300_end_effector"
        client.getJointInfo.return_value = info
        client.getLinkState.return_value = ([1, 2, 3],)
        pulse = perf.ArmPerturbation(0)
        pulse.bind(client, 7)
        for now in (0.24, 0.25, 0.26, 0.45):
            pulse.observe(np.zeros(21), now, 0.0, 0.01)
            pulse.before_step()
        self.assertEqual(client.applyExternalForce.call_count, 2)
        client.applyExternalForce.assert_called_with(7, 0, [4.0, 0.0, 0.0], [1, 2, 3], 2)

    def test_early_stop_reports_partial_and_untriggered_events(self):
        pulse = perf.ArmPerturbation(0)
        times = np.arange(30) * 0.01
        for now in times:
            pulse.observe(np.zeros(21), now, 0.0, 0.01)
            pulse.record(np.zeros(21))
        extra, report = pulse.finish(dict(time=times, ee_error=np.ones(30)),
                                     dict(final_error_m=0.5, duration_s=0.3), 0.01)
        self.assertEqual(extra, {})
        np.testing.assert_array_equal(pulse.applied_steps, [5, 0, 0])
        self.assertFalse(report["all_events_completed"])

    def test_complete_pulses(self):
        pulse = perf.ArmPerturbation(0)
        times = np.arange(250) * 0.01
        for now in times:
            pulse.observe(np.zeros(21), now, 0.0, 0.01)
            pulse.record(np.zeros(21))
        _, report = pulse.finish({}, {}, 0.01)
        self.assertTrue(report["all_events_completed"])
        np.testing.assert_array_equal(pulse.applied_steps, [20, 20, 20])

    def test_both_switches_off_do_not_load_model(self):
        with patch.object(perf, "TEST_GENERALIZATION", False), \
                patch.object(perf, "TEST_ROBUSTNESS", False), \
                patch.object(perf.evaluation, "load_model") as load:
            perf.main()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
