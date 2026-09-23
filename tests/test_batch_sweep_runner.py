"""CPU-only orchestration checks; runnable directly with the standard library."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_batch_sizes as sweep


class BatchSweepTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((sweep.ROOT / 'configs/batch_size_sweep.json').read_text())
        self.jobs, self.tiles = sweep.prepare(self.config)

    def test_grid_and_fixed_baseline_tiles(self):
        self.assertEqual(len(self.jobs), 90)
        self.assertEqual(len({sweep.case_key(j) for j in self.jobs}), 90)
        groups = {}
        for job in self.jobs:
            if job['implementation'] == 'naive':
                self.assertIsNone(job['fixed_tile'])
                continue
            key = (job['implementation'], min(job['sequence_length'], 2048), job['dtype'])
            groups.setdefault(key, set()).add(tuple(job['fixed_tile']))
        self.assertTrue(all(len(choices) == 1 for choices in groups.values()))
        self.assertEqual(len(groups), 8)

    def test_invalid_grid_and_missing_tiles_fail_before_launch(self):
        config = copy.deepcopy(self.config)
        config['batch_sizes'] = [1, 1]
        with self.assertRaises(ValueError):
            sweep.prepare(config)
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            config['baseline_directory'] = temporary
            with self.assertRaises(FileNotFoundError):
                sweep.prepare(config)

    def test_worker_crash_discards_stale_checkpoint_and_resume_skips_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for subdir in ('cases', 'jobs', 'logs'):
                (directory / subdir).mkdir()
            job = self.jobs[0]
            result = directory / 'cases' / (sweep.case_key(job) + '.json')
            sweep.save_json(result, dict(status='forward_done', forward_ms=999))
            with patch.object(sweep.subprocess, 'Popen') as popen:
                popen.return_value.returncode = 1
                row = sweep.run_case(directory, job, 0, True)
            self.assertEqual(row['status'], 'error')
            self.assertNotIn('forward_ms', row)
            with patch.object(sweep.subprocess, 'Popen') as popen:
                self.assertEqual(sweep.run_case(directory, job, 0, True), row)
                popen.assert_not_called()

    def test_partial_failure_and_sample_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = dict(self.jobs[0], status='oom', stage='backward_warmup', forward_ms=1.2, forward_samples=1,
                       validation={'batch_size': 1})
            sweep.write_results(directory, [row], 5)
            review = json.loads((directory / 'review.json').read_text())
            self.assertEqual(review['failures'][0]['status'], 'oom')
            self.assertEqual(review['low_sample_counts'][0]['phase'], 'forward')
            self.assertEqual(review['missing_validation'], [])
            import csv
            with (directory / 'results.csv').open() as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved['forward_ms'], '1.2')
            self.assertEqual(saved['tokens_per_second'], '')


if __name__ == '__main__':
    unittest.main()
