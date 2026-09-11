import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from dagger import DAgger
from ablation_retention import RetentionDAgger


class RetentionChecks(unittest.TestCase):
    def test_modes_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'tiny.data'
            rows = []
            for fold in range(10):
                for label in range(2):
                    rows.append('\t'.join([str(len(rows)), 'ab'[label], '-1', '0', '0', str(fold)] + [str(label)]*128))
            source.write_text('\n'.join(rows))
            original = DAgger(str(source), alpha=1e-5)
            expected, models = original.run(N=5, plot=False, output_dir=tmp)
            for mode, sizes in [('all',[18,18,36,54,72,90]), ('recent',[18,18,36,54,54,54]), ('latest',[18,18,36,36,36,36])]:
                d = RetentionDAgger(str(source), alpha=1e-5, retention=mode, keep_last=2)
                scores, policies = d.run(N=5, plot=False, output_dir=tmp)
                self.assertEqual([r['dataset_size'] for r in d.last_run_records], sizes)
                self.assertEqual([r['cumulative_queries'] for r in d.last_run_records], [0,0,18,36,54,72])
                self.assertEqual(d.last_run_records[-1]['total_labels_used'],90)
                cfg = json.loads((d.last_run_dir/'config.json').read_text())
                self.assertEqual(cfg['retention'],mode)
                self.assertEqual(cfg['status'],'complete')
                if mode == 'all':
                    np.testing.assert_array_equal(scores,expected)
                    for p,m in zip(policies,models):
                        np.testing.assert_array_equal(p.coef_,m.coef_)
                d.run(N=2, test_fold=8, excluded_folds=(9,), plot=False, output_dir=tmp)
                self.assertEqual(d.last_run_records[-1]['dataset_size'],32)
                self.assertEqual(d.last_run_records[-1]['cumulative_queries'],16)

    def test_keeps_initial_and_latest_batches(self):
        def collect(self, policy, data, *args, **kwargs):
            data[0].append(policy)
            data[1].append(policy)
            return data
        for mode, expected in [('latest',['initial','third']),('recent',['initial','second','third'])]:
            d = RetentionDAgger(retention=mode,keep_last=2)
            data = [['initial'],['initial']]
            with patch.object(DAgger,'aggregate_dataset',collect):
                for label in ['first','second','third']:
                    d.aggregate_dataset(label,data)
            self.assertEqual(data,[expected,expected])

    def test_rejects_confounded_experiment(self):
        with self.assertRaises(ValueError):
            RetentionDAgger().run(query_budget=10)
        with self.assertRaises(ValueError):
            RetentionDAgger(keep_last=0)


if __name__ == '__main__':
    unittest.main()
