import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dagger import DAgger

class BaselineChecks(unittest.TestCase):
    def test_repeatability_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tiny.data'
            rows=[]
            for fold in range(10):
                for y in range(2):
                    for pos in range(2):
                        rows.append('\t'.join([str(len(rows)), 'ab'[y], '-1' if pos else '1', '0', str(pos), str(fold)] + [str(y)]*128))
            source.write_text('\n'.join(rows))
            results=[]
            d=DAgger(str(source), random_state=3)
            for _ in range(2):
                scores,_=d.run(N=3,test_fold=8,excluded_folds=(9,),plot=False,output_dir=tmp,evaluation_role='validation')
                cfg=json.loads((d.last_run_dir/'config.json').read_text())
                with (d.last_run_dir/'metrics.csv').open(encoding='utf-8') as file:
                    metrics=list(csv.DictReader(file))
                self.assertEqual(cfg['train_folds'],list(range(8)))
                self.assertEqual(cfg['status'],'complete')
                self.assertEqual([int(r['dataset_size']) for r in metrics],[32,32,64,96])
                self.assertEqual([r['method'] for r in metrics],['no_structure','structured_bc','dagger','dagger'])
                for key, expected in {
                    'initial_labels': [32,32,32,32],
                    'queries_this_iteration': [0,0,32,32],
                    'cumulative_queries': [0,0,32,64],
                    'total_labels_used': [32,32,64,96],
                }.items():
                    self.assertEqual([int(r[key]) for r in metrics], expected)
                for r in metrics:
                    self.assertAlmostEqual(float(r['accuracy'])+float(r['imitation_error']),1)
                    for key in ['training_seconds','rollout_seconds','evaluation_seconds']:
                        self.assertGreaterEqual(float(r[key]),0)
                results.append(scores)
            np.testing.assert_array_equal(*results)
            # A fresh run on the same object resets all query accounting.
            d.run(N=1, test_fold=9, plot=False, output_dir=tmp)
            for row in d.last_run_records:
                self.assertEqual(row['initial_labels'],36)
                self.assertEqual(row['queries_this_iteration'],0)
                self.assertEqual(row['cumulative_queries'],0)
                self.assertEqual(row['total_labels_used'],36)
            # Expert-controlled rollouts reuse the label for the action.
            d.run(N=2, test_fold=9, beta_decay=1, plot=False, output_dir=tmp)
            self.assertEqual(d.last_run_records[-1]['queries_this_iteration'],36)
            self.assertEqual(d.last_run_records[-1]['total_labels_used'],72)

if __name__=='__main__':
    unittest.main()
