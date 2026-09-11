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
            for _ in range(2):
                d=DAgger(str(source), random_state=3)
                scores,_=d.run(N=2,test_fold=8,excluded_folds=(9,),plot=False,output_dir=tmp,evaluation_role='validation')
                cfg=json.loads((d.last_run_dir/'config.json').read_text())
                with (d.last_run_dir/'metrics.csv').open(encoding='utf-8') as file:
                    metrics=list(csv.DictReader(file))
                self.assertEqual(cfg['train_folds'],list(range(8)))
                self.assertEqual(cfg['status'],'complete')
                self.assertEqual([int(r['dataset_size']) for r in metrics],[32,32,64])
                self.assertEqual([r['method'] for r in metrics],['no_structure','structured_bc','dagger'])
                for r in metrics:
                    self.assertAlmostEqual(float(r['accuracy'])+float(r['imitation_error']),1)
                    for key in ['training_seconds','rollout_seconds','evaluation_seconds']:
                        self.assertGreaterEqual(float(r[key]),0)
                results.append(scores)
            np.testing.assert_array_equal(*results)

if __name__=='__main__':
    unittest.main()
