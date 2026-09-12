import unittest
import tempfile
from pathlib import Path
import numpy as np
from dagger import DAgger
from ablation_position import evaluate_positions, PositionDAgger, plot_positions


class PositionChecks(unittest.TestCase):
    def test_unequal_lengths_and_predicted_context(self):
        d = DAgger()
        d.words = [[np.zeros(128)]*3, [np.zeros(128)], [np.zeros(128)]*5]
        d.sequences = [[1,0,1], [0], [1]*5]
        d.words_fold = [9,9,0]
        class Policy:
            def __init__(self): self.contexts = []
            def predict(self, state):
                self.contexts.append(state.toarray()[0,128:].copy())
                return [0]
        p = Policy()
        rows, score = evaluate_positions(d,p)
        self.assertEqual([r['count'] for r in rows],[2,1,1])
        self.assertEqual([r['correct'] for r in rows],[1,1,0])
        self.assertEqual(score,0.5)
        self.assertEqual(p.contexts[0].sum(),0)
        self.assertEqual(p.contexts[1][0],1)  # prediction 0, not ground truth 1
        self.assertEqual(p.contexts[3].sum(),0)  # reset at next word
        self.assertEqual(score,d.evaluate_policy(Policy()))
        with self.assertRaises(ValueError): evaluate_positions(d,p,test_fold=8)

    def test_saved_counts_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.data'
            rows=[]
            for fold in range(10):
                for label in range(2):
                    rows.append('\t'.join([str(len(rows)),'ab'[label],'-1','0','0',str(fold)]+[str(label)]*128))
            path.write_text('\n'.join(rows))
            d=PositionDAgger(str(path))
            for _ in range(2):
                scores,_=d.run(N=2,plot=False,output_dir=tmp)
                self.assertEqual([r['iteration'] for r in d.position_records],[1,2])
                self.assertEqual([r['count'] for r in d.position_records],[2,2])
                np.testing.assert_array_equal(scores,[r['accuracy'] for r in d.position_records])
                self.assertTrue((d.last_run_dir/'position_metrics.csv').exists())
            plot_positions(d.position_records,d.last_run_dir,show=False)
            self.assertTrue((d.last_run_dir/'accuracy_by_position.png').exists())


if __name__ == '__main__': unittest.main()
