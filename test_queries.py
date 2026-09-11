import tempfile
import unittest
from pathlib import Path
import numpy as np
from label_budget_dagger import LabelBudgetDAgger as DAgger


class GuardLabels:
    def __init__(self, allowed):
        self.allowed = allowed
        self.reads = []
    def __getitem__(self, index):
        if index not in self.allowed:
            raise AssertionError("Unqueried label accessed")
        self.reads.append(index)
        return 1


class Policy:
    def __init__(self):
        self.contexts = []
    def predict(self, state):
        self.contexts.append(state.toarray()[0,128:])
        return np.array([0])
    def predict_proba(self, state):
        return np.array([[0.5,0.5]])


class QueryChecks(unittest.TestCase):
    def learner(self, allowed):
        d = DAgger()
        d.words = [[np.zeros(128) for _ in range(5)], [np.zeros(128)]]
        d.words_fold = [0,9]
        d.sequences = [GuardLabels(allowed), GuardLabels(set())]
        return d

    def test_standard_cap_and_continued_rollout(self):
        d = self.learner({0,1})
        p = Policy()
        data = [[],[]]
        d.aggregate_dataset(p,data,query_budget=2)
        self.assertEqual(d.sequences[0].reads,[0,1])
        self.assertEqual(len(p.contexts),5)
        self.assertEqual(len(data[0]),2)
        self.assertEqual(p.contexts[0].sum(),0)
        self.assertTrue(all(c[0] == 1 for c in p.contexts[1:]))

    def test_periodic_continues_across_rollouts(self):
        d = self.learner({2})
        d.aggregate_dataset(Policy(),[[],[]],query_strategy="periodic",query_period=3)
        self.assertEqual(d.sequences[0].reads,[2])
        d.sequences[0] = GuardLabels({0,3})
        d.aggregate_dataset(Policy(),[[],[]],query_strategy="periodic",query_period=3,visited_offset=5)
        self.assertEqual(d.sequences[0].reads,[0,3])

    def test_uncertainty_boundary_and_zero_budget(self):
        for threshold,budget,allowed in [(0.5,2,{0,1}),(0.6,5,set()),(0,0,set())]:
            d = self.learner(allowed)
            d.aggregate_dataset(Policy(),[[],[]],query_strategy="uncertainty",
                                uncertainty_threshold=threshold,query_budget=budget)
            self.assertEqual(d.last_rollout_queries,len(allowed))
            self.assertEqual(d.last_rollout_visited,5)

    def test_invalid_options(self):
        for strategy,budget,period,threshold,beta in [
            ("wrong",None,1,0.5,0),("standard",-1,1,0.5,0),
            ("periodic",None,0,0.5,0),("uncertainty",None,1,float('nan'),0),
            ("standard",2,1,0.5,1),("periodic",None,1,0.5,0.1)]:
            with self.assertRaises(ValueError):
                DAgger._validate_queries(strategy,budget,period,threshold,beta)

    def test_run_budget_reset_and_no_query_policy_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"tiny.data"
            rows=[]
            for fold in range(10):
                for label in range(2):
                    rows.append("\t".join([str(len(rows)),"ab"[label],"-1","0","0",str(fold)]+[str(label)]*128))
            source.write_text("\n".join(rows))
            d=DAgger(str(source))
            for budget in [3,0,3]:
                scores,policies=d.run(N=3,plot=False,query_budget=budget,output_dir=tmp)
                records=d.last_run_records
                self.assertEqual([r['cumulative_queries'] for r in records],[0,0,budget,budget])
                self.assertEqual(records[-1]['total_labels_used'],18+budget)
                self.assertEqual(records[-1]['remaining_queries'],0)
                self.assertEqual(records[-1]['cumulative_visited_states'],36)
                self.assertIs(policies[-1],policies[-2])
                if budget == 0:
                    np.testing.assert_array_equal(scores,np.full(3,scores[0]))


if __name__ == "__main__":
    unittest.main()
