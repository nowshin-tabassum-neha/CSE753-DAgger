import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lstm_policy import LSTMPolicy, expert_episodes, rollout, run_study


class LSTMChecks(unittest.TestCase):
    def test_memory_and_step_equivalence(self):
        torch.manual_seed(0)
        model=LSTMPolicy(8).eval()
        x=torch.randn(1,3,154)
        full,_=model(x)
        hidden=None
        steps=[]
        for t in range(3):
            logits,hidden=model(x[:,t:t+1],hidden)
            steps.append(logits)
        torch.testing.assert_close(full,torch.cat(steps,dim=1))
        fresh,_=model(x[:,2:3])
        self.assertFalse(torch.allclose(fresh,full[:,2:3]))

    def test_rollout_context_batch_invariance_and_labels(self):
        torch.manual_seed(0)
        model=LSTMPolicy(8)
        with torch.no_grad():
            model.head.weight.zero_();model.head.bias.zero_();model.head.bias[0]=10
        words=[np.ones((3,128),dtype=np.float32),np.zeros((1,128),dtype=np.float32)]
        labels=[[1,2,3],[4]]
        a,metrics=rollout(model,words,labels,batch_size=2,collect=True)
        b,other=rollout(model,words,labels,batch_size=1,collect=True)
        self.assertEqual(metrics,other)
        self.assertEqual(metrics['characters'],4)
        for left,right in zip(a,b):torch.testing.assert_close(left[0],right[0])
        self.assertEqual(a[0][0][0,128:].sum().item(),0)
        self.assertEqual(a[1][0][0,128:].sum().item(),0)
        self.assertEqual(a[0][0][1,128].item(),1)
        self.assertEqual(a[0][1].tolist(),labels[0])
        expert=expert_episodes(words,labels)
        self.assertEqual(expert[0][0][1,129].item(),1)

    def test_training_and_saved_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tiny.data'
            rows=[]
            for fold in range(10):
                for label in range(2):
                    for position in range(2):
                        rows.append('\t'.join([str(len(rows)),'ab'[label],'-1' if position else '1',
                                                '0',str(position),str(fold)]+[str(label)]*128))
            source.write_text('\n'.join(rows))
            args=argparse.Namespace(rounds=1,epochs=1,hidden_size=8,batch_size=8,
                                    learning_rate=.001,threads=1,seed=0,test_fold=9,
                                    validation_fold=8,mask_rate=.2,noise_std=0,
                                    observation_seed=0,device='cpu',ocr_path=str(source),
                                    output_dir=tmp,no_plot=True)
            with contextlib.redirect_stdout(io.StringIO()):folder=run_study(args)
            config=json.loads((folder/'config.json').read_text())
            self.assertEqual(config['train_folds'],list(range(8)))
            self.assertEqual(config['status'],'complete')
            for name in ['bc.pt','dagger_latest.pt','metrics.csv','accuracy.png']:
                self.assertTrue((folder/name).exists())
            checkpoint=torch.load(folder/'dagger_latest.pt',weights_only=True)
            LSTMPolicy(8).load_state_dict(checkpoint['model_state'])


if __name__=='__main__':unittest.main()
