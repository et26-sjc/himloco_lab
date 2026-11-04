"""Replay contracts: finite payloads, fixed bytes and sampling-prior correction."""
import importlib.util
from pathlib import Path
import pickle
import unittest

import numpy as np
import torch

spec=importlib.util.spec_from_file_location('continual_pilot',Path(__file__).resolve().parents[1]/'scripts/impact/run_continual_pilot.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Pickle needs the module registered, also when imported outside the CLI.
import sys
sys.modules[spec.name]=module


class ReplayTests(unittest.TestCase):
    def payload(self):
        x=torch.arange(1000,dtype=torch.float32)[:,None].expand(-1,270).clone()
        y=torch.zeros(1000,8,4)
        y[800:950,7]=3000.;y[950:,7]=8000.
        y[:,4]=200000.  # Native-rate target must not overflow a half payload.
        return x,y

    def test_stream_partition_does_not_change_reservoir(self):
        x,y=self.payload()
        one=module.Replay(100,'severity',41);chunks=module.Replay(100,'severity',41)
        one.add_batch(x,y)
        for start in range(0,len(x),37):chunks.add_batch(x[start:start+37],y[start:start+37])
        np.testing.assert_array_equal(np.sort(one.x[:,0]),np.sort(chunks.x[:,0]))
        np.testing.assert_array_equal(np.bincount(one.stratum),[50,25,25])
        np.testing.assert_array_equal(one.count,[800,150,50])
        self.assertTrue(np.isfinite(one.y).all())

    def test_importance_weights_restore_robot_interval_prior(self):
        x,y=self.payload();mem=module.Replay(100,'severity',5);mem.add_batch(x,y)
        xx,yy,w=mem.sample(200000,device='cpu')
        score=yy[:,7].max(1).values
        self.assertAlmostEqual(float(w.mean()),1.,places=2)
        self.assertAlmostEqual(float((w*(score>4000)).mean()),.05,places=2)
        self.assertAlmostEqual(float((w*(score>2000)).mean()),.2,places=2)

    def test_serialized_budget_and_rng_resume(self):
        x,y=self.payload();mem=module.Replay(100,'uniform',6);mem.add_batch(x,y)
        self.assertEqual(mem.bytes(),100*(270*4+32*4+9))
        self.assertLess(mem.serialized_bytes(),mem.bytes()+4096)
        restored=pickle.loads(pickle.dumps(mem,protocol=5))
        for a,b in zip(mem.sample(12,device='cpu'),restored.sample(12,device='cpu')):
            torch.testing.assert_close(a,b)

    def test_unit_weights_match_original_four_foot_bce(self):
        bank=module.ForceBank(270,[(1,'bce',7)])
        x,y=self.payload();x=x[:8]/1000;y=y[:8]
        original=bank.loss_output(bank(x),y)[0][0].item()
        opt=torch.optim.Adam(bank.parameters(),lr=0.)
        actual=module.train_step(bank,opt,x,y,torch.ones(8))
        self.assertAlmostEqual(original,actual,places=6)

    def test_fifo_keeps_last_unique_rows(self):
        x,y=self.payload();mem=module.Replay(100,'fifo',5)
        for start in range(0,len(x),37):mem.add_batch(x[start:start+37],y[start:start+37])
        np.testing.assert_array_equal(mem.x[:,0],np.arange(900,1000))
        self.assertEqual(mem.seen,1000)

    def test_batched_model_gradients_do_not_mix(self):
        banks=[module.ForceBank(270,[(seed,'bce',7)]) for seed in (1,2)]
        batched=module.ForceBank(270,[(seed,'bce',7) for seed in (1,2)])
        x=torch.randn(2,11,270);y=torch.randint(0,2,(2,11,4)).float()
        opt=torch.optim.Adam(batched.parameters(),lr=.0001)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(batched(x)[:,:,:4],y,reduction='none').mean((1,2)).sum()
        loss.backward();opt.step()
        for i,b in enumerate(banks):
            other=torch.optim.Adam(b.parameters(),lr=.0001)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(b(x[i])[0,:,:4],y[i]);loss.backward();other.step()
            for k,v in b.named_parameters():torch.testing.assert_close(v[0],dict(batched.named_parameters())[k][i],atol=1e-6,rtol=1e-5)

    def test_prediction_keeps_all_four_feet(self):
        class Data:
            def batch(self,ids,task):
                y=torch.zeros(len(ids),8,4);y[:,7,3]=3000.
                return torch.zeros(len(ids),270),y
        bank=module.ForceBank(270,[(1,'bce',7)])
        p,y=module.predict(bank,Data(),np.arange(7))
        self.assertEqual(p.shape,(7,4));self.assertEqual(y.shape,(7,4))
        np.testing.assert_array_equal(y.sum(0),[0,0,0,7])


if __name__=='__main__':unittest.main()
