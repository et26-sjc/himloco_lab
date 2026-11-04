"""Scientific contracts: physical-time labels, independent updates, proper event MSE."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from himloco_lab.impact.force_study import ForceBank,ForceData,force_targets

spec=importlib.util.spec_from_file_location('force_train',Path(__file__).resolve().parents[1]/'scripts/impact/train_force_study.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ForceStudyTests(unittest.TestCase):
    def test_fixed_time_ramp(self):
        for dt in (.005,.0025,.00125):
            count=round(.04/dt);dec=round(.02/dt)
            t=(np.arange(count+dec)+1)*dt
            x=np.broadcast_to((500*t)[None,:,None],(2,len(t),4))
            y=force_targets(x[:,:count],x[:,count:],dt)
            np.testing.assert_allclose(y[:,[1,2,6,7]],500,atol=.001)
            np.testing.assert_allclose(y[:,4],500,atol=.001)

    def test_constant_load_is_not_impact(self):
        y=force_targets(np.full((3,16,4),80.),np.full((3,8,4),80.),.0025)
        np.testing.assert_allclose(y[:,[0,3,5]],80)
        np.testing.assert_allclose(y[:,[1,2,4,6,7]],0)

    def test_brier_uses_probability(self):
        bank=ForceBank(2,[(1,'brier',7)],hidden=(3,2))
        y=torch.zeros(4,8,4);y[0,7]=4000
        values=[]
        for p in (.1,.25,.5):
            r=torch.zeros(1,4,40);r[...,:4]=torch.logit(torch.tensor(p))
            values.append(bank.loss_output(r,y)[0].item())
        self.assertLess(values[1],values[0]);self.assertLess(values[1],values[2])

    def test_independent_adam_matches_torch(self):
        bank=ForceBank(2,[(1,'mse',0),(2,'mse',0)],hidden=(3,2))
        configs=[{'lr':.001,'wd':0.},{'lr':.003,'wd':.00001}]
        opt=module.IndependentAdam(bank,configs)
        singles=[]
        for i,c in enumerate(configs):
            p=torch.nn.Parameter(bank.w1[i].detach().clone())
            singles.append((p,torch.optim.Adam([p],lr=c['lr'],weight_decay=c['wd'])))
        for step in range(3):
            for p in bank.parameters():p.grad=torch.full_like(p,.1+step*.03)
            opt.step()
            for i,(p,o) in enumerate(singles):
                p.grad=torch.full_like(p,.1+step*.03);o.step()
                torch.testing.assert_close(bank.w1[i],p,rtol=2e-6,atol=2e-7)

    def test_mixture_is_finite_at_zero_and_tail(self):
        bank=ForceBank(2,[(1,'mixture',7)],hidden=(3,2))
        x=torch.randn(16,2);y=torch.zeros(16,8,4);y[8:,7]=20000
        loss,score,point=bank.loss_output(bank(x),y)
        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(score).all())
        self.assertTrue((point>=0).all());loss.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in bank.parameters()))

    def test_high_frequency_cutoffs_do_not_read_future(self):
        data=ForceData.__new__(ForceData)
        data.device='cpu';data.kind='I1';data.n=1;data.dec=4
        data.hf=torch.arange(160).reshape(40,1,4,1).expand(-1,-1,-1,60).half()
        data.y=torch.zeros(40,1,8,4)
        forecast,_=data.batch([30],'forecast');detection,_=data.batch([30],'detection')
        np.testing.assert_array_equal(forecast.reshape(21,45)[:,0],np.arange(119,98,-1))
        np.testing.assert_array_equal(detection.reshape(21,45)[:,0],np.arange(123,102,-1))


if __name__=='__main__':unittest.main()
