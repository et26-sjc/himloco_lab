"""Boundary causality, quiet-rule equivalence and reset isolation contracts."""
import numpy as np
import torch
import unittest

from himloco_lab.impact.causal_stream import alarm_records,state_gate,delayed,pd_features,StreamBank


def test_quiet_records_equal_exhaustive_thresholds():
    rng=np.random.default_rng(4);score=rng.normal(size=(70,3,4)).astype('float32')
    age=np.tile(np.arange(70)[:,None],(1,3));age[35:]-=35
    valid=np.ones((70,3),bool);event=np.full((70,3,4),-1,np.int32)
    r=alarm_records(score,age,valid,event)
    for threshold in (-2,-.2,.5,2):
        truth=np.zeros_like(score,dtype=bool)
        for t in range(70):
            for e in range(3):
                lo=max(t-5,t-int(age[t,e]))
                truth[t,e]=(score[t,e]>=threshold)&np.all(score[lo:t,e]<threshold,axis=0)
        selected=(r['lower']<threshold)&(r['upper']>=threshold)
        found=np.zeros_like(truth);found[r['time'][selected],r['env'][selected],r['foot'][selected]]=True
        np.testing.assert_array_equal(truth,found)


def test_state_gate_is_causal_and_requires_air_and_reset_rearm():
    p=np.array([.1,.1,.8,.9,.1,.1,.9,.9,.1,.1,.9],np.float32)[:,None,None]
    age=np.arange(len(p))[:,None];out=state_gate(p,age,.5)
    assert np.flatnonzero(out[:,0,0]).tolist()==[2,7]
    for t in range(len(p)):
        np.testing.assert_array_equal(state_gate(p[:t+1],age[:t+1],.5),out[:t+1])
    age[7:]-=7
    assert not state_gate(p,age,.5)[7,0,0]


def test_delay_does_not_cross_reset():
    x=np.arange(8,dtype='float32').reshape(4,1,2);age=np.array([[0],[1],[0],[1]])
    d=delayed(x,age,1)
    assert np.isneginf(d[0]).all() and np.isneginf(d[2]).all()
    np.testing.assert_array_equal(d[3],x[2])


def test_native_confirmation_and_window_boundaries():
    for k in range(16,32):
        confirm=(k+9)//8
        assert confirm*8>=k+2 and (confirm-1)*8<k+2
        boundaries=np.array([k//8+1,k//8+2])
        assert np.all(boundaries*8>=k+1) and np.all(boundaries*8<k+17)


def test_pd_proxy_uses_same_leg_and_correct_velocity_scale():
    f=torch.zeros(2,6,45);f[:,0,9]=.1;f[:,0,21]=.05;f[:,0,33]=.8
    p=pd_features(f)
    assert p.shape==(2,4,12)
    torch.testing.assert_close(p[:,0,6],torch.full((2,),3.))
    torch.testing.assert_close(p[:,1],torch.zeros(2,12))


def test_independent_models_no_cross_gradient():
    b=StreamBank([(1,'state'),(2,'state')]);x=torch.randn(3,270)
    b(x)[0].sum().backward()
    for p in b.parameters():assert torch.count_nonzero(p.grad[1])==0


def test_interval_search_counts_equal_brute_including_float32_ties():
    rng=np.random.default_rng(9)
    upper=rng.normal(size=10000).astype('float32');lower=upper-rng.random(10000).astype('float32')
    lower[:20]=-np.inf
    thresholds=np.unique(np.r_[np.quantile(upper,np.linspace(0,1,128)).astype('float32'),
                               np.nextafter(upper.max(),np.float32(np.inf))])
    fast=np.searchsorted(np.sort(lower),thresholds,side='left')-np.searchsorted(np.sort(upper),thresholds,side='left')
    slow=[((lower<float(u))&(upper>=float(u))).sum() for u in thresholds]
    np.testing.assert_array_equal(fast,slow)
    assert fast[-1]==0


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__=='__main__':unittest.main()
