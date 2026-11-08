"""Independent timing fixtures, exact HIM objective, and gradient routing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from himloco_lab.impact.anticipation import (future_event_ids,prediction_valid,SharedBank,
    batched_sinkhorn,clip_models,AnticipationData,future_alarm_records,retime_future_records,
    future_summary,eligible_future_events)


def original_module():
    path=Path(__file__).parents[1]/'source/himloco_lab/himloco_lab/rsl_rl/modules/him_estimator.py'
    spec=importlib.util.spec_from_file_location('original_him_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def checkpoint():
    module=original_module();torch.manual_seed(42)
    estimator=module.HIMEstimator(6,45)
    return estimator,dict(model_state_dict={'estimator.'+k:v for k,v in estimator.state_dict().items()})


def test_first_ordinary_landing_blocks_later_strong():
    age=np.arange(80)[:,None]
    events=dict(native_onset=np.array([240,216,260]),env=np.zeros(3,int),foot=np.zeros(3,int))
    ids=future_event_ids(events,age)
    for t in range(80):
        candidates=[j for j,k in enumerate(events['native_onset']) if 0<k+1-t*8<=32]
        expected=min(candidates,key=lambda j:events['native_onset'][j]) if candidates else -1
        assert ids[t,0,0]==expected,(t,ids[t,0,0],expected)


def test_future_does_not_cross_reset_and_censors_tail():
    age=np.arange(80)[:,None];age[40:]-=40
    events=dict(native_onset=np.array([344]),env=np.array([0]),foot=np.array([0]))
    ids=future_event_ids(events,age)
    assert (ids[:40]==-1).all()
    valid=prediction_valid(age)
    for t in range(80):
        expected=t+6<80 and age[t,0]>=25 and age[t+6,0]==age[t,0]+6
        assert valid[t,0]==expected


def test_him_outputs_loss_and_gradients_match_original():
    ref,ckpt=checkpoint()
    bank=SharedBank([dict(seed=1,group='joint',loss='bce')],ckpt)
    x=torch.randn(32,270);next48=torch.randn(32,48)
    bank.normalize_prototypes()
    with torch.no_grad():ref.proto.weight.copy_(F.normalize(ref.proto.weight,dim=-1))
    vel,z=ref.encode(x)
    bv,bz=bank.original(bank.trunk(x))
    torch.testing.assert_close(bv[0],vel);torch.testing.assert_close(bz[0],z)
    target=F.normalize(ref.target(next48[:,3:48]),dim=-1)
    ss=z@ref.proto.weight.T;st=target@ref.proto.weight.T
    swap=-.5*(original_module().sinkhorn(ss)*F.log_softmax(st/3.,-1)+
               original_module().sinkhorn(st)*F.log_softmax(ss/3.,-1)).mean()
    velocity=F.mse_loss(vel,next48[:,45:48])
    a,b=bank.original_losses(bank.trunk(x),next48)
    torch.testing.assert_close(a[0],velocity);torch.testing.assert_close(b[0],swap)
    (a+b).sum().backward();(velocity+swap).backward()
    torch.testing.assert_close(bank.w1.grad[0].T,ref.encoder[0].weight.grad)
    torch.testing.assert_close(bank.proto.grad[0],ref.proto.weight.grad)


def test_impact_routes_only_into_dedicated_and_joint_trunk():
    _,ckpt=checkpoint();groups=['dedicated','frozen','joint','moving','probe19']
    bank=SharedBank([dict(seed=1,group=g,loss='bce') for g in groups],ckpt)
    raw,_=bank(torch.randn(16,270))
    bank.impact_losses(raw,torch.rand(16,4)*3).sum().backward()
    for i,group in enumerate(groups):
        assert bool(bank.w1.grad[i].abs().sum()>0)==(group in ('dedicated','joint'))
        assert bank.h2.grad[i].abs().sum()>0
    assert bank.wo.grad is None


def test_original_loss_and_model_axis_do_not_leak():
    _,ckpt=checkpoint();groups=['dedicated','frozen','joint','moving']
    bank=SharedBank([dict(seed=1,group=g,loss='mse') for g in groups],ckpt)
    loss,_,_=bank.objective(torch.randn(32,270),torch.rand(32,4),torch.randn(32,48))
    loss[3].backward()
    for p in bank.parameters():
        assert p.grad is not None
        assert not torch.count_nonzero(p.grad[:3])
    assert bank.w1.grad[3].abs().sum()>0


def test_clip_models_is_independent():
    _,ckpt=checkpoint();bank=SharedBank([dict(seed=s,group='joint',loss='bce') for s in (1,2)],ckpt)
    for p in bank.parameters():
        p.grad=torch.zeros_like(p);p.grad[0]=100;p.grad[1]=.0001
    before=bank.w1.grad[1].clone();norm=clip_models(bank)
    assert norm[0]>10 and norm[1]<10
    torch.testing.assert_close(before,bank.w1.grad[1])
    total=sum(p.grad[0].square().sum() for p in bank.parameters()).sqrt()
    torch.testing.assert_close(total,torch.tensor(10.),atol=1e-4,rtol=0)


def test_reset_history_is_zero_filled_not_repeated():
    data=object.__new__(AnticipationData);data.device='cpu'
    data.pre=torch.arange(4*2*45).reshape(4,2,45).float()
    data.age=torch.tensor([[0,0],[1,1],[0,2],[1,3]])
    frames=data.frames(torch.tensor([4,6]))
    torch.testing.assert_close(frames[0,0],data.pre[2,0]);assert not frames[0,1:].any()
    torch.testing.assert_close(frames[1,1],data.pre[2,0]);assert not frames[1,2:].any()


def test_future_alarm_deadlines_and_ordinary_false_actions():
    age=np.tile(np.arange(80)[:,None],(1,3))
    events=dict(native_onset=np.array([283,283,283]),env=np.arange(3),foot=np.zeros(3,int),
                severity=np.array([3000.,1000.,3000.]))
    data=dict(age=age,valid=prediction_valid(age),event_id=future_event_ids(events,age))
    score=np.full((80,3,4),-2.,np.float32)
    score[32,0,0]=2.;score[32,1,0]=2.;score[35,2,0]=2.
    base=future_alarm_records(score,data,events,0.,0.)
    records=retime_future_records(base,20.,0.)
    summary=future_summary(records,0.,events,data,np.zeros(3,int),np.arange(3),draws=20)
    assert summary['tp']==1 and summary['fp']==2 and summary['events']==2
    assert summary['false_ordinary']==1 and summary['false_late_strong']==1
    assert summary['lead_ms_median']==70.
    # A30ms-ahead prediction becomes too late after20ms of link delay.
    score[:]=-2.;score[34,0,0]=2.
    base=future_alarm_records(score,data,events,0.,0.)
    no_delay=future_summary(retime_future_records(base,20.,0.),0.,events,data,np.zeros(3,int),np.arange(3),draws=20)
    delayed=future_summary(retime_future_records(base,20.,20.),0.,events,data,np.zeros(3,int),np.arange(3),
                           delay_ms=20.,draws=20)
    assert no_delay['tp']==1 and delayed['tp']==0 and delayed['false_late_strong']==1


def test_future_labels_do_not_enter_inference():
    _,ckpt=checkpoint();bank=SharedBank([dict(seed=1,group='joint',loss='bce')],ckpt)
    x=torch.randn(16,270)
    _,_,a=bank.objective(x,torch.zeros(16,4),torch.zeros(16,48))
    _,_,b=bank.objective(x,torch.ones(16,4)*100,torch.randn(16,48)*100)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items()
                              if k.startswith('test_') and callable(v))


if __name__=='__main__':unittest.main()
