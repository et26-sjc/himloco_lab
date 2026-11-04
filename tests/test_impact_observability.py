"""Scientific contracts for native contact timing and phase-controlled severity."""
import unittest
import numpy as np
import torch

from himloco_lab.impact.observability import contact_candidates,native_rate,matched_auc,EventData,EventBank,OFFSETS


class ObservabilityTests(unittest.TestCase):
    def test_onset_needs_air_gap_and_persistence(self):
        force=np.zeros((100,1,1));force[20:30]=10;force[35:45]=10;force[70]=10
        onset,_,_=contact_candidates(force)
        np.testing.assert_array_equal(onset,[20])

    def test_native_rate_constant_and_ramp(self):
        np.testing.assert_allclose(native_rate(np.full((100,2,4),70.))[15:],0)
        force=np.broadcast_to((np.arange(100)*.0025*700)[:,None,None],(100,2,4)).copy()
        np.testing.assert_allclose(native_rate(force)[15:],700,atol=.001)

    def test_base_boundary_strictly_precedes_onset_even_exact_boundary(self):
        sample=np.arange(80)
        self.assertTrue(np.all(sample//8*.02 < (sample+1)*.0025))
        self.assertTrue(np.all((sample//8+1)*.02 >= (sample+1)*.0025-1e-12))

    def test_feature_history_cutoff_and_no_future(self):
        data=EventData.__new__(EventData);data.device='cpu'
        data.ev={'base':torch.tensor([30]),'env':torch.tensor([0])}
        data.pre=torch.arange(100).float()[:,None,None].expand(-1,1,45).clone()
        frame=data.frames([0])
        for h,offset in enumerate(OFFSETS):
            np.testing.assert_array_equal(frame[h,0,:,0],30+offset-np.arange(6))
        before=frame[2].clone();data.pre[31:]=10000
        torch.testing.assert_close(data.frames([0])[2],before)

    def test_matching_does_not_use_between_stratum_discrimination(self):
        z=np.tile([0,1],20).astype(bool);group=np.repeat([0,1],20)
        score=group.astype(float)
        row=matched_auc(z,score,group)
        self.assertEqual(row['auc'],.5)
        self.assertEqual(row['coverage_fraction'],1)
        self.assertEqual(matched_auc(z,z.astype(float),group)['auc'],1.)

    def test_event_loss_uses_correct_foot_and_valid_cutoff(self):
        bank=EventBank(2,[(1,'bce',0),(2,'mse',1)],(3,2))
        raw=torch.zeros(2,2,4);raw[1,:,2]=1.
        loss,score=bank.objective(raw,torch.tensor([2,2]),torch.tensor([2000.,10000.]),
                                 torch.tensor([[True,True],[True,False]]))
        self.assertAlmostEqual(loss[0].item(),np.log(2),places=6)
        self.assertEqual(loss[1].item(),0)
        self.assertEqual(score.shape,(2,2))


if __name__=='__main__':unittest.main()
