"""Evaluation contracts against pseudoreplication and zero-count certainty."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/impact'))
import evaluate_continual_confirmation as evaluation


class ContinualEvaluationTests(unittest.TestCase):
    def test_ci_uses_five_independent_collections(self):
        result=evaluation.mean_ci([1,2,3,4,5])
        self.assertAlmostEqual(result['mean'],3.)
        np.testing.assert_allclose(result['CI95'],[1.03675684,4.96324316],atol=1e-7)
        self.assertEqual(len(result['per_replication']),5)

    def test_zero_false_alarms_do_not_prove_zero_far(self):
        upper=evaluation.poisson_upper(0,100.)
        self.assertAlmostEqual(upper,-np.log(.05)/100.)
        self.assertGreater(upper,0.)
        self.assertIsNone(evaluation.poisson_upper(0,0.))

    def test_frame_ap_counts_events_on_any_foot(self):
        score=np.array([[[0.,0.,0.,3.]],[[0.,0.,0.,-3.]]])
        severity=np.zeros_like(score);severity[0,0,3]=3000.
        result=evaluation.frame(score,np.ones((2,1),bool),severity)
        self.assertAlmostEqual(result['AP'],1.)
        self.assertAlmostEqual(result['foot_prevalence'],1/8)

    def test_stream_scatter_keeps_time_env_and_foot_order(self):
        class Data:
            t=30;n=2;family=np.array([0,6]);tile=np.array([10,11])
            cpu={'age':np.full((30,2),30),'targets':np.zeros((30,2,8,4))}
            def batch(self,ids,task):return torch.as_tensor(ids,dtype=torch.float32)[:,None],None
        class Model:
            entries=[(1,'bce',7)]
            def __call__(self,x):return x[None]*10+torch.arange(40)[None,None]
        score,valid,severity,tiles=evaluation.stream(Model(),Data(),[0])
        np.testing.assert_array_equal(score[0,:,0],np.arange(30)[:,None]*20+np.arange(4)[None])
        self.assertTrue(valid.all());np.testing.assert_array_equal(tiles,[10])


if __name__=='__main__':unittest.main()
