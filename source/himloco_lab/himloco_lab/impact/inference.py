"""Small standalone inference module, independent of Isaac Sim and locomotion.

The owner must supply the causal 270D history for the declared task: completed
interval history for detection, pre-action history for forecast. No contact,
terrain identifier or privileged state is read here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LoadingEstimator(nn.Module):
    __constants__ = ['ordinal']

    def __init__(self, bank, index, calibration):
        super().__init__()
        method = bank.entries[index][1]
        if method not in ('bce','ordinal'):
            raise ValueError('Standalone export currently supports BCE and ordinal')
        self.ordinal = method == 'ordinal'
        self.register_buffer('mean',bank.mean.detach().clone())
        self.register_buffer('std',bank.std.detach().clone())
        self.register_buffer('thresholds_n',torch.tensor([20.,40.,80.] if self.ordinal else [40.]))
        self.register_buffer('calibration_slope',torch.tensor(calibration['slope']))
        self.register_buffer('calibration_intercept',torch.tensor(calibration['intercept']))
        self.layer1=nn.Linear(270,bank.w1.shape[2])
        self.layer2=nn.Linear(bank.w2.shape[1],bank.w2.shape[2])
        self.head=nn.Linear(bank.w3.shape[1],12)
        with torch.no_grad():
            for layer,weight,bias in [(self.layer1,bank.w1,bank.b1),(self.layer2,bank.w2,bank.b2),
                                      (self.head,bank.w3,bank.b3)]:
                layer.weight.copy_(weight[index].T)
                layer.bias.copy_(bias[index,0])

    def forward(self, history: torch.Tensor):
        x=((history.reshape(-1,6,45)-self.mean)/self.std).clamp(-20,20).flatten(1)
        raw=self.head(F.silu(self.layer2(F.silu(self.layer1(x)))))
        if self.ordinal:
            r=raw.reshape(-1,4,3)
            logits=torch.cat((r[...,:1],r[...,:1]-F.softplus(r[...,1:]).cumsum(-1)),dim=-1)
            # The SAME increasing calibration transform for all severities
            # preserves their order. It was fitted only at the primary level.
            probabilities=torch.sigmoid(self.calibration_slope*logits+self.calibration_intercept)
            primary=probabilities[...,1]
        else:
            primary=torch.sigmoid(self.calibration_slope*raw[...,:4]+self.calibration_intercept)
            probabilities=primary.unsqueeze(-1)
        return {'primary_probability':primary,'exceedance_probabilities':probabilities}
