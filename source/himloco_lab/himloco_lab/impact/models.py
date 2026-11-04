"""Independent vectorized supervised models; nothing feeds the Go1 controller.

Each leading-dimension slice owns parameters and Adam state. Losses are summed
over models, never averaged across models or coupled by gradient clipping.
"""
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F


class ModelBank(nn.Module):
    def __init__(self, methods, seeds, hidden=(128, 64)):
        super().__init__()
        self.entries = [(seed, method) for seed in seeds for method in methods]
        self.methods = methods
        m = len(self.entries)
        self.w1 = nn.Parameter(torch.empty(m, 270, hidden[0]))
        self.b1 = nn.Parameter(torch.zeros(m, 1, hidden[0]))
        self.w2 = nn.Parameter(torch.empty(m, hidden[0], hidden[1]))
        self.b2 = nn.Parameter(torch.zeros(m, 1, hidden[1]))
        self.w3 = nn.Parameter(torch.empty(m, hidden[1], 12))
        self.b3 = nn.Parameter(torch.zeros(m, 1, 12))
        self.wlin = nn.Parameter(torch.empty(m, 270, 4))
        self.blin = nn.Parameter(torch.zeros(m, 1, 4))
        # Paired seeds share initial backbone values across representations.
        for i, (seed, _) in enumerate(self.entries):
            generator = torch.Generator().manual_seed(seed)
            for weight in (self.w1, self.w2, self.w3, self.wlin):
                with torch.no_grad():
                    weight[i].uniform_(-1 / math.sqrt(weight.shape[1]),
                                       1 / math.sqrt(weight.shape[1]), generator=generator)
        self.register_buffer('mean', torch.zeros(45))
        self.register_buffer('std', torch.ones(45))
        self.register_buffer('permutations', torch.tensor(list(itertools.permutations(range(1, 6)))))

    def forward(self, x, ids=None):
        # x is shared [batch,270]; variants are made without touching labels.
        z = ((x.reshape(-1, 6, 45) - self.mean) / self.std).clamp(-20, 20)
        inputs = z.reshape(-1, 270).unsqueeze(0).expand(len(self.entries), -1, -1).clone()
        for method in self.methods:
            indices = [i for i, (_, name) in enumerate(self.entries) if name == method]
            if method == 'single_bce':
                inputs[indices, :, 45:] = 0
            elif method == 'shuffled_bce':
                # Stable per-example permutation makes repeated eval identical.
                # Includes all 120 permutations, selected by source index.
                order = self.permutations[ids % 120]
                source = z.gather(1, order[..., None].expand(-1, -1, 45))
                inputs[indices, :, 45:] = source.reshape(-1, 225)
        hidden = F.silu(torch.bmm(inputs, self.w1) + self.b1)
        hidden = F.silu(torch.bmm(hidden, self.w2) + self.b2)
        result = torch.bmm(hidden, self.w3) + self.b3
        linear = torch.bmm(inputs, self.wlin) + self.blin
        linear_ids = [i for i, (_, method) in enumerate(self.entries) if method == 'linear_bce']
        result[linear_ids, :, :4] = linear[linear_ids]
        return result


def outputs_and_losses(raw, severity, entries, thresholds):
    """Return primary raw scores, natural-unit point estimates and per-model loss."""
    scores = torch.empty(raw.shape[0], raw.shape[1], 4, device=raw.device)
    points = torch.full_like(scores, float('nan'))
    losses = torch.empty(raw.shape[0], device=raw.device)
    probabilities = torch.empty(raw.shape[0], raw.shape[1], 4, 3, device=raw.device)
    labels = (severity[..., None] > thresholds).float()
    primary = 1
    for method in dict.fromkeys(name for _, name in entries):
        indices = [i for i, (_, name) in enumerate(entries) if name == method]
        r = raw[indices]
        target_labels = labels.unsqueeze(0).expand(len(indices), -1, -1, -1)
        if method in ('linear_bce', 'bce', 'single_bce', 'shuffled_bce'):
            score = r[..., :4]
            p = score.sigmoid()
            loss = F.binary_cross_entropy_with_logits(score, target_labels[..., primary], reduction='none').mean((1, 2))
            point = torch.full_like(p, float('nan'))
            ps = torch.stack((p, p, p), dim=-1)  # only primary is a prediction
        elif method in ('independent', 'ordinal'):
            logits = r.reshape(len(indices), -1, 4, 3)
            if method == 'ordinal':
                logits = torch.cat((logits[..., :1], logits[..., :1] -
                                    F.softplus(logits[..., 1:]).cumsum(-1)), dim=-1)
            ps = logits.sigmoid()
            score = logits[..., primary]
            point = torch.full_like(score, float('nan'))
            loss = F.binary_cross_entropy_with_logits(logits, target_labels, reduction='none').mean((1, 2, 3))
        elif method in ('mse', 'huber'):
            normalized = r[..., :4] if method == 'mse' else F.softplus(r[..., :4])
            point = normalized * 40.0
            target = (severity / 40.0).unsqueeze(0).expand_as(normalized)
            loss = (F.mse_loss(normalized, target, reduction='none') if method == 'mse' else
                    F.huber_loss(normalized, target, reduction='none')).mean((1, 2))
            # A monotonic score, NOT a claimed event probability. Calibration
            # on disjoint validation episodes is mandatory for point models.
            score = (point - thresholds[primary]) / 40.0
            ps = torch.sigmoid((point[..., None] - thresholds) / 40.0)
        elif method == 'lognormal':
            mu = r[..., :4]
            sigma = F.softplus(r[..., 4:8]) + 0.05
            logtarget = torch.log1p(severity)
            loss = (sigma.log() + 0.5 * ((logtarget - mu) / sigma).square()).mean((1, 2))
            z = (torch.log1p(thresholds) - mu[..., None]) / sigma[..., None]
            ps = 0.5 * torch.erfc(z / math.sqrt(2))
            score = torch.logit(ps[..., primary].clamp(1e-6, 1 - 1e-6))
            point = torch.expm1((mu + 0.5 * sigma.square()).clamp(max=12)).clamp_min(0)
        else:
            raise ValueError(method)
        scores[indices] = score
        points[indices] = point
        losses[indices] = loss
        probabilities[indices] = ps
    return scores, points, losses, probabilities
