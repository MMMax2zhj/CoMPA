"""CoMPA visibility-constrained prediction aggregation (VPA).

The gate consumes the fusion and branch logits returned by the same model
forward. Missing branches receive exactly zero weight. No label or second
masked view is needed at inference.
"""
import torch
from torch import nn
import torch.nn.functional as F


class VisibilityPredictionAggregator(nn.Module):
    def __init__(self, modalities: int):
        super().__init__()
        self.modalities = modalities
        # Shared branch scorer: confidence, agreement, fusion confidence,
        # fraction of present modalities, and branch probability.
        self.scorer = nn.Linear(5, 1)
        self.modality_bias = nn.Parameter(torch.zeros(modalities))
        nn.init.zeros_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, -3.0)

    def forward(self, fused_logits, branch_logits, mask):
        if branch_logits.shape[1] != self.modalities or mask.shape != branch_logits.shape[:2]:
            raise ValueError('VPA input dimensions do not match modalities')
        if not mask.bool().any(1).all():
            raise ValueError('At least one modality must be present')
        fused = fused_logits.softmax(-1)[:, 1]
        branch = branch_logits.softmax(-1)[:, :, 1]
        count = mask.sum(1, keepdim=True).float() / self.modalities
        x = torch.stack((
            (branch - .5).abs() * 2,
            1 - (branch - fused[:, None]).abs(),
            ((fused - .5).abs() * 2)[:, None].expand_as(branch),
            count.expand_as(branch),
            branch,
        ), -1)
        scores = self.scorer(x).squeeze(-1) + self.modality_bias[None]
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = F.softmax(torch.cat((torch.zeros_like(fused[:, None]), scores), 1), 1)
        probability = weights[:, 0] * fused + (weights[:, 1:] * branch).sum(1)
        return probability, weights


class CoMPA(nn.Module):
    """Inference adapter: one frozen backbone call followed by VPA."""
    def __init__(self, backbone, gate):
        super().__init__()
        self.backbone = backbone
        self.gate = gate

    def forward(self, features, mask):
        out = self.backbone(features, mask)
        probability, weights = self.gate(out['logits'], out['branch_logits'], mask)
        out['backbone_logits'] = out['logits']
        out['backbone_probabilities'] = out['probabilities']
        out['vpa_weights'] = weights
        out['probabilities'] = torch.stack((1 - probability, probability), -1)
        out['logits'] = out['probabilities'].clamp_min(1e-8).log()
        return out


# Historical imports remain loadable for code that evaluates archived V3-RG runs.
ReliabilityGate = VisibilityPredictionAggregator
GatedUnified = CoMPA
