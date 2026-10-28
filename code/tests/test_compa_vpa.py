import sys
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from icc.vpa import CoMPA, VisibilityPredictionAggregator


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, mask):
        self.calls += 1
        logits, branches = features
        return {'logits': logits, 'branch_logits': branches,
                'probabilities': logits.softmax(-1)}


class CoMPAVPATests(unittest.TestCase):
    def test_single_forward_and_missing_branch_invariance(self):
        fused = torch.tensor([[.3, -.1], [.1, .8], [-.5, .2]])
        branches = torch.tensor([[[.2, .7], [-.1, .1]],
                                 [[.6, -.2], [.2, .9]],
                                 [[.1, .8], [.7, -.1]]])
        mask = torch.tensor([[True, False], [True, True], [False, True]])
        gate = VisibilityPredictionAggregator(2)
        with torch.no_grad():
            gate.scorer.weight.fill_(.2)
        backbone = CountingBackbone()
        model = CoMPA(backbone, gate)
        out = model((fused, branches), mask)
        self.assertEqual(backbone.calls, 1)
        self.assertTrue(torch.allclose(out['vpa_weights'].sum(1), torch.ones(3)))
        self.assertEqual(out['vpa_weights'][0, 2].item(), 0)
        self.assertEqual(out['vpa_weights'][2, 1].item(), 0)
        changed = branches.clone()
        changed[~mask] = torch.tensor([[-1000., 1000.]])
        other = model((fused, changed), mask)
        torch.testing.assert_close(other['probabilities'], out['probabilities'], rtol=0, atol=0)
        self.assertEqual(backbone.calls, 2)
        self.assertGreater(out['vpa_weights'][:, 1:].sum(1).std().item(), 0)


if __name__ == '__main__':
    unittest.main()
