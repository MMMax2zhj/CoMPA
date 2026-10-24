import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))

from icc.full_models import unified_model
from icc.full_protocol import views
from icc.reliability_gate import GatedUnified, ReliabilityGate


class V3ReleaseTests(unittest.TestCase):
    def test_two_and_three_modality_forward_and_hidden_isolation(self):
        torch.set_num_threads(2)
        settings = dict(hidden=8, heads=2, layers=1, memory_layers=1,
                        encoder_layers=1, dropout=0, prompt_length=2, topics=2)
        for dims, lengths, modalities in (
            ((8, 6), (5, 3), ('T', 'I')),
            ((8, 6, 4), (5, 3, 1), ('T', 'V', 'A')),
        ):
            with self.subTest(modalities=modalities):
                torch.manual_seed(20261024)
                model = unified_model(dims, lengths, settings).set_stage('eval')
                gated = GatedUnified(model, ReliabilityGate(len(dims))).eval()
                mask = torch.tensor(list(views(modalities).values()), dtype=torch.bool)
                features = tuple(torch.randn(len(mask), length, dim)
                                 for length, dim in zip(lengths, dims))
                hidden_nan = tuple(torch.where(mask[:, k, None, None], feature,
                                               torch.full_like(feature, float('nan')))
                                   for k, feature in enumerate(features))
                with torch.inference_mode():
                    expected = gated(features, mask)
                    actual = gated(hidden_nan, mask)
                torch.testing.assert_close(actual['probabilities'], expected['probabilities'],
                                           rtol=0, atol=0)
                self.assertEqual(expected['probabilities'].shape, (len(mask), 2))
                self.assertTrue(torch.isfinite(expected['probabilities']).all())


if __name__ == '__main__':
    unittest.main()
