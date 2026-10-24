"""IMOL section 5.3, equations 9–11: in-batch gold positive/hard negative.

Three-modality representation adaptation; this is not the full IMOL system.
Mining is discrete, while selected cosine similarities retain both-side gradients.
"""
import torch
from torch import nn
import torch.nn.functional as F


class IMOLRACL(nn.Module):
    def __init__(self, negative_k=1, temperature=1.):
        super().__init__()
        if not isinstance(negative_k, int) or isinstance(negative_k, bool) or negative_k < 1 or temperature <= 0:
            raise ValueError('RACL requires positive integer negative_k and positive temperature')
        self.negative_k, self.temperature = negative_k, temperature

    def forward(self, representation, labels, ids, groups, split='D_fit'):
        if split != 'D_fit':
            raise ValueError('RACL optimization is restricted to D_fit')
        n = len(representation)
        if representation.ndim != 2 or labels.shape != (n,) or len(ids) != n or len(groups) != n:
            raise ValueError('RACL expects [batch,dim] and aligned labels/metadata')
        if labels.dtype != torch.long or not ((labels == 0) | (labels == 1)).all():
            raise ValueError('RACL expects binary int64 labels')
        if not all(ids) or not all(groups) or not torch.isfinite(representation).all():
            raise ValueError('RACL requires finite representations and explicit sample/group IDs')
        unit = F.normalize(representation, dim=-1)
        similarity = unit @ unit.T
        eligible = torch.tensor([[i != j and ids[i] != ids[j] and groups[i] != groups[j]
                                 for j in range(n)] for i in range(n)],
                                dtype=torch.bool, device=representation.device).reshape(n, n)
        same = labels[:, None] == labels[None, :]
        positive_mask, negative_mask = eligible & same, eligible & ~same
        valid = positive_mask.any(1) & negative_mask.any(1)
        terms, pairs = [], []
        for i in valid.nonzero(as_tuple=True)[0].tolist():
            # Stable tie rule: first eligible batch index.
            positive = similarity[i].detach().masked_fill(~positive_mask[i], -torch.inf).argmax()
            candidates = negative_mask[i].nonzero(as_tuple=True)[0]
            order = torch.argsort(similarity[i, candidates].detach(), descending=True, stable=True)
            negatives = candidates[order[:self.negative_k]]
            logits = torch.cat([similarity[i, positive].reshape(1), similarity[i, negatives]]) / self.temperature
            terms.append(torch.logsumexp(logits, 0) - logits[0])
            pairs.append({'anchor': i, 'positive': int(positive), 'negatives': negatives.tolist()})
        loss = torch.stack(terms).mean() if terms else representation.sum()*0
        return loss, {'attempted': n, 'used': len(terms), 'skipped': n-len(terms),
                      'no_positive': int((~positive_mask.any(1)).sum()),
                      'no_negative': int((~negative_mask.any(1)).sum()), 'pairs': pairs}
