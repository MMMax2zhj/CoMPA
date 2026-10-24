"""Single conditional prompt-completion path followed by one full MulT.

Labels, retrieval and teachers are deliberately absent from forward().
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from .model import ModelConfig, sanitize
from .backbones import SequenceEncoder, SequenceCompensation, pooled


@dataclass
class UnifiedConfig:
    input_dims: tuple = (768, 512, 768)
    hidden: int = 256
    heads: int = 4
    topics: int = 8
    conditional: bool = True
    backbone: str = 'mplmm'
    compensation: bool = False
    sequence_lengths: tuple = (1, 1, 1)
    layers: int = 2
    memory_layers: int = 3
    encoder_layers: int = 1
    dropout: float = .1
    prompt_length: int = 8

    def __post_init__(self):
        ModelConfig(**asdict(self))  # Shared dimensional/depth validation only.
        if self.backbone != 'mplmm' or self.compensation:
            raise ValueError('Unified model requires one MPLMM path, compensation=False')


class UnifiedPromptModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or UnifiedConfig()
        self.projectors = nn.ModuleList([SequenceEncoder(d, self.cfg) for d in self.cfg.input_dims])
        self.branch_heads = nn.ModuleList([nn.Linear(self.cfg.hidden, 2) for _ in range(3)])
        self.completion = SequenceCompensation(self.cfg)
        # These official convolutions only transform observed modalities, whose
        # tokens we restore below. Do not register permanently unused parameters.
        for name in ('proj_l', 'proj_a', 'proj_v'):
            setattr(self.completion.fusion.core, name, nn.Identity())
        self.source_prompt = nn.Embedding(2, self.cfg.hidden)
        nn.init.normal_(self.source_prompt.weight, std=.02)
        self.stage = 'A'
        self.set_stage('A')

    def backbone_modules(self):
        c = self.completion.fusion.core
        return [self.projectors, self.branch_heads] + [module for name, module in c.named_children()
            if name.startswith('trans_') or name in ('proj1', 'proj2', 'out_layer')]

    def backbone_parameters(self):
        return [p for module in self.backbone_modules() for p in module.parameters()]

    def set_stage(self, stage):
        if stage not in ('A', 'B', 'J', 'eval'):
            raise ValueError('Unified model stages are A/B/J/eval; there is no gain-gate stage')
        self.stage = stage
        self.requires_grad_(stage in ('B', 'J'))
        for module in self.backbone_modules():
            module.requires_grad_(stage in ('A', 'J'))
        for p in self.parameters():
            p.grad = None
        self.train(stage != 'eval')
        return self

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == 'B':
            for module in self.backbone_modules():
                module.eval()
        if mode and self.stage == 'A':
            self.completion.eval()
            for module in self.backbone_modules():
                module.train()
        # fuse() uses functional dropout with core.training, in addition to its
        # child modules' dropout. Keep frozen B fusion in eval and A/J active.
        self.completion.fusion.core.training = mode and self.stage in ('A', 'J')
        return self

    def forward(self, features, visible_mask):
        clean = sanitize(features, visible_mask, self.cfg.input_dims)
        sequences = []
        for k, (x, encoder, length) in enumerate(zip(clean, self.projectors, self.cfg.sequence_lengths)):
            x = x[:, None] if x.ndim == 2 else x
            if x.shape[1] != length:
                raise ValueError(f'Sequence length mismatch for modality {k}')
            z = encoder.sequence(x)
            sequences.append(torch.where(visible_mask[:, k, None, None], z, torch.zeros_like(z)))
        sequences = tuple(sequences)
        projected = torch.stack(pooled(sequences), 1)
        branch_logits = torch.stack([head(projected[:, k]) for k, head in enumerate(self.branch_heads)], 1)
        if self.stage == 'A':
            completed, prompts = sequences, None
            fusion_inputs = sequences
        else:
            generated, prompts = self.completion.prepare(sequences, visible_mask)
            # Official signal prompts/dropout must not overwrite real encoded tokens.
            completed = tuple(torch.where(visible_mask[:, k, None, None], sequences[k], x)
                              for k, x in enumerate(generated))
            fusion_inputs = tuple(x + self.source_prompt((~visible_mask[:, k]).long())[:, None]
                                  for k, x in enumerate(completed))
        logits = self.completion.fusion.fuse(fusion_inputs, prompts)
        representation = torch.cat(pooled(completed), -1)
        reconstructed = tuple(dec(x.mean(1)) for dec, x in zip(self.completion.decoders, completed))
        return {'logits': logits, 'probabilities': logits.softmax(-1),
                'branch_logits': branch_logits, 'projected': projected,
                'sequences': sequences, 'completed': completed,
                'representation': representation, 'reconstructed': reconstructed}

    def save(self, path):
        torch.save({'family': 'unified-prompt-v3', 'config': asdict(self.cfg),
                    'stage': self.stage, 'model': self.state_dict()}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=True)
        if state.get('family') != 'unified-prompt-v3':
            raise ValueError('Expected a unified-prompt-v3 checkpoint')
        model = cls(UnifiedConfig(**state['config'])).to(device)
        model.load_state_dict(state['model'])
        model.set_stage(state['stage'])
        model.eval()
        return model
