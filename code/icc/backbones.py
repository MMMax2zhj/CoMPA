"""Sequence backbones using the locked MPLMM implementation and G2D fusion.

All sequences are fixed-length, padding-free cache representations. Raw encoders
remain frozen outside this module. T,V,A is the public order; upstream uses T,A,V.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .upstream import mplmm_source, mplmm_config, mplmm_missing_modes
from .model import sanitize, mask_codes


def pooled(features):
    return tuple(x.mean(1) if x.ndim == 3 else x for x in features)


class SequenceEncoder(nn.Module):
    """Trainable modality encoder, also used by independent unimodal teachers."""
    def __init__(self, input_dim, cfg):
        super().__init__()
        d = cfg.hidden
        self.input = nn.Sequential(nn.Linear(input_dim, d), nn.GELU(), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4*d, cfg.dropout,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, cfg.encoder_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)

    def sequence(self, x):
        if x.ndim == 2:
            x = x[:, None]
        z = self.input(x)
        position = torch.arange(z.shape[1], device=z.device, dtype=z.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, z.shape[-1], 2, device=z.device, dtype=z.dtype)
                              * (-math.log(10000.) / z.shape[-1]))
        pe = torch.zeros(z.shape[1], z.shape[-1], device=z.device, dtype=z.dtype)
        pe[:, 0::2] = torch.sin(position*frequency)
        pe[:, 1::2] = torch.cos(position*frequency[:pe[:, 1::2].shape[1]])
        return self.temporal(z + pe[None])

    def forward(self, x):
        return self.sequence(x).mean(1)


class SequenceTeacher(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.projector = SequenceEncoder(input_dim, cfg)
        self.head = nn.Linear(cfg.hidden, 2)

    def forward(self, x):
        return self.head(self.projector(x))


class FullMulT(nn.Module):
    """The six directed crossmodal stacks and three memory stacks from MPLMM.

    With prompts=True complete() and fuse() expose the original PromptModel
    boundary for ICC conditions and reconstruction supervision. With no added
    condition this path is numerically checked against upstream forward().
    """
    def __init__(self, cfg, prompts=False):
        super().__init__()
        lengths = cfg.sequence_lengths
        hp = mplmm_config((cfg.hidden,)*3, cfg.hidden,
                          seq_len=(lengths[0], lengths[2], lengths[1]), heads=cfg.heads)
        hp.layers = cfg.layers
        hp.prompt_length = cfg.prompt_length
        for name in ("attn_dropout", "attn_dropout_a", "attn_dropout_v", "relu_dropout",
                     "res_dropout", "out_dropout", "embed_dropout"):
            setattr(hp, name, cfg.dropout)
        self.core = (mplmm_source().PromptModel if prompts else mplmm_source().MULTModel)(hp)
        if prompts:
            # Upstream initializes both factors of missing_type_prompt @ mp to
            # zero. Without upstream pretraining that path is permanently dead.
            # Seed its type and projection factors; signal prompts keep their
            # official initialization and acquire gradients from completion.
            for name in ('missing_type_prompt', 'm_a', 'm_v', 'm_l'):
                nn.init.normal_(getattr(self.core, name), std=.02)
        # Upstream enforces at least three memory layers; expose the requested
        # depth explicitly, retaining the exact official TransformerEncoder class.
        for m in ("l", "a", "v"):
            old = getattr(self.core, f"trans_{m}_mem")
            memory = type(old)(2*cfg.hidden, cfg.heads, cfg.memory_layers,
                attn_dropout=cfg.dropout, relu_dropout=cfg.dropout,
                res_dropout=cfg.dropout, embed_dropout=cfg.dropout, attn_mask=False)
            setattr(self.core, f"trans_{m}_mem", memory)
        # Upstream's dtype/device sentinel uses uninitialized FloatTensor(1).
        # Its value never participates in arithmetic, but can serialize NaN and
        # make otherwise identical frozen checkpoints fail equality checks.
        for name, buffer in self.core.named_buffers():
            if name.endswith('embed_positions._float_tensor'):
                buffer.zero_()
        self.prompts = prompts

    def complete(self, sequences, mask):
        t, v, a = sequences
        t = F.dropout(t.transpose(1, 2), p=self.core.embed_dropout, training=self.training)
        a, v = a.transpose(1, 2), v.transpose(1, 2)
        modes = mplmm_missing_modes(mask)
        rows = [self.core.get_complete_data(t[i], a[i], v[i], int(modes[i])) for i in range(len(t))]
        completed = [torch.cat([r[k] for r in rows], 0).transpose(1, 2) for k in range(3)]
        self.core.get_proj_matrix()
        prompt = torch.matmul(self.core.missing_type_prompt[None], self.core.mp[modes, None])
        return (completed[0], completed[2], completed[1]), prompt

    def fuse(self, sequences, prompts=None):
        # Exact upstream crossmodal -> memory -> residual MLP path; public T,V,A.
        c = self.core
        values = dict(zip(("l", "v", "a"), (x.transpose(0, 1) for x in sequences)))
        summaries = []
        for target, sources, prompt_index in (("l", ("a", "v"), 0), ("a", ("l", "v"), 1), ("v", ("l", "a"), 2)):
            x = torch.cat([getattr(c, f"trans_{target}_with_{source}")(
                values[target], values[source], values[source]) for source in sources], -1)
            if prompts is not None:
                x = torch.cat([x, prompts[:, prompt_index].transpose(0, 1)], 0)
            x = getattr(c, f"trans_{target}_mem")(x)
            summaries.append(x[-1])
        h = torch.cat(summaries, -1)
        h = h + c.proj2(F.dropout(F.relu(c.proj1(h)), p=c.out_dropout, training=c.training))
        return c.out_layer(h)

    def forward(self, sequences, mask):
        if self.prompts:
            complete, prompts = self.complete(sequences, mask)
            return self.fuse(complete, prompts)
        t, v, a = sequences
        return self.core(t, a, v)


class SequenceCompensation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fusion = FullMulT(cfg, prompts=True)
        self.decoders = nn.ModuleList([nn.Linear(cfg.hidden, d) for d in cfg.input_dims])
        if cfg.conditional:
            self.topic_prototypes = nn.Parameter(torch.randn(cfg.topics, cfg.hidden)*.02)
            self.topic_prompts = nn.Parameter(torch.randn(cfg.topics, cfg.hidden)*.02)
            self.queries = nn.ParameterList([nn.Parameter(torch.randn(n, cfg.hidden)*.02) for n in cfg.sequence_lengths])
            self.missing_prompt = nn.Embedding(8, cfg.hidden)
            self.cross_attention = nn.MultiheadAttention(cfg.hidden, cfg.heads, cfg.dropout, batch_first=True)
            self.condition_norm = nn.LayerNorm(cfg.hidden)

    def initialize_from_observed(self, observed):
        # Copy selected stage-A fusion weights; leave new prompts initialized.
        if isinstance(observed.fusion, FullMulT):
            source = observed.fusion.core.state_dict()
            target = self.fusion.core.state_dict()
            shared = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
            self.fusion.core.load_state_dict(shared, strict=False)

    def prepare(self, sequences, mask):
        generated, prompts = self.fusion.complete(sequences, mask)
        if self.cfg.conditional:
            z = torch.stack(pooled(sequences), 1)
            mean = z.sum(1)/mask.sum(1, keepdim=True)
            topics = (F.normalize(mean, dim=-1) @ F.normalize(self.topic_prototypes, dim=-1).T / .2).softmax(-1)
            entropy = -(topics*topics.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            confidence = (1-entropy/math.log(self.cfg.topics)).clamp(0, 1)
            condition = confidence*(topics @ self.topic_prompts) + self.missing_prompt(mask_codes(mask))
            memory = torch.cat(sequences, 1)
            padding = torch.cat([~mask[:, k, None].expand(-1, x.shape[1]) for k, x in enumerate(sequences)], 1)
            completed = []
            for k, x in enumerate(generated):
                q = self.queries[k][None] + condition[:, None]
                residual, _ = self.cross_attention(q, memory, memory, key_padding_mask=padding, need_weights=False)
                conditioned = self.condition_norm(x + residual + q)
                completed.append(torch.where(mask[:, k, None, None], x, conditioned))
            generated = tuple(completed)
        return generated, prompts

    def forward(self, sequences, mask):
        generated, prompts = self.prepare(sequences, mask)
        logits = self.fusion.fuse(generated, prompts)
        reconstructed = tuple(dec(x.mean(1)) for dec, x in zip(self.decoders, generated))
        return {"logits": logits, "reconstructed": reconstructed, "generated": generated}
