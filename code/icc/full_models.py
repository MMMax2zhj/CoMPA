"""Full-protocol Unified variants and a genuine two-modality MulT path.

The video default executes the existing reviewed MPLMM implementation. TI owns
two encoders, exactly two directed crossmodal stacks and two memory stacks.
"""
import copy
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import math
import torch
from torch import nn
import torch.nn.functional as F
from .unified import UnifiedConfig, UnifiedPromptModel
from .backbones import SequenceEncoder, pooled
from .upstream import mplmm_source
from .full_protocol import sanitize


@dataclass
class TIConfig:
    input_dims:tuple=(384,768)
    sequence_lengths:tuple=(32,49)
    hidden:int=128
    heads:int=4
    layers:int=2
    memory_layers:int=3
    encoder_layers:int=1
    dropout:float=.1
    prompt_length:int=8
    topics:int=8
    conditional:bool=True

    def __post_init__(self):
        if len(self.input_dims)!=2 or len(self.sequence_lengths)!=2 or min(self.input_dims+self.sequence_lengths)<1:
            raise ValueError('TI requires exactly two positive dimensions/lengths')
        if self.hidden<4 or self.hidden%self.heads or min(self.layers,self.memory_layers,self.encoder_layers,self.prompt_length)<1:
            raise ValueError('Invalid attention dimensions or depth')
        if not 0<=self.dropout<1 or self.topics<2: raise ValueError('Invalid dropout/topics')


class TIMulT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        source=mplmm_source()
        def stack(depth):
            return source.TransformerEncoder(cfg.hidden,cfg.heads,depth,
                attn_dropout=cfg.dropout,relu_dropout=cfg.dropout,res_dropout=cfg.dropout,
                embed_dropout=cfg.dropout,attn_mask=False)
        self.cross=nn.ModuleList([stack(cfg.layers) for _ in range(2)])
        self.memory=nn.ModuleList([stack(cfg.memory_layers) for _ in range(2)])
        self.proj1=nn.Linear(2*cfg.hidden,2*cfg.hidden)
        self.proj2=nn.Linear(2*cfg.hidden,2*cfg.hidden)
        self.head=nn.Linear(2*cfg.hidden,2)
        for name,b in self.named_buffers():
            if name.endswith('embed_positions._float_tensor'): b.zero_()

    def forward(self,sequences,prompts=None):
        values=[x.transpose(0,1) for x in sequences]
        summaries=[]
        for k in range(2):
            x=self.cross[k](values[k],values[1-k],values[1-k])
            if prompts is not None: x=torch.cat([x,prompts[:,k].transpose(0,1)],0)
            summaries.append(self.memory[k](x)[-1])
        h=torch.cat(summaries,-1)
        h=h+self.proj2(F.dropout(F.relu(self.proj1(h)),p=self.cfg.dropout,training=self.training))
        return self.head(h)


class TIPrompts(nn.Module):
    """MPLMM's generation, signal and missing-type prompts, reduced to T/I."""
    def __init__(self,cfg,conditional=True):
        super().__init__()
        self.cfg=cfg; self.conditional=conditional
        h=cfg.hidden; p=cfg.prompt_length
        source=mplmm_source()
        self.generative=nn.Parameter(torch.zeros(2,h,p))
        self.translate=nn.ModuleList([source.MLPLayer(h,h) for _ in range(2)])
        self.to_length=nn.ModuleList([source.MLPLayer(p+cfg.sequence_lengths[1-k],cfg.sequence_lengths[k],True) for k in range(2)])
        self.signal_missing=nn.ParameterList([nn.Parameter(torch.zeros(n,h)) for n in cfg.sequence_lengths])
        self.signal_visible=nn.ParameterList([nn.Parameter(torch.zeros(n,h)) for n in cfg.sequence_lengths])
        self.missing_type=nn.Parameter(torch.randn(2,p,h)*.02)
        self.type_projection=nn.ParameterList([nn.Parameter(torch.randn(n,h)*.02) for n in cfg.sequence_lengths])
        if conditional:
            self.topic_prototypes=nn.Parameter(torch.randn(cfg.topics,h)*.02)
            self.topic_prompts=nn.Parameter(torch.randn(cfg.topics,h)*.02)
            self.queries=nn.ParameterList([nn.Parameter(torch.randn(n,h)*.02) for n in cfg.sequence_lengths])
            self.missing_prompt=nn.Embedding(4,h)
            self.cross_attention=nn.MultiheadAttention(h,cfg.heads,cfg.dropout,batch_first=True)
            self.condition_norm=nn.LayerNorm(h)

    def forward(self,sequences,mask):
        values=[]; signals=[]
        for k,x in enumerate(sequences):
            other=sequences[1-k].transpose(1,2)
            translated=self.translate[k](other)
            prompt=self.generative[k][None].expand(len(mask),-1,-1)
            generated=self.to_length[k](torch.cat([prompt,translated],-1).transpose(1,2))
            signal=torch.where(mask[:,k,None,None],self.signal_visible[k][None],self.signal_missing[k][None])
            values.append(torch.where(mask[:,k,None,None],x,generated)+signal)
            signals.append(signal)
        projection=sum(s.transpose(1,2)@p for s,p in zip(signals,self.type_projection))
        prompts=self.missing_type[None]@projection[:,None]
        if self.conditional:
            z=torch.stack(pooled(sequences),1)
            mean=z.sum(1)/mask.sum(1,keepdim=True)
            topics=(F.normalize(mean,dim=-1)@F.normalize(self.topic_prototypes,dim=-1).T/.2).softmax(-1)
            entropy=-(topics*topics.clamp_min(1e-8).log()).sum(-1,keepdim=True)
            confidence=(1-entropy/math.log(self.cfg.topics)).clamp(0,1)
            codes=(mask.long()*mask.new_tensor([2,1],dtype=torch.long)).sum(1)
            condition=confidence*(topics@self.topic_prompts)+self.missing_prompt(codes)
            memory=torch.cat(sequences,1)
            padding=torch.cat([~mask[:,k,None].expand(-1,x.shape[1]) for k,x in enumerate(sequences)],1)
            for k in range(2):
                q=self.queries[k][None]+condition[:,None]
                residual,_=self.cross_attention(q,memory,memory,key_padding_mask=padding,need_weights=False)
                conditioned=self.condition_norm(values[k]+residual+q)
                values[k]=torch.where(mask[:,k,None,None],values[k],conditioned)
        return tuple(values),prompts


class TIUnified(nn.Module):
    def __init__(self,cfg,completion=True,source=True,fixed=False,pool=False):
        super().__init__()
        self.cfg=cfg; self.use_completion=completion; self.use_source=source; self.fixed=fixed; self.pool=pool
        self.projectors=nn.ModuleList([SequenceEncoder(d,cfg) for d in cfg.input_dims])
        self.branch_heads=nn.ModuleList([nn.Linear(cfg.hidden,2) for _ in range(2)])
        fusioncfg=copy.deepcopy(cfg)
        if pool: fusioncfg.sequence_lengths=(1,1)
        self.fusion=TIMulT(fusioncfg)
        self.prompts=TIPrompts(fusioncfg,conditional=cfg.conditional) if completion and not fixed else None
        self.fixed_tokens=nn.ParameterList([nn.Parameter(torch.randn(n,cfg.hidden)*.02) for n in fusioncfg.sequence_lengths]) if fixed else None
        self.source_prompt=nn.Embedding(2,cfg.hidden)
        nn.init.normal_(self.source_prompt.weight,std=.02)
        self.type_prompt=nn.Embedding(4,cfg.prompt_length*cfg.hidden)
        self.decoders=nn.ModuleList([nn.Linear(cfg.hidden,d) for d in cfg.input_dims]) if completion else None
        self.stage='A'; self.set_stage('A')

    def backbone_modules(self): return [self.projectors,self.branch_heads,self.fusion]
    def backbone_parameters(self): return [p for m in self.backbone_modules() for p in m.parameters()]
    def set_stage(self,stage):
        if stage not in ('A','B','J','eval'): raise ValueError('Invalid stage')
        self.stage=stage
        self.requires_grad_(stage in ('B','J'))
        for module in self.backbone_modules(): module.requires_grad_(stage in ('A','J'))
        for p in self.parameters(): p.grad=None
        return self.train(stage!='eval')
    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='B':
            for module in self.backbone_modules(): module.eval()
        if mode and self.stage=='A':
            if self.prompts is not None: self.prompts.eval()
        return self
    def forward(self,features,mask):
        clean=sanitize(features,mask,self.cfg.input_dims)
        sequences=[]
        for k,(encoder,x,n) in enumerate(zip(self.projectors,clean,self.cfg.sequence_lengths)):
            if x.shape[1]!=n: raise ValueError('Unexpected token length')
            z=encoder.sequence(x)
            if self.pool: z=z.mean(1,keepdim=True)
            sequences.append(torch.where(mask[:,k,None,None],z,torch.zeros_like(z)))
        sequences=tuple(sequences); projected=torch.stack(pooled(sequences),1)
        branches=torch.stack([h(projected[:,k]) for k,h in enumerate(self.branch_heads)],1)
        prompts=None
        if self.stage=='A': completed=sequences; inputs=completed
        else:
            if not self.use_completion: completed=sequences
            elif self.fixed:
                completed=tuple(torch.where(mask[:,k,None,None],z,t[None]) for k,(z,t) in enumerate(zip(sequences,self.fixed_tokens)))
            else:
                generated,prompts=self.prompts(sequences,mask)
                completed=tuple(torch.where(mask[:,k,None,None],z,g) for k,(z,g) in enumerate(zip(sequences,generated)))
            if prompts is None:
                code=(mask.long()*mask.new_tensor([2,1],dtype=torch.long)).sum(1)
                prompts=self.type_prompt(code).reshape(len(mask),1,self.cfg.prompt_length,self.cfg.hidden).expand(-1,2,-1,-1)
            inputs=tuple(x+self.source_prompt((~mask[:,k]).long())[:,None] if self.use_source else x for k,x in enumerate(completed))
        logits=self.fusion(inputs,prompts)
        return {'logits':logits,'probabilities':logits.softmax(-1),'branch_logits':branches,
                'projected':projected,'sequences':sequences,'completed':completed,
                'representation':torch.cat(pooled(completed),-1),
                'reconstructed':tuple(d(x.mean(1)) for d,x in zip(self.decoders,completed)) if self.decoders is not None else ()}


class VideoUnified(UnifiedPromptModel):
    def __init__(self,cfg,completion=True,source=True,fixed=False,pool=False,dual=False):
        raw=copy.deepcopy(cfg); fusion=copy.deepcopy(cfg)
        if pool: fusion.sequence_lengths=(1,1,1)
        super().__init__(fusion)
        self.cfg=raw; self.use_completion=completion; self.use_source=source; self.fixed=fixed; self.pool=pool; self.dual=dual
        self.fixed_tokens=nn.ParameterList([nn.Parameter(torch.randn(n,cfg.hidden)*.02) for n in fusion.sequence_lengths]) if fixed else None
        if not completion: self.completion.decoders=nn.ModuleList()
        if dual:
            from .backbones import FullMulT
            self.observed_fusion=FullMulT(fusion,prompts=False)
        self.set_stage('A')

    def backbone_modules(self):
        modules=super().backbone_modules()
        if hasattr(self,'observed_fusion'): modules.append(self.observed_fusion)
        return modules

    def forward(self,features,mask):
        clean=sanitize(features,mask,self.cfg.input_dims)
        sequences=[]
        for k,(encoder,x,n) in enumerate(zip(self.projectors,clean,self.cfg.sequence_lengths)):
            if x.shape[1]!=n: raise ValueError('Unexpected token length')
            z=encoder.sequence(x)
            if self.pool: z=z.mean(1,keepdim=True)
            sequences.append(torch.where(mask[:,k,None,None],z,torch.zeros_like(z)))
        sequences=tuple(sequences); projected=torch.stack(pooled(sequences),1)
        branches=torch.stack([h(projected[:,k]) for k,h in enumerate(self.branch_heads)],1)
        prompts=None
        if self.stage=='A': completed=sequences; inputs=completed
        else:
            if not self.use_completion: completed=sequences
            elif self.fixed:
                completed=tuple(torch.where(mask[:,k,None,None],z,t[None]) for k,(z,t) in enumerate(zip(sequences,self.fixed_tokens)))
            else:
                generated,prompts=self.completion.prepare(sequences,mask)
                completed=tuple(torch.where(mask[:,k,None,None],z,g) for k,(z,g) in enumerate(zip(sequences,generated)))
            if prompts is None:
                c=self.completion.fusion.core
                from .upstream import mplmm_missing_modes
                modes=mplmm_missing_modes(mask); c.get_proj_matrix()
                prompts=c.missing_type_prompt[None]@c.mp[modes,None]
            inputs=tuple(x+self.source_prompt((~mask[:,k]).long())[:,None] if self.use_source else x for k,x in enumerate(completed))
        logits=self.completion.fusion.fuse(inputs,prompts)
        probability=logits.softmax(-1)
        if self.dual:
            observed=self.observed_fusion(sequences,mask)
            probability=(probability+observed.softmax(-1))/2
            logits=probability.clamp_min(1e-12).log()
        return {'logits':logits,'probabilities':probability,'branch_logits':branches,
                'projected':projected,'sequences':sequences,'completed':completed,
                'representation':torch.cat(pooled(completed),-1),
                'reconstructed':tuple(d(x.mean(1)) for d,x in zip(self.completion.decoders,completed))}


def unified_model(dims,lengths,model_settings=None,variant='F111'):
    # Preserve the exact F111 development path. Controls copy every compatible
    # common tensor from a same-seed F111 instance and leave the RNG at F111's
    # post-construction position, so downstream teachers have identical starts.
    reference=None
    if variant!='F111':
        before=torch.get_rng_state()
        reference=unified_model(dims,lengths,model_settings,'F111')
        after_reference=torch.get_rng_state()
        torch.set_rng_state(before)
    settings=dict(model_settings or {})
    settings.update(input_dims=tuple(dims),sequence_lengths=tuple(lengths))
    options={'completion':True,'source':variant!='S-source','fixed':variant=='S-cond','pool':variant=='S-pool'}
    if variant.startswith('F') and len(variant)==4: options['completion']=variant[2]=='1'
    if len(dims)==3:
        settings.update(backbone='mplmm',compensation=False)
        model=VideoUnified(UnifiedConfig(**settings),dual=variant=='S-dual',**options)
    else:
        settings.pop('backbone',None); settings.pop('compensation',None)
        if variant=='S-dual': raise ValueError('Dual control is defined only for video')
        model=TIUnified(TIConfig(**settings),**options)
    if reference is not None:
        source=reference.state_dict();target=model.state_dict();shared={}
        for name,value in target.items():
            key=name.replace('observed_fusion.','completion.fusion.',1) if name.startswith('observed_fusion.') else name
            if key in source and source[key].shape==value.shape:shared[name]=source[key]
        model.load_state_dict(shared,strict=False)
        torch.set_rng_state(after_reference)
    return model
