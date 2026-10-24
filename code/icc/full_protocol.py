"""Dimension-aware data, masks and metrics for the 20261007 full experiment."""
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def views(modalities):
    return {''.join(modalities[k] for k in index):tuple(k in index for k in range(len(modalities)))
            for size in range(1,len(modalities)+1) for index in combinations(range(len(modalities)),size)}


def validate_mask(mask, m, n=None):
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != m or (n is not None and len(mask)!=n):
        raise ValueError('Expected bool [batch,number_of_modalities] mask')
    if not mask.any(1).all():
        raise ValueError('All-empty input is outside the protocol')


def sanitize(features,mask,dims):
    validate_mask(mask,len(dims))
    if len(features)!=len(dims):
        raise ValueError('Feature/modality count mismatch')
    clean=[]
    for k,(x,d) in enumerate(zip(features,dims)):
        if x.ndim!=3 or x.shape[0]!=len(mask) or x.shape[-1]!=d or not x.is_floating_point():
            raise ValueError('Invalid sequence shape')
        z=torch.where(mask[:,k,None,None],x,torch.zeros_like(x))
        if not torch.isfinite(z).all():
            raise ValueError('Visible features are nonfinite')
        clean.append(z)
    return tuple(clean)


@dataclass
class Batch:
    features:tuple
    native:torch.Tensor
    labels:torch.Tensor
    ids:list
    groups:list
    events:list
    split:str

    def __len__(self): return len(self.ids)

    def select(self,index):
        if isinstance(index,slice): index=torch.arange(len(self))[index]
        indices=index.tolist()
        return Batch(tuple(x[index.to(x.device)] for x in self.features),self.native[index.to(self.native.device)],
            self.labels[index.to(self.labels.device)],[self.ids[i] for i in indices],
            [self.groups[i] for i in indices],[self.events[i] for i in indices],self.split)

    def to(self,device):
        return Batch(tuple(x.to(device) for x in self.features),self.native.to(device),self.labels.to(device),
                     self.ids,self.groups,self.events,self.split)


def load_cache(path):
    raw=torch.load(path,weights_only=True,map_location='cpu')
    if raw.get('schema') not in ('icc-sequence-v2','icc-sequence-ti-v1') or raw.get('status')!='verified':
        raise ValueError('Expected audited sequence cache')
    if not all(raw.get('verification',{}).get(k) is True for k in ('text_publication_only','numeric_audit','split_audit')):
        raise ValueError('Incomplete audit declarations')
    dims,lengths=tuple(raw['input_dims']),tuple(raw['sequence_lengths'])
    modalities=tuple(raw.get('modalities',('T','V','A')))
    if len(dims)!=len(lengths) or len(dims)!=len(modalities) or len(dims) not in (2,3):
        raise ValueError('Invalid modality specification')
    if raw.get('sequence_policy')!='adaptive_average_no_padding':
        raise ValueError('Unsupported token policy')
    seen={k:{} for k in ('ids','groups','events')}
    result={}
    for name,item in raw['splits'].items():
        if name not in ('D_fit','D_probe','D_gate','D_val','D_test'):
            raise ValueError('Unknown partition')
        b=Batch(split=name,**item)
        validate_mask(b.native,len(dims),len(b))
        if len(b.features)!=len(dims) or any(x.shape!=(len(b),n,d) or not torch.isfinite(x).all() for x,n,d in zip(b.features,lengths,dims)):
            raise ValueError('Invalid features')
        if b.labels.dtype!=torch.long or b.labels.shape!=(len(b),) or not ((b.labels==0)|(b.labels==1)).all():
            raise ValueError('Invalid binary labels')
        for field in seen:
            values=getattr(b,field)
            if len(values)!=len(b) or not all(values) or (field=='ids' and len(set(values))!=len(values)):
                raise ValueError('Invalid metadata')
            for value in values:
                if value in seen[field] and seen[field][value]!=name:
                    raise ValueError(f'Cross-partition {field}: {value}')
                seen[field][value]=name
        result[name]=b
    return result,dims,lengths,modalities,{k:v for k,v in raw.items() if k!='splits'}


def sample_masks(n,m,rng,p=None,rho=None):
    if p is not None and rho is not None: raise ValueError('Specify one mask protocol')
    choices=np.array(list(views(tuple(str(i) for i in range(m))).values()),dtype=bool)
    if p is not None:
        if not 0<=p<1: raise ValueError('p must be in [0,1)')
        masks=rng.random((n,m))>=p
        bad=~masks.any(1)
        while bad.any():
            masks[bad]=rng.random((int(bad.sum()),m))>=p
            bad=~masks.any(1)
        return masks
    if rho is not None:
        if not 0<=rho<=1: raise ValueError('rho must be in [0,1]')
        masks=np.ones((n,m),dtype=bool)
        bad=rng.random(n)<rho
        masks[bad]=choices[rng.integers(0,len(choices)-1,int(bad.sum()))]
        return masks
    return choices[rng.integers(0,len(choices),n)]


def fixed_masks(ids,m,split,p,repeat):
    # ID-keyed, class-independent and shared by every method/seed.
    result=[]
    for identity in ids:
        key=f'ICC-P1-v1|{split}|{p:.1f}|{repeat}|{identity}'
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        result.append(sample_masks(1,m,np.random.default_rng(seed),p=p)[0])
    return np.array(result)


def binary_metrics(labels,probabilities):
    y=np.asarray(labels,dtype=int); p=np.asarray(probabilities,dtype=float)
    if len(y)!=len(p) or not np.isfinite(p).all() or not len(y):
        raise ValueError('Invalid predictions')
    pred=(p>.5).astype(int) # argmax([1-p,p]); ties choose class 0.
    f=[]
    for c in (0,1):
        tp=int(((pred==c)&(y==c)).sum()); fp=int(((pred==c)&(y!=c)).sum()); fn=int(((pred!=c)&(y==c)).sum())
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f.append((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,precision,recall))
    # Tie-aware ROC/AUPRC using all distinct score thresholds.
    order=np.argsort(-p,kind='stable'); ys=y[order]; ps=p[order]
    end=np.r_[np.flatnonzero(np.diff(ps))+1,len(p)]-1
    tp=np.cumsum(ys)[end]; fp=(end+1)-tp
    positives=int(y.sum()); negatives=len(y)-positives
    auroc=float(np.trapezoid(np.r_[0,tp]/positives,np.r_[0,fp]/negatives)) if positives and negatives else None
    ap=float(((tp/np.maximum(tp+fp,1))*np.diff(np.r_[0,tp]/positives)).sum()) if positives else None
    confidence=np.maximum(p,1-p); correct=pred==y; ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        chosen=(confidence>=lo)&((confidence<hi) if hi<1 else (confidence<=hi))
        if chosen.any(): ece+=chosen.mean()*abs(confidence[chosen].mean()-correct[chosen].mean())
    return {'macro_f1':(f[0][0]+f[1][0])/2,'accuracy':float(correct.mean()),
            'fake_f1':f[1][0],'fake_precision':f[1][1],'fake_recall':f[1][2],
            'auroc':auroc,'fake_auprc':ap,'ece_15':float(ece),'brier':float(((p-y)**2).mean())}


@torch.inference_mode()
def predict(model,batch,mask,batch_size=16,device='cuda'):
    was=model.training
    model.eval()
    output=[]
    for start in range(0,len(batch),batch_size):
        part=batch.select(slice(start,start+batch_size)).to(device)
        visible=torch.as_tensor(mask[start:start+len(part)],dtype=torch.bool,device=device)
        out=model(part.features,visible)
        output.extend(out['probabilities'][:,1].cpu().tolist())
    model.train(was)
    return np.asarray(output)


def evaluate(model,batch,modalities,batch_size=16,device='cuda',p=None):
    probabilities={}
    if p is None:
        for name,bits in views(modalities).items():
            mask=np.broadcast_to(bits,(len(batch),len(modalities))).copy()
            probabilities[name]=predict(model,batch,mask,batch_size,device)
        scores={k:binary_metrics(batch.labels.cpu().numpy(),v) for k,v in probabilities.items()}
        f=np.array([v['macro_f1'] for v in scores.values()]); full=''.join(modalities)
        missing=[v['macro_f1'] for k,v in scores.items() if k!=full]
        summary={'F_all':float(f.mean()),'F_miss':float(np.mean(missing)),'F_worst':float(f.min()),
                 'F_full':scores[full]['macro_f1'],'Delta_drop':scores[full]['macro_f1']-float(np.mean(missing))}
        return {'selection_score':summary['F_all'],'summary':summary,'views':scores},probabilities
    # Three/seven view forwards suffice for all ten mixed-population masks.
    per_view=[];view_bits=list(views(modalities).values())
    for bits in view_bits:
        mask=np.broadcast_to(bits,(len(batch),len(modalities))).copy()
        per_view.append(predict(model,batch,mask,batch_size,device))
    table=np.stack(per_view,1)
    repetitions=[]
    for repeat in range(10):
        mask=fixed_masks(batch.ids,len(modalities),batch.split,p,repeat)
        chosen=(mask[:,None,:]==np.asarray(view_bits)[None,:,:]).all(-1).argmax(1)
        probability=table[np.arange(len(batch)),chosen]
        probabilities[str(repeat)]=probability
        repetitions.append({'repeat':repeat,'metrics':binary_metrics(batch.labels.cpu().numpy(),probability),
                            'actual_drop_by_modality':(~mask).mean(0).tolist(),
                            'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
    f=[r['metrics']['macro_f1'] for r in repetitions]
    return {'selection_score':float(np.mean(f)),'mask_std':float(np.std(f,ddof=1)),
            'p':p,'repetitions':repetitions},probabilities
