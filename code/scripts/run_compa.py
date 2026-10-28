"""Fit CoMPA VPA on a frozen backbone, then evaluate D_test.

Calibration uses D_val for GC/PF and D_gate for SV/TT. The v3 checkpoints
and all modality encoders/fusion layers remain frozen. No test prediction is
loaded during calibration. Evaluation reads the saved gate without fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from icc.full_models import unified_model
from icc.full_protocol import binary_metrics, file_sha, fixed_masks, load_cache, views
from icc.vpa import VisibilityPredictionAggregator

RUNS = ROOT / 'outputs/runs'
CACHES = {
    'GC': ROOT / 'data/cache/full_protocol_v1/GossipCop.pt',
    'PF': ROOT / 'data/cache/full_protocol_v1/PolitiFact.pt',
    'SV': ROOT / 'data/cache/sequence_protocol_v2/FakeSV.pt',
    'TT': ROOT / 'data/cache/sequence_protocol_v2/FakeTT.pt',
}
SEEDS = (20261011, 20261012, 20261013)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def extract(model, batch, modalities, device, batch_size):
    model.eval()
    result = {}
    tick = time.perf_counter()
    with torch.inference_mode():
        for name, bits in views(modalities).items():
            fusion, branches = [], []
            for start in range(0, len(batch), batch_size):
                part = batch.select(slice(start, start + batch_size)).to(device)
                mask = torch.tensor(bits, dtype=torch.bool, device=device)[None].expand(len(part), -1)
                output = model(part.features, mask)
                fusion.append(output['logits'].cpu())
                branches.append(output['branch_logits'].cpu())
            result[name] = {'fused': torch.cat(fusion), 'branches': torch.cat(branches), 'mask': torch.tensor(bits, dtype=torch.bool)[None].expand(len(batch), -1).clone()}
    return result, time.perf_counter() - tick


def flat_examples(extracted, labels):
    fused = torch.cat([v['fused'] for v in extracted.values()])
    branches = torch.cat([v['branches'] for v in extracted.values()])
    masks = torch.cat([v['mask'] for v in extracted.values()])
    targets = labels.repeat(len(extracted))
    return fused, branches, masks, targets


def fit_vpa(extracted, labels, modalities):
    torch.manual_seed(31337)
    gate = VisibilityPredictionAggregator(len(modalities))
    fused, branches, masks, targets = flat_examples(extracted, labels)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.03, weight_decay=.02)
    frequency = torch.bincount(targets, minlength=2).float()
    sample_weight = .5 * len(targets) / frequency[targets]
    baseline = torch.stack([F.binary_cross_entropy_with_logits(x[:, 1] - x[:, 0], labels.float()) for x in (v['fused'] for v in extracted.values())]).mean().item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        probability, weights = gate(fused, branches, masks)
        loss = F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), targets.float(), weight=sample_weight)
        # Shrink toward the unchanged v3 prediction in small datasets.
        loss = loss + .015 * weights[:, 1:].sum(1).mean()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probability, weights = gate(fused, branches, masks)
        final_loss = F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), targets.float()).item()
    return gate, {'calibration_bce_backbone': baseline, 'calibration_bce_vpa': final_loss,
                  'mean_branch_weight': weights[:, 1:].sum(1).mean().item(),
                  'branch_weight_by_view': {name: float(gate(v['fused'], v['branches'], v['mask'])[1][:, 1:].sum(1).mean().detach()) for name, v in extracted.items()}}


def summarize(extracted, labels, ids, split, modalities, gate):
    names = list(extracted)
    bits = np.asarray(list(views(modalities).values()), dtype=bool)
    values = {}
    for name, item in extracted.items():
        with torch.inference_mode():
            gated, weights = gate(item['fused'], item['branches'], item['mask'])
        values[name] = {'backbone': item['fused'].softmax(-1)[:, 1].numpy(), 'compa': gated.numpy(),
                        'branch_weight': weights[:, 1:].sum(1).numpy()}
    y = labels.numpy()
    full = ''.join(modalities)
    results = {}
    for arm in ('backbone', 'compa'):
        per_view = {name: binary_metrics(y, values[name][arm]) for name in names}
        f = np.asarray([per_view[n]['macro_f1'] for n in names])
        summary = {'F_all': float(f.mean()), 'F_full': per_view[full]['macro_f1'],
                   'F_miss': float(np.mean([per_view[n]['macro_f1'] for n in names if n != full])),
                   'F_worst': float(f.min())}
        table = np.stack([values[n][arm] for n in names], axis=1)
        p1 = {}
        for p in (.1, .2, .3, .4, .5, .6, .7):
            repetitions = []
            for repeat in range(10):
                mask = fixed_masks(ids, len(modalities), split, p, repeat)
                chosen = (mask[:, None, :] == bits[None]).all(-1).argmax(1)
                score = binary_metrics(y, table[np.arange(len(y)), chosen])
                repetitions.append(score['macro_f1'])
            p1[str(p)] = {'mean_F1': float(np.mean(repetitions)), 'std_F1': float(np.std(repetitions, ddof=1)), 'repetitions': repetitions}
        results[arm] = {'P0': summary, 'views': per_view, 'P1': p1}
    results['vpa_weights'] = {n: float(values[n]['branch_weight'].mean()) for n in names}
    return results


def run_one(condition, seed, phase, output, device, batch_size):
    cell = output / f'{condition}_s{seed}'
    current = RUNS / f'B1_{condition}_CoMPA_Backbone_s{seed}' / 'model.pt'
    historical = RUNS / f'B1_{condition}_Unified_v3_s{seed}' / 'model.pt'
    checkpoint = current if current.exists() else historical
    splits, dims, lengths, modalities, _ = load_cache(CACHES[condition])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = unified_model(dims, lengths, saved['model_settings'], 'F111')
    model.load_state_dict(saved['model'], strict=True)
    model.set_stage('eval').to(device)
    if phase == 'calibrate':
        split = 'D_gate' if 'D_gate' in splits else 'D_val'
        if (cell / 'vpa.pt').exists():
            print(json.dumps({'skip_calibrated': cell.name}), flush=True)
            return
        extracted, seconds = extract(model, splits[split], modalities, device, batch_size)
        gate, training = fit_vpa(extracted, splits[split].labels, modalities)
        report = summarize(extracted, splits[split].labels, splits[split].ids, split, modalities, gate)
        cell.mkdir(parents=True, exist_ok=True)
        torch.save({'model': gate.state_dict(), 'modalities': modalities, 'calibration_split': split,
                    'checkpoint_sha256': file_sha(checkpoint), 'cache_sha256': file_sha(CACHES[condition]),
                    'script_sha256': file_sha(__file__)}, cell / 'vpa.pt')
        atomic_json(cell / 'calibration.json', {'condition': condition, 'seed': seed, 'split': split,
                                               'size': len(splits[split]), 'training': training,
                                               'results': report, 'extract_seconds': seconds,
                                               'D_test': 'not_loaded'})
        print(json.dumps({'calibrated': cell.name, 'F_all_backbone': report['backbone']['P0']['F_all'],
                          'F_all_compa': report['compa']['P0']['F_all'], 'seconds': seconds}), flush=True)
    else:
        if (cell / 'test.json').exists():
            print(json.dumps({'skip_evaluated': cell.name}), flush=True)
            return
        frozen = torch.load(cell / 'vpa.pt', map_location='cpu', weights_only=True)
        if frozen['checkpoint_sha256'] != file_sha(checkpoint) or frozen['cache_sha256'] != file_sha(CACHES[condition]) or frozen['script_sha256'] != file_sha(__file__):
            raise RuntimeError('Checkpoint, cache, or implementation changed after calibration')
        gate = VisibilityPredictionAggregator(len(modalities))
        gate.load_state_dict(frozen['model'], strict=True)
        gate.eval()
        extracted, seconds = extract(model, splits['D_test'], modalities, device, batch_size)
        report = summarize(extracted, splits['D_test'].labels, splits['D_test'].ids, 'D_test', modalities, gate)
        # Verify the published v3 model is the same before comparing new output.
        reference = checkpoint.with_name('D_test_predictions.npz')
        baseline_max_abs = None
        if reference.exists():
            with np.load(reference) as original:
                baseline_max_abs = max(float(np.max(np.abs(extracted[n]['fused'].softmax(-1)[:, 1].numpy() - original[n]))) for n in extracted)
            if baseline_max_abs > 1e-4:
                raise RuntimeError(f'Backbone checkpoint reproduction failed: {baseline_max_abs}')
        atomic_json(cell / 'test.json', {'condition': condition, 'seed': seed, 'split': 'D_test',
                                        'size': len(splits['D_test']), 'results': report,
                                        'backbone_prediction_max_abs_error': baseline_max_abs,
                                        'extract_seconds': seconds, 'gate_training_split': frozen['calibration_split'],
                                        'single_backbone_forward_per_view_batch': True})
        print(json.dumps({'evaluated': cell.name, 'F_all_backbone': report['backbone']['P0']['F_all'],
                          'F_all_compa': report['compa']['P0']['F_all'], 'seconds': seconds}), flush=True)


def main():
    global RUNS, CACHES
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('calibrate', 'evaluate'), required=True)
    p.add_argument('--output', default='outputs/compa')
    p.add_argument('--runs-dir', type=Path, default=RUNS,
                   help='Directory containing B1_<condition>_CoMPA_Backbone_s<seed> training runs (historical Unified_v3 names accepted)')
    p.add_argument('--cache-root', type=Path, default=ROOT / 'data/cache',
                   help='Directory containing full_protocol_v1 and sequence_protocol_v2')
    p.add_argument('--conditions', nargs='+', choices=tuple(CACHES), default=tuple(CACHES))
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    RUNS = args.runs_dir.resolve()
    CACHES = {
        'GC': args.cache_root / 'full_protocol_v1/GossipCop.pt',
        'PF': args.cache_root / 'full_protocol_v1/PolitiFact.pt',
        'SV': args.cache_root / 'sequence_protocol_v2/FakeSV.pt',
        'TT': args.cache_root / 'sequence_protocol_v2/FakeTT.pt',
    }
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    output = ROOT / args.output
    for condition in args.conditions:
        for seed in args.seeds:
            run_one(condition, seed, args.phase, output, args.device, args.batch_size)


if __name__ == '__main__':
    main()
