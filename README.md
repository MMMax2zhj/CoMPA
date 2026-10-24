# ICC V3-RG

This repository contains the current ICC model path: a Unified V3 backbone and a reliability gate calibrated with the backbone frozen. One backbone forward returns fusion and modality branch logits; the gate mixes their probabilities using the visible modality mask. It supports text/image (GC, PF) and text/video/audio (SV, TT).

## Contents

- `code/icc/full_models.py`, `unified.py`, `backbones.py`: V3 model and its two/three modality implementations.
- `code/icc/reliability_gate.py`: V3-RG gate and single-forward inference adapter.
- `code/scripts/train_v3.py`: the Unified V3 training path, including teachers and A/B/J stages. The runner retains the original training and validation logic while excluding comparison baselines.
- `code/scripts/run_v3_reliability_gate.py`: fit the gate on the calibration partition, then evaluate the frozen pair.
- `code/configs/*.json`: F111 settings used for the four conditions.
- `code/third_party/`: the MPLMM modules and G2D loss functions used by this model, with their original MIT licenses.

The repository excludes datasets, pretrained encoders, feature caches, trained checkpoints, prediction files, experiment outputs, old model versions, pilot variants, and comparison baseline implementations. Supply audited caches and checkpoints separately. The paper's result values are governed by the source project's `docs/research/EXPERIMENT_RESULTS_REPORT_20261024.md`; no results are recomputed by this package.

## Environment and tests

Python 3.10 or newer, PyTorch and NumPy are required. Install dependencies from `requirements.txt`, then run:

```bash
python -m unittest discover -s code/tests -v
```

The checked in tests cover two and three modality forward passes, hidden input isolation, gate weights, and the one-forward adapter. The training and gate command paths were also smoke tested with synthetic audited caches on CPU.

## Train the V3 backbone

From the repository root, for a supplied audited sequence cache:

```bash
python code/scripts/train_v3.py --cache data/cache/full_protocol_v1/GossipCop.pt --config code/configs/GC.json --output outputs/runs/B1_GC_Unified_v3_s20261011 --seed 20261011 --device cuda
```

Use `PF.json`, `SV.json`, or `TT.json` and the corresponding cache for the other conditions. Training reads `D_fit`, selects checkpoints on `D_val`, and does not evaluate `D_test`. The new source layout changes the implementation hash recorded in run metadata, so this package is not a byte-for-byte recreation of the original experiment archive.

## Calibrate and evaluate V3-RG

The run directory names must follow `B1_<condition>_Unified_v3_s<seed>`. The cache root must contain `full_protocol_v1/GossipCop.pt`, `full_protocol_v1/PolitiFact.pt`, `sequence_protocol_v2/FakeSV.pt`, and `sequence_protocol_v2/FakeTT.pt` for the conditions selected.

```bash
python code/scripts/run_v3_reliability_gate.py --phase calibrate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root data/cache --output outputs/v3_rg --device cuda
python code/scripts/run_v3_reliability_gate.py --phase evaluate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root data/cache --output outputs/v3_rg --device cuda
```

Calibration uses `D_val` for GC/PF and `D_gate` for SV/TT when present. Evaluation verifies the saved gate against the backbone checkpoint and cache hashes. If the V3 run includes archived `D_test_predictions.npz`, it also verifies V3 prediction reproduction.

## Provenance

The release is extracted from the local ICC project as of 2026-10-24. MPLMM and G2D source licenses are retained in `code/third_party/`. The G2D file contains only the two loss functions loaded by this implementation.
