# CoMPA

CoMPA (Completion and Modality-aware Prediction Aggregation) is the model described in the current paper. A single backbone forward produces a completion-fused prediction and direct predictions from visible modality branches. The separately fitted visibility-constrained prediction aggregator (VPA) combines them, assigning zero weight to absent branches.

The model supports text/image (GC, PF) and text/video/audio (SV, TT). The code includes only model components and training, calibration, and evaluation interfaces. It contains no datasets, feature caches, pretrained weights, checkpoints, predictions, experiment outputs, or baseline implementations.

## Paper terminology and implementation

| Paper | Code | Historical experiment name |
| --- | --- | --- |
| CoMPA | `icc.vpa.CoMPA` | V3-RG |
| Backbone | `icc.full_models.unified_model` | Unified V3 / V3 |
| VPC: visibility-conditioned prompt completion | `icc.unified`, `icc.full_models` | conditional completion, C |
| MPD: modality-prioritized distillation | `train_compa_backbone.py`, `icc.upstream` | G²D adaptation, D |
| DCR: discriminative contrastive regularization | `icc.racl.IMOLRACL` | RACL, R |
| VPA: visibility-constrained prediction aggregation | `icc.vpa.VisibilityPredictionAggregator` | ReliabilityGate |

The historical names remain in archived experiment records and some internal checkpoint-compatible classes. These aliases do not represent different models. The paper's reported numbers come from the source project's `docs/research/EXPERIMENT_RESULTS_REPORT_20261024.md`; this repository does not reproduce them without the audited data and training artifacts.

## Files

- `code/icc/`: backbone, VPA, protocol helpers, and the upstream loss adapters required by CoMPA.
- `code/scripts/train_compa_backbone.py`: teacher training and A/B/J backbone stages; training uses `D_fit` and validation selection uses `D_val`.
- `code/scripts/run_compa.py`: VPA calibration on `D_val` (GC/PF) or `D_gate` (SV/TT), followed by separate `D_test` evaluation.
- `code/configs/*.json`: F111 configurations for the four conditions.
- `code/third_party/`: only the MPLMM and G²D source modules called by the model, with their licenses.

## Install and verify

Use Python 3.10 or newer and install `requirements.txt`. From the repository root:

```bash
python -m unittest discover -s code/tests -v
```

## Train the backbone

Supply an audited cache separately. For GC:

```bash
python code/scripts/train_compa_backbone.py --cache /path/to/GossipCop.pt --config code/configs/GC.json --output outputs/runs/B1_GC_CoMPA_Backbone_s20261011 --seed 20261011 --device cuda
```

Use `PF.json`, `SV.json`, or `TT.json` with the corresponding audited cache for the other conditions. Training does not read `D_test`. The configuration identifies the method as `CoMPA-Backbone` while retaining the F111 experiment variant.

## Fit VPA and evaluate CoMPA

The `--cache-root` folder is supplied by the user. For the requested condition, it must contain `full_protocol_v1/GossipCop.pt` or `PolitiFact.pt`, or `sequence_protocol_v2/FakeSV.pt` or `FakeTT.pt`.

```bash
python code/scripts/run_compa.py --phase calibrate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/audited/cache --output outputs/compa --device cuda
python code/scripts/run_compa.py --phase evaluate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/audited/cache --output outputs/compa --device cuda
```

The interface reads new `B1_<condition>_CoMPA_Backbone_s<seed>` checkpoints and accepts historical `B1_<condition>_Unified_v3_s<seed>` names when present. Calibration saves `vpa.pt` without loading `D_test`; evaluation verifies checkpoint, cache, and implementation hashes before loading it. Model outputs expose `backbone_probabilities` and `vpa_weights` alongside final `probabilities`.

Changing public names changes implementation hashes and does not reproduce the original archive byte for byte. The code is extracted from the ICC project; no training data or generated research artifacts are shipped here.
