# CoMPA

CoMPA (Completion and Modality-aware Prediction Aggregation) detects fake news with missing modalities. This repository contains the model and its training, calibration, and evaluation interfaces for text/image and text/video/audio inputs.

## Setup

Install Python 3.10+ and the dependencies:

```bash
pip install -r requirements.txt
python -m unittest discover -s code/tests -v
```

## Run

Supply an audited feature cache separately. Example for GossipCop:

```bash
python code/scripts/train_compa_backbone.py --cache /path/to/GossipCop.pt --config code/configs/GC.json --output outputs/runs/B1_GC_CoMPA_Backbone_s20261011 --seed 20261011 --device cuda
python code/scripts/run_compa.py --phase calibrate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/cache --output outputs/compa --device cuda
python code/scripts/run_compa.py --phase evaluate --conditions GC --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/cache --output outputs/compa --device cuda
```

Configs for PF, SV, and TT are in `code/configs/`. The cache root must use the paths expected by `run_compa.py`. Datasets, feature caches, and trained weights are not included.
