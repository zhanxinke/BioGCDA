# BioGCDA

Complete single-model implementation for circRNA–disease association prediction.
Two biological attribute encoders are fused with four query-excluded support
values and two degree-normalized similarity paths in one neural decoder. A
learnable bounded path residual contributes to the same association logit.

This package contains only the complete model, its required data and its selected
raw checkpoints. Each prediction uses one checkpoint; the five folds predict
disjoint test partitions and are not averaged.

## 1. Environment setup

The recorded runtime is Python 3.10.21 with CPU training and inference.

```bash
conda create -n BioGCDA python=3.10 pip -y
conda activate BioGCDA
pip install -r requirements.txt
```

No PyTorch Geometric or pretrained feature-generation package is needed: the
provided features have already been reduced to 64 dimensions.

## 2. Repository structure

```text
BalancedPathFusion/
├── bp_main.py             # Training, candidate selection and test evaluation
├── bp_tester.py           # Single-checkpoint test/custom-pair prediction
├── requirements.txt
├── src/                   # Full model, data, graph, training and inference code
├── data/
│   ├── 1-circAtlas/
│   ├── 2-CircR2Disease/
│   └── 3-circRNADisease/

```

`results/` is generated when a command runs and is excluded from version control.
The included small checkpoint files in `weights/` are intentionally tracked.

## 3. Data and checkpoints

Each dataset directory contains four required files:

| File                             | Format                                                |
| -------------------------------- | ----------------------------------------------------- |
| `circRNA_RNAFM_PCA_64.csv`       | No header; circRNA ID followed by 64 numeric features |
| `disease_features_64d.csv`       | Header; disease ID followed by 64 numeric features    |
| `interaction.csv`                | No header; known-positive circRNA/disease ID pairs    |
| `sample_pairs_easy_far_dual.csv` | No header; benchmark circRNA/disease pairs            |

Labels are assigned by membership in `interaction.csv`. Other supplied pairs are
sampled unknowns, not experimentally confirmed biological negatives. Input rows
are L2-normalized. Supplied PCA features and pair-level splits define a transductive
benchmark; the original pretrained feature models and PCA are not refitted here.

The 15 included checkpoint sets cover three datasets and five folds of the complete
model. Each set contains its raw state dictionary, configuration, exact training/
validation/test pair indices and SHA-256 metadata. Inference checks the dataset,
configuration, split and weight hashes to preserve entity ordering and graph context.
Only positive inner-training pairs enter the association graph; each query edge is
excluded exactly when computing its evidence.

## 4. Retrain the complete model

Run all six configurations, five folds and three datasets, then evaluate the 15
validation-selected checkpoints:

```bash
python bp_main.py --output results/retraining --workers 6
```

This trains **90 complete-model candidates**. Each uses at most 100 epochs and
patience 20, with AdamW, warmup/cosine scheduling and raw checkpoint selection by
validation BCE. Candidate selection prioritizes float64 validation log loss, then
ACC, AP and earlier candidate index. The saved selections are frozen before test
evaluation. Test results do not enter candidate selection.

For one dataset:

```bash
python bp_main.py --datasets 1-circAtlas --output results/circAtlas_training --workers 6
```

A smaller execution check can use one fold and one configuration:

```bash
python bp_main.py --datasets 1-circAtlas --folds 0 --candidates 0 --workers 1 --output results/quick_run
```

The smaller command is a restricted run, not the full five-fold/six-candidate protocol.
To separate training from test evaluation:

```bash
python bp_main.py --phase train --output results/separate_run --workers 6
python bp_main.py --phase evaluate --output results/separate_run
```

Training requires a new output directory and evaluation refuses existing evaluation
output. The run saves candidate checkpoints and validation histories under `tuning/`,
the chosen records in `selected.json`, test predictions under `evaluation/`, and
fold metrics/mean/sample SD as CSV files. A single-fold run has no estimable sample
SD. Frozen source/data checks guard evaluation against changes after selection.

Use a newly trained checkpoint for inference by pointing to its candidate directory:

```bash
python bp_tester.py --dataset 1-circAtlas --checkpoint-dir results/quick_run/tuning/1-circAtlas/fold0/candidate0 --output results/retrained_test.csv
```

All paths default relative to the repository location; `--data-dir` and `--output`
accept explicit paths. Exact numerical repeatability depends on the runtime and
hardware. This release preserves the manuscript's complete-model architecture and
selected weights; no ablation implementations, manuscript files or historical result
bundles are included.
