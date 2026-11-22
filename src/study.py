"""Six-candidate training, validation-only selection, and held-out evaluation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import json
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .config import BASE, GRID
from .data import load_dataset, make_splits
from .metrics import metrics, report_metrics
from .prediction import load_checkpoint, test_predictions
from .training import fit
from .utils import ROOT, checkpoint_metadata, data_hashes, sha, single_thread, write_json


def candidate_key(row):
    return -row['logloss'], row['accuracy'], row['ap'], -row['candidate']


def train_candidate(job):
    data_dir, output, dataset, fold, candidate = job
    controller = single_thread()
    data = load_dataset(Path(data_dir)/dataset)
    split = list(make_splits(data))[fold]
    config = BASE | GRID[candidate]
    folder = Path(output)/'tuning'/dataset/f'fold{fold}'/f'candidate{candidate}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'config.json', config)
    seed = 2026 + 1000*fold
    outcome = fit(data, split, seed, SimpleNamespace(**config), folder)
    pred = pd.read_csv(folder/'validation_predictions.csv')
    probability = pred.score.to_numpy(dtype=np.float32).astype(np.float64)
    values = metrics(pred.label, probability)
    metadata = checkpoint_metadata(folder, dataset, fold, candidate,
                                   outcome['best_epoch'], seed,
                                   data_hashes(Path(data_dir)/dataset))
    return dict(dataset=dataset, fold=fold, candidate=candidate, seed=seed,
                checkpoint_dir=str(folder.relative_to(output)), best_epoch=outcome['best_epoch'],
                checkpoint_sha256=metadata['files_sha256']['model.pt'],
                **values, logloss=float(log_loss(pred.label, probability, labels=[0,1])))


def train(data_dir, output, datasets, folds, candidates, workers):
    output, data_dir = Path(output), Path(data_dir)
    if output.exists():
        raise ValueError('Training output must be a new directory.')
    inputs = {dataset: data_hashes(data_dir/dataset) for dataset in datasets}
    sources = [ROOT/'bp_main.py', ROOT/'bp_tester.py', *sorted((ROOT/'src').glob('*.py'))]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    output.mkdir(parents=True)
    protocol = dict(model='BalancedPathFusion', datasets=datasets, folds=folds,
                    candidates=candidates, base=BASE, grid=GRID, split_seed=2026,
                    initialization_seed='2026 + 1000 * fold',
                    checkpoint_selection='Lowest validation BCE; ties higher ACC, higher AP, earlier epoch.',
                    candidate_selection='Lowest float64 validation logloss; ties higher ACC, higher AP, earlier candidate.',
                    prediction_model_count=1, prediction_averaging=False, weight_averaging=False,
                    ema_decay=0.0, threshold=0.5, data_sha256=inputs,
                    source_sha256=source_hashes,
                    candidate_fits=len(datasets)*len(folds)*len(candidates))
    write_json(output/'protocol.json', protocol)
    jobs = [(str(data_dir), str(output), d, f, c) for d in datasets for f in folds for c in candidates]
    records = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(train_candidate, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            pd.DataFrame(records).sort_values(['dataset','fold','candidate']).to_csv(
                output/'validation_trials.csv', index=False)
            print(f'Completed {n}/{len(jobs)} candidates; test not evaluated.', flush=True)
    selected = []
    for dataset in datasets:
        for fold in folds:
            group = [r for r in records if r['dataset']==dataset and r['fold']==fold]
            if len(group) != len(candidates):
                raise RuntimeError('Incomplete candidate group.')
            selected.append(max(group, key=candidate_key))
    write_json(output/'selected.json', selected)
    write_json(output/'selection_freeze.json', dict(
        protocol_sha256=sha(output/'protocol.json'),
        trials_sha256=sha(output/'validation_trials.csv'),
        selected_sha256=sha(output/'selected.json'), selected_checkpoints=len(selected)))
    print(f'Frozen {len(selected)} raw checkpoints selected on validation loss.', flush=True)


def evaluate(data_dir, output):
    output, data_dir = Path(output), Path(data_dir)
    if (output/'evaluation').exists():
        raise ValueError('This run already has evaluation output; it will not be overwritten.')
    freeze = json.loads((output/'selection_freeze.json').read_text())
    for name,key in [('protocol.json','protocol_sha256'),('validation_trials.csv','trials_sha256'),
                     ('selected.json','selected_sha256')]:
        if sha(output/name) != freeze[key]:
            raise ValueError(f'Frozen selection artifact changed: {name}')
    protocol = json.loads((output/'protocol.json').read_text())
    for name,h in protocol['source_sha256'].items():
        if sha(ROOT/name) != h:
            raise ValueError(f'Code changed since selection: {name}')
    for dataset,h in protocol['data_sha256'].items():
        if data_hashes(data_dir/dataset) != h:
            raise ValueError(f'Data changed since selection: {dataset}')
    selected = json.loads((output/'selected.json').read_text())
    expected = {(d,f) for d in protocol['datasets'] for f in protocol['folds']}
    if len(selected) != len(expected) or {(r['dataset'],r['fold']) for r in selected} != expected:
        raise ValueError('Incomplete or duplicate selected checkpoint records.')
    records = []
    for row in selected:
        folder = output/row['checkpoint_dir']
        if sha(folder/'model.pt') != row['checkpoint_sha256']:
            raise ValueError('Selected raw weights changed.')
        model,encoded,data,split,metadata = load_checkpoint(data_dir, folder, row['dataset'])
        if metadata['fold'] != row['fold'] or metadata['candidate'] != row['candidate']:
            raise ValueError('Checkpoint metadata disagrees with selection.')
        frame = test_predictions(model,encoded,data,split)
        dest = output/'evaluation'/row['dataset']/f'fold{row["fold"]}'
        dest.mkdir(parents=True, exist_ok=False)
        frame.to_csv(dest/'predictions.csv', index=False)
        values = report_metrics(frame.label,frame.score)
        write_json(dest/'metrics.json', values)
        records.append(dict(dataset=row['dataset'],fold=row['fold'],candidate=row['candidate'],**values))
    frame = pd.DataFrame(records)
    frame.to_csv(output/'fold_metrics.csv', index=False)
    metric_names = ['accuracy','auc','ap','aupr_trapezoid','precision','f1','mcc']
    summary = frame.groupby('dataset')[metric_names].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'metrics_summary.csv')
    print(frame[['dataset','fold','accuracy','auc','ap']].to_string(index=False), flush=True)
