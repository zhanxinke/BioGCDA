"""Inference with one complete model and its original training graph."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import torch

from .data import load_dataset
from .graphs import association_graph
from .model import build_model
from .training import build_feature_graphs
from .utils import data_hashes, sha


def load_checkpoint(data_dir, checkpoint_dir, dataset):
    folder = Path(checkpoint_dir)
    metadata = json.loads((folder/'checkpoint.json').read_text())
    if metadata['dataset'] != dataset or metadata['prediction_model_count'] != 1 or metadata['ema_decay'] != 0:
        raise ValueError('Checkpoint dataset or single-model metadata mismatch.')
    if data_hashes(Path(data_dir)/dataset) != metadata['data_sha256']:
        raise ValueError('Dataset files differ from those used by this checkpoint.')
    for name in ['model.pt', 'config.json', 'split.npz']:
        if sha(folder/name) != metadata['files_sha256'][name]:
            raise ValueError(f'Checkpoint artifact changed: {name}')
    config = json.loads((folder/'config.json').read_text())
    if config['ema_decay'] != 0 or config['dim'] != 64:
        raise ValueError('Expected raw PCA-64 BalancedPathFusion weights.')
    data = load_dataset(Path(data_dir)/dataset)
    with np.load(folder/'split.npz', allow_pickle=False) as stored:
        split = {key: stored[key].copy() for key in ['train', 'validation', 'test']}
    indices = np.concatenate(list(split.values()))
    if any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in split.values()):
        raise ValueError('Saved partition indices must be one-dimensional integers.')
    if not np.array_equal(np.sort(indices), np.arange(len(data.labels))):
        raise ValueError('Saved partitions do not form a disjoint partition of the dataset.')
    graphs = build_feature_graphs(data, SimpleNamespace(**config), 'cpu')
    train = split['train']
    positive = data.pairs[train][data.labels[train] == 1]
    association = association_graph(positive, len(data.circ), len(data.disease), 'cpu')
    model = build_model(config).eval()
    model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
    with torch.no_grad():
        encoded = model.encode(torch.tensor(data.circ), torch.tensor(data.disease), graphs, association)
    return model, encoded, data, split, metadata


def predict_scores(model, encoded, pairs):
    with torch.no_grad():
        return torch.sigmoid(model.score(encoded, torch.tensor(pairs, dtype=torch.long))).numpy()


def test_predictions(model, encoded, data, split):
    indices = split['test']
    scores = predict_scores(model, encoded, data.pairs[indices])
    return pd.DataFrame(dict(pair_index=indices, label=data.labels[indices], score=scores.astype(np.float64),
                             circRNA=[data.circ_names[c] for c in data.pairs[indices, 0]],
                             disease=[data.disease_names[d] for d in data.pairs[indices, 1]]))
