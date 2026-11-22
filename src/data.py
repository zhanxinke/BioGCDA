"""Load supplied PCA-64 data and reproduce the manuscript pair splits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Dataset:
    circ_names: list
    disease_names: list
    circ: np.ndarray
    disease: np.ndarray
    pairs: np.ndarray
    labels: np.ndarray
    positives: np.ndarray


def read_features(path, header):
    frame = pd.read_csv(path, header=header)
    names = frame.iloc[:, 0].astype(str).str.strip().tolist()
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate entity IDs: {path}')
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.shape[1] != 64 or not np.isfinite(values).all():
        raise ValueError(f'Expected finite 64-dimensional features: {path}')
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return names, values


def load_dataset(path):
    path = Path(path)
    cn, cx = read_features(path / 'circRNA_RNAFM_PCA_64.csv', None)
    dn, dx = read_features(path / 'disease_features_64d.csv', 0)
    cm, dm = {s: i for i, s in enumerate(cn)}, {s: i for i, s in enumerate(dn)}

    def read_pairs(filename):
        frame = pd.read_csv(path / filename, header=None, dtype=str)
        if frame.shape[1] != 2 or frame.isna().any().any():
            raise ValueError(f'Expected two nonempty identifier columns: {filename}')
        rows = [(c.strip(), d.strip()) for c, d in frame.itertuples(index=False, name=None)]
        if len(set(rows)) != len(rows):
            raise ValueError(f'Duplicate pairs: {filename}')
        try:
            return np.array([(cm[c], dm[d]) for c, d in rows], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f'Unknown entity in {filename}: {error}') from error

    positive = read_pairs('interaction.csv')
    positive_codes = set((positive[:, 0] * len(dn) + positive[:, 1]).tolist())
    pairs = read_pairs('sample_pairs_easy_far_dual.csv')
    labels = np.array([int(c * len(dn) + d in positive_codes) for c, d in pairs], dtype=np.int64)
    if labels.sum() != len(positive) or len(np.unique(labels)) != 2:
        raise ValueError('Pair file must include all positives once and sampled unknown pairs.')
    return Dataset(cn, dn, cx, dx, pairs, labels, positive)


def make_splits(data, seed=2026):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed).split(data.pairs, data.labels)
    for fold, (development, test) in enumerate(outer):
        train, val = train_test_split(development, test_size=0.2,
                                     stratify=data.labels[development],
                                     random_state=seed + fold + 100)
        for indices in [train, val, test]:
            if len(np.unique(data.labels[indices])) != 2:
                raise ValueError('A partition has only one class.')
        yield train, val, test


def read_queries(path, data):
    """CSV header: circRNA,disease[,label]; all entities need supplied features."""
    frame = pd.read_csv(path, dtype={'circRNA': str, 'disease': str})
    if not {'circRNA', 'disease'}.issubset(frame.columns):
        raise ValueError('Query CSV requires columns circRNA,disease and optional label.')
    if frame.empty or frame[['circRNA', 'disease']].isna().any().any():
        raise ValueError('Query CSV is empty or has missing identifiers.')
    frame['circRNA'] = frame.circRNA.str.strip()
    frame['disease'] = frame.disease.str.strip()
    if frame.duplicated(['circRNA', 'disease']).any():
        raise ValueError('Query CSV contains duplicate pairs.')
    cm = {s: i for i, s in enumerate(data.circ_names)}
    dm = {s: i for i, s in enumerate(data.disease_names)}
    try:
        pairs = np.array([(cm[c], dm[d]) for c, d in
                          frame[['circRNA', 'disease']].itertuples(index=False, name=None)], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'Query entity has no supplied feature row: {error}') from error
    if 'label' in frame and (frame.label.isna().any() or not frame.label.isin([0, 1]).all()):
        raise ValueError('Optional labels must be 0 or 1.')
    return frame, pairs
