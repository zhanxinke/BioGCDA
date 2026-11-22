"""Portable paths, checkpoint metadata and reproducible CPU execution."""
from pathlib import Path
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

from .config import DATA_FILES

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def single_thread():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Keep the controller alive for the duration of the process.
    return threadpool_limits(limits=1)


def data_hashes(folder):
    return {name: sha(Path(folder) / name) for name in DATA_FILES}


def checkpoint_metadata(folder, dataset, fold, candidate, epoch, seed, inputs):
    folder = Path(folder)
    value = dict(dataset=dataset, fold=fold, candidate=candidate, best_epoch=epoch,
                 seed=seed, prediction_model_count=1, ema_decay=0.0, data_sha256=inputs,
                 files_sha256={name: sha(folder/name) for name in ['model.pt', 'config.json', 'split.npz']})
    write_json(folder/'checkpoint.json', value)
    return value
