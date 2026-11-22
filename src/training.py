"""Raw-checkpoint training for degree-balanced structural-path fusion with a fixed decision threshold, without averaging."""
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .graphs import association_graph
from .model import build_model as build_complete_model
from .graphs import evidence_similarity_graph
from .metrics import metrics


def build_model(args):
    return build_complete_model(vars(args))


def build_feature_graphs(data, args, device):
    kw = dict(k=args.k, temperature=args.temperature, mutual=True, device=device)
    return evidence_similarity_graph(data.circ, **kw), evidence_similarity_graph(data.disease, **kw)



def feature_dropout(features, probability, generator):
    if not 0 <= probability < 1:
        raise ValueError('Feature dropout must be in [0, 1)')
    if probability == 0:
        return features
    mask = torch.rand(features.shape, generator=generator, device=features.device) >= probability
    return features * mask.to(features.dtype) / (1 - probability)

def fit(data, split, seed, args, out):
    if args.ema_decay != 0 or not args.validation_only:
        raise ValueError('Training permits raw weights and validation-only selection')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out)
    train, val, test = split
    np.savez(out / 'split.npz', train=train, validation=val, test=test)
    start = time.monotonic()
    xc, xd = torch.tensor(data.circ), torch.tensor(data.disease)
    pairs = torch.tensor(data.pairs)
    labels = torch.tensor(data.labels, dtype=torch.float32)
    positive_edges = data.pairs[train][data.labels[train] == 1]
    graph = association_graph(positive_edges, len(xc), len(xd), 'cpu')
    edge_rng = np.random.default_rng(seed + 100000)
    feature_rng = torch.Generator().manual_seed(seed + 200000)
    graphs = build_feature_graphs(data, args, 'cpu')
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    best_key, best_state, best_epoch = None, None, 0
    history = []
    for epoch in range(args.epochs):
        warmup = min(1., (epoch + 1) / 5.)
        progress = max(0., epoch - 4.) / max(1, args.epochs - 5)
        lr = warmup * (args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        shuffled = rng.permutation(train)
        # One common, epoch-specific edge mask; no labels outside training enter it.
        if args.edge_dropout > 0:
            retained = positive_edges[edge_rng.random(len(positive_edges)) >= args.edge_dropout]
            training_graph = association_graph(retained, len(xc), len(xd), 'cpu')
        else:
            training_graph = graph
        losses = []
        for offset in range(0, len(train), args.batch_size):
            index = shuffled[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            noisy_c = feature_dropout(xc, args.input_dropout, feature_rng)
            noisy_d = feature_dropout(xd, args.input_dropout, feature_rng)
            logits = model.score(model.encode(noisy_c, noisy_d, graphs, training_graph), pairs[index])
            loss = F.binary_cross_entropy_with_logits(logits, labels[index])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits = model.score(model.encode(xc, xd, graphs, graph), pairs[val])
            probability = torch.sigmoid(logits).numpy()
            val_loss = float(F.binary_cross_entropy_with_logits(logits, labels[val]))
        values = metrics(data.labels[val], probability)
        key = (-val_loss, values['accuracy'], values['ap'])
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), learning_rate=lr,
                            validation_logloss=val_loss, validation_accuracy=values['accuracy'], validation_ap=values['ap']))
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch + 1 - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        encoded = model.encode(xc, xd, graphs, graph)
        logits = model.score(encoded, pairs[val])
        probability = torch.sigmoid(logits).numpy()
    result = metrics(data.labels[val], probability)
    result.update(best_epoch=best_epoch, validation_ap=best_key[2], parameters=sum(p.numel() for p in model.parameters()),
                  validation_accuracy=best_key[1], validation_logloss=-best_key[0],
                  selection_metric='logloss', selection_value=-best_key[0], weights_evaluated='instantaneous',
                  ema_decay=0., seconds=time.monotonic() - start, train_pairs=len(train),
                  validation_pairs=len(val), test_pairs=len(test), evaluation_partition='validation')
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    torch.save(best_state, out / 'model.pt')
    pd.DataFrame(dict(pair_index=val, circRNA=[data.circ_names[c] for c in data.pairs[val, 0]],
                      disease=[data.disease_names[d] for d in data.pairs[val, 1]],
                      label=data.labels[val], score=probability)).to_csv(out / 'validation_predictions.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    return result
