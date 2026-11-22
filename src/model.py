"""Compact neural fusion of biological attributes and symmetrically degree-normalized query-excluded graph evidence."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import mlp


class BalancedPathFusion(nn.Module):
    def __init__(self, dim=64, hidden=64, dropout=.2):
        nn.Module.__init__(self)
        self.pc, self.pd = mlp(dim, hidden, hidden, dropout), mlp(dim, hidden, hidden, dropout)
        self.base = nn.Sequential(nn.Linear(2 * hidden + 6, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                  nn.Linear(hidden // 2, 1))
        self.path_scale = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.base[0].weight[:, -6:].zero_()

    def encode(self, xc, xd, similarity_graphs, association_graphs):
        sc, sd = similarity_graphs
        for graph in (sc, sd):
            index = graph.coalesce().indices()
            if torch.any(index[0] == index[1]):
                raise ValueError('Exact query exclusion requires zero-diagonal similarity graphs')
        c, d = F.normalize(self.pc(xc), dim=-1), F.normalize(self.pd(xd), dim=-1)
        with torch.no_grad():
            a = (association_graphs[0].to_dense() > 0).to(c.dtype)
            degree_c, degree_d = a.sum(1), a.sum(0)
            encoded = dict(a=a, degree_c=degree_c, degree_d=degree_d)
            encoded['circ_path'] = torch.sparse.mm(sc, a / degree_c.clamp_min(1).sqrt()[:, None])
            weighted = a / degree_d.clamp_min(1).sqrt()[None, :]
            encoded['disease_numerator'] = torch.sparse.mm(sd.transpose(0, 1), weighted.T).T
        encoded.update(c=c, d=d)
        return encoded

    def score(self, encoded, pairs):
        ci, di = pairs[:, 0], pairs[:, 1]
        c, d = encoded['c'][ci], encoded['d'][di]
        present = encoded['a'][ci, di]
        nc, nd = encoded['degree_c'][ci] - present, encoded['degree_d'][di] - present
        support = torch.stack([torch.log1p(nc) / math.log1p(len(encoded['d'])),
                               torch.log1p(nd) / math.log1p(len(encoded['c'])),
                               (nc > 0).to(c.dtype), (nd > 0).to(c.dtype)], -1)
        paths = torch.stack([encoded['circ_path'][ci, di] / nd.clamp_min(1).sqrt(),
                                 encoded['disease_numerator'][ci, di] / nc.clamp_min(1).sqrt()], -1)
        paths = torch.log1p(paths.clamp_min(0) * len(encoded['d']))
        score = self.base(torch.cat([c, d, support, paths], -1)).squeeze(-1)
        score = score + torch.tanh(self.path_scale) * paths.mean(-1)
        return score


def build_model(config):
    """Build the complete BP architecture; existing raw states load unchanged."""
    return BalancedPathFusion(config['dim'], config['hidden'], config['dropout'])
