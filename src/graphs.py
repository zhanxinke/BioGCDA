"""Fixed similarity and training-positive association graphs."""
import numpy as np
import torch


def sparse_matrix(rows, cols, weights, shape, device, symmetric=False):
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    # COO coalescing is important for the symmetrized kNN graph.
    a = torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])),
                                torch.tensor(weights), shape, check_invariants=True).coalesce()
    idx, value = a.indices(), a.values()
    if symmetric:
        degree = torch.zeros(shape[0]).scatter_add_(0, idx[0], value).clamp_min(1e-12)
        value = value / torch.sqrt(degree[idx[0]] * degree[idx[1]])
    else:
        degree = torch.zeros(shape[0]).scatter_add_(0, idx[0], value).clamp_min(1e-12)
        value = value / degree[idx[0]]
    return torch.sparse_coo_tensor(idx, value, shape, check_invariants=True).coalesce().to(device)


def association_graph(train_positive, nc, nd, device, excluded=None):
    edges = np.asarray(train_positive, dtype=np.int64).reshape(-1, 2)
    if excluded is not None and len(excluded):
        codes = edges[:, 0] * nd + edges[:, 1]
        query = np.asarray(excluded)
        edges = edges[~np.isin(codes, query[:, 0] * nd + query[:, 1])]
    c, d = edges[:, 0], edges[:, 1]
    cd = sparse_matrix(c, d, np.ones(len(edges)), (nc, nd), device)
    dc = sparse_matrix(d, c, np.ones(len(edges)), (nd, nc), device)
    support_c = torch.tensor(np.bincount(c, minlength=nc) > 0, device=device)
    support_d = torch.tensor(np.bincount(d, minlength=nd) > 0, device=device)
    return cd, dc, support_c, support_d


def evidence_similarity_graph(features, k, device="cpu", temperature=0.15, mutual=True):
    """Positive-cosine kNN, reciprocal or union edges, row stochastic, no self loops."""
    n = len(features)
    if not 1 <= k < n or temperature <= 0:
        raise ValueError("Invalid k or temperature")
    cosine = features @ features.T
    np.fill_diagonal(cosine, -np.inf)
    neighbors = np.argsort(-cosine, axis=1, kind="stable")[:, :k]
    chosen = np.zeros((n, n), dtype=bool)
    chosen[np.arange(n)[:, None], neighbors] = True
    chosen &= cosine > 0
    keep = (chosen & chosen.T) if mutual else (chosen | chosen.T)
    row, col = np.nonzero(keep)
    weights = np.exp((cosine[row, col] - 1) / temperature).astype(np.float32)
    return sparse_matrix(row, col, weights, (n, n), device)

