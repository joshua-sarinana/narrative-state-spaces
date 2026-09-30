"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
GAMMA = 0.9
DIMENSIONS = (4, 10, 13)
SEED = 42
N_PERM = 10000

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write_csv(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)

def write_matrix(matrix, labels, path):
    write_csv(pd.DataFrame(matrix, index=labels, columns=labels), path, True)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def counts_from_sequences(sequences, k, collapse=False):
    counts = np.zeros((k, k), dtype=float)
    for values in sequences.values():
        values = np.asarray(values, dtype=np.int16)
        if collapse:
            values = collapse_runs(values)
        if len(values) > 1:
            np.add.at(counts, (values[:-1], values[1:]), 1.0)
    return counts

def collapse_runs(values):
    values = np.asarray(values)
    return values[np.r_[True, values[1:] != values[:-1]]] if len(values) else values.copy()

def row_normalize(counts):
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)

def successor(probabilities):
    identity = np.eye(len(probabilities))
    a = identity - GAMMA * probabilities
    matrix = np.linalg.solve(a, identity)
    require(np.isfinite(matrix).all() and (matrix >= -1e-12).all(), 'Invalid successor matrix.')
    require(np.linalg.norm(a @ matrix - identity, ord='fro') < 1e-09, 'Successor solve residual too large.')
    return matrix

def distances(values):
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def centroids(events, d, k):
    return events.groupby('state', sort=True)[[f'Dim{i}' for i in range(d)]].mean().reindex(range(1, k + 1)).to_numpy(float)

def bh_fdr(values):
    p = np.asarray(values, dtype=float)
    require(np.isfinite(p).all(), 'Nonfinite p-values in BH family.')
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    result = np.empty_like(p)
    result[order] = q
    return result

def upper_tail(null, observed):
    return (1 + np.count_nonzero(null >= np.asarray(observed)[None, ...], axis=0)) / (len(null) + 1)

def permutation_bank(k, n_perm=N_PERM):
    rng = np.random.default_rng(SEED)
    return np.array([rng.permutation(k) for _ in range(n_perm)], dtype=np.int16)

def correspondence(geometry, predictive, bank):
    k = len(geometry)
    upper = np.triu_indices(k, 1)
    x, y = (geometry[upper], predictive[upper])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return ({'pearson_r': np.nan, 'permutation_p': np.nan, 'status': 'not_estimable_missing_centroid'}, np.full(len(bank), np.nan))
    if np.std(x) == 0 or np.std(y) == 0:
        return ({'pearson_r': np.nan, 'permutation_p': np.nan, 'status': 'not_estimable_constant_distances'}, np.full(len(bank), np.nan))
    observed = float(np.corrcoef(x, y)[0, 1])
    unique, inverse = np.unique(bank, axis=0, return_inverse=True)
    null_unique = np.array([float(np.corrcoef(x, predictive[order][:, order][upper])[0, 1]) for order in unique])
    null = null_unique[inverse]
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)
    return ({'pearson_r': observed, 'permutation_p': float(p), 'null_mean_r': float(null.mean()), 'null_sd_r': float(null.std(ddof=1)), 'status': 'ok'}, null)
