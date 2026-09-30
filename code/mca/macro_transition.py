"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
DATASET_VERSION = 'chronological_256_v2'
N_EVENTS = 256
N_TRANS = 251
N_PARTICIPANTS = 5
K = 4
PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
KEYS = ['participant_id_norm', 'hippo_id_norm']
PARTICIPANT_KEYS = ['participant_id_norm', 'participant_id']
EVENT_KEYS = ['hippo_id_norm', 'hippo_id']
ROLES = {4: 'primary_macro_geometry', 10: 'participant_robust_geometry_sensitivity', 13: 'full_sample_geometry_sensitivity'}

def req(x, msg):
    if not x:
        raise RuntimeError(f'QA FAIL: {msg}')

def norm(v):
    return '' if pd.isna(v) else str(v).strip().lower()

def find_col(cols, cands, label):
    cols = [str(c).strip() for c in cols]
    for c in cands:
        if c in cols:
            return c
    raise RuntimeError(f'QA FAIL: cannot identify {label}; expected {cands}')

def std_keys(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    p = find_col(out.columns, PARTICIPANT_KEYS, 'participant key')
    e = find_col(out.columns, EVENT_KEYS, 'event key')
    if p != 'participant_id_norm':
        out = out.rename(columns={p: 'participant_id_norm'})
    if e != 'hippo_id_norm':
        out = out.rename(columns={e: 'hippo_id_norm'})
    out['participant_id_norm'] = out['participant_id_norm'].map(norm)
    out['hippo_id_norm'] = out['hippo_id_norm'].map(norm)
    return out

def build_transitions(events, seq):
    keys = ['participant_id_norm', 'hippo_id_norm']
    x = events[keys + ['cluster_k4_d4']].merge(seq, on=keys, validate='one_to_one')
    lookup = {(r.participant_id_norm, r.hippo_id_norm): int(r.cluster_k4_d4) for r in x.itertuples(index=False)}
    rows = []
    for r in x.itertuples(index=False):
        if not r.hippo_next_global:
            continue
        key = (r.participant_id_norm, r.hippo_next_global)
        req(key in lookup, f'unresolved next-event {key}')
        rows.append({'participant_id_norm': r.participant_id_norm, 'source_event': r.hippo_id_norm, 'target_event': r.hippo_next_global, 'source_cluster': int(r.cluster_k4_d4), 'target_cluster': lookup[key]})
    out = pd.DataFrame(rows)
    req(len(out) == N_TRANS, 'authoritative transitions !=251')
    return out

def matrices(trans):
    C = np.zeros((K, K), float)
    np.add.at(C, (trans['source_cluster'].to_numpy(int) - 1, trans['target_cluster'].to_numpy(int) - 1), 1)
    req(int(C.sum()) == N_TRANS, 'cluster count total !=251')
    rs = C.sum(axis=1, keepdims=True)
    req((rs[:, 0] > 0).all(), 'zero-outgoing cluster')
    return (C, C / rs)

def centroid_dist(events, d):
    dims = [f'Dim{i}' for i in range(d)]
    cent = events.groupby('cluster_k4_d4', as_index=False)[dims].mean().sort_values('cluster_k4_d4').reset_index(drop=True)
    req(len(cent) == K, f'D{d} centroid count !=4')
    X = cent[dims].to_numpy(float)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    return (D, cent)

def corr(x, y):
    req(len(x) >= 3 and np.std(x) > 0 and (np.std(y) > 0), 'invalid correlation input')
    return float(np.corrcoef(x, y)[0, 1])

def exact_test(D, C, P):
    mask = C > 0
    np.fill_diagonal(mask, False)
    x = D[mask]
    y = P[mask]
    obs = corr(x, y)
    rho = float(spearmanr(x, y).statistic)
    rows = []
    for order in itertools.permutations(range(K)):
        o = np.asarray(order)
        Dp = D[o][:, o]
        rows.append({'permutation': ''.join((str(i + 1) for i in order)), 'is_identity': order == tuple(range(K)), 'pearson_r': corr(Dp[mask], y), 'spearman_rho': float(spearmanr(Dp[mask], y).statistic)})
    perm = pd.DataFrame(rows)
    req(len(perm) == 24 and perm['is_identity'].sum() == 1, 'exact permutation set invalid')
    vals = perm['pearson_r'].to_numpy(float)
    p2 = float(np.count_nonzero(np.abs(vals) >= abs(obs) - 1e-15) / 24)
    pn = float(np.count_nonzero(vals <= obs + 1e-15) / 24)
    return ({'n_observed_nonzero_directed_nonself_pairs': int(mask.sum()), 'pearson_r': obs, 'spearman_rho': rho, 'exact_two_sided_p': p2, 'exact_negative_tail_p': pn, 'n_exact_label_permutations': 24}, perm, mask)
