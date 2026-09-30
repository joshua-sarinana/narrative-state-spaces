"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, json, time, warnings, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits
FEATURES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
KEYS = ['participant_id_norm', 'hippo_id_norm']
LOWER = 10
STARTS = 10
SEED = 42
MAX_ITER = 200

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def load_inputs(inp):
    e = pd.read_csv(inp / 'new_events.csv', keep_default_na=False)
    edges = pd.read_csv(inp / 'event_edges.csv', keep_default_na=False)
    X = e[[f'Dim{i}' for i in range(4)]].to_numpy(float)
    lookup = {tuple(r): i for i, r in enumerate(e[KEYS].to_numpy())}
    src = np.array([lookup[r.participant_id_norm, r.source_event] for r in edges.itertuples()])
    dst = np.array([lookup[r.participant_id_norm, r.target_event] for r in edges.itertuples()])
    assert len(e) == 256 and len(src) == 251 and (len(lookup) == 256)
    assert len(set(zip(src, dst))) == 251 and np.bincount(src, minlength=len(e)).max() == 1
    assert np.all(e.participant_id_norm.to_numpy()[src] == e.participant_id_norm.to_numpy()[dst])
    active = e[FEATURES].copy()
    for c in FEATURES:
        s = active[c].fillna('none').astype(str).str.strip()
        active[c] = s.mask(s.str.lower().isin({'', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing'}), 'none')
    profiles, inv, w = np.unique(active.to_numpy(dtype=str), axis=0, return_inverse=True, return_counts=True)
    Y = np.array([X[inv == i].mean(0) for i in range(len(w))])
    assert np.max(np.abs(X - Y[inv])) < 1e-10
    return (e, X, src, dst, Y, w, inv)

def canonical(labels):
    ids = sorted(np.unique(labels), key=lambda j: int(np.flatnonzero(labels == j)[0]))
    mapping = {j: i for i, j in enumerate(ids)}
    return np.array([mapping[j] for j in labels], int)

class Engine:

    def __init__(self, Y, w, k, lower=10):
        self.lower = lower
        self.Y = Y
        self.w = w
        self.k = k
        self.m = len(w)
        self.calls = 0
        self.cache = {}
        m = self.m
        idx = np.arange(m * k)
        rows = np.r_[np.repeat(np.arange(m), k), m + np.tile(np.arange(k), m)]
        self.A = coo_matrix((np.r_[np.ones(m * k), np.repeat(w, k)], (rows, np.r_[idx, idx])), shape=(m + k, m * k)).tocsc()
        self.bounds = Bounds(np.zeros(m * k), np.ones(m * k))
        self.integrality = np.ones(m * k, int)

    def centers(self, labels):
        return np.array([np.average(self.Y[labels == j], axis=0, weights=self.w[labels == j]) for j in range(self.k)])

    def sse(self, labels):
        centers = self.centers(labels)
        return float(np.sum(self.w * np.sum((self.Y - centers[labels]) ** 2, axis=1)))

    def fit(self, centers, upper):
        key = (upper, np.asarray(centers).tobytes())
        if key in self.cache:
            r = dict(self.cache[key])
            r['cache_reused'] = True
            return r
        m = self.m
        k = self.k
        previous = None
        last_ss = np.inf
        history = set()
        t0 = time.monotonic()
        calls0 = self.calls
        cons = LinearConstraint(self.A, np.r_[np.ones(m), np.full(k, self.lower)], np.r_[np.ones(m), np.full(k, upper)])
        max_gap = 0.0
        for iteration in range(1, MAX_ITER + 1):
            cost = self.w[:, None] * cdist(self.Y, centers, 'sqeuclidean')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected')
                fit = milp(c=cost.ravel(), integrality=self.integrality, bounds=self.bounds, constraints=cons, options={'mip_rel_gap': 1e-09, 'time_limit': 30.0, 'threads': 1, 'presolve': True})
            self.calls += 1
            if not fit.success:
                raise RuntimeError(f'fixed-center solver status {fit.status}: {fit.message}')
            z = fit.x.reshape(m, k)
            assert np.max(np.abs(z - np.rint(z))) < 1e-06
            labels = z.argmax(1)
            sizes = np.bincount(labels, weights=self.w, minlength=k)
            assert sizes.min() >= self.lower and sizes.max() <= upper
            assert np.allclose(z.sum(1), 1) and np.isclose(cost[np.arange(m), labels].sum(), fit.fun, rtol=1e-08, atol=1e-08)
            centers = self.centers(labels)
            ss = float(np.sum(self.w * np.sum((self.Y - centers[labels]) ** 2, axis=1)))
            assert ss <= last_ss + 1e-07
            max_gap = max(max_gap, float(fit.mip_gap or 0))
            if previous is not None and np.array_equal(labels, previous):
                break
            if labels.tobytes() in history:
                raise RuntimeError('nontrivial equal-objective assignment cycle')
            history.add(labels.tobytes())
            previous = labels.copy()
            last_ss = ss
        else:
            raise RuntimeError('Lloyd iteration limit reached')
        labels = canonical(labels)
        result = {'labels': labels, 'sse': self.sse(labels), 'iterations': iteration, 'solver_calls': self.calls - calls0, 'seconds': time.monotonic() - t0, 'max_fixed_center_mip_gap': max_gap, 'cache_reused': False}
        self.cache[key] = result
        return dict(result)
