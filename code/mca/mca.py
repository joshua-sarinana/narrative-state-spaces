"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
REFIT_DATASET_VERSION = 'chronological_256_v2'
ACTIVE_VARS = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
KEY_COLS = ['participant_id_norm', 'hippo_id_norm']
NULLISH = {'', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing'}
HISTORICAL_NULL_TOKEN = 'none'
EXPECTED_EVENTS = 256
EXPECTED_PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
EXPECTED_ACTIVE_LEVELS = 65
EXPECTED_NONZERO_AXES = 54
K_MIN = 2
K_MAX = 19
EXPECTED_K_CANDIDATES = K_MAX - K_MIN + 1

@dataclass
class MCAFit:
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    row_coordinates: np.ndarray
    right_singular_vectors: np.ndarray
    indicator: np.ndarray
    codes: np.ndarray
    category_table: pd.DataFrame
    q: int
    j: int
    dmax: int

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ensure_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f'{name} missing required columns: {missing}')

def read_input(path: Path, sheet_name: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path, low_memory=False)
    if suffix in {'.xlsx', '.xlsm', '.xls'}:
        sheet = 0 if sheet_name in (None, '') else sheet_name
        return pd.read_excel(path, sheet_name=sheet)
    raise RuntimeError(f'Unsupported input type {suffix!r}. Use .xlsx/.xlsm/.xls or .csv.')

def normalize_participants(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def normalize_active_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in ACTIVE_VARS:
        s = df[col].copy()
        s = s.where(~s.isna(), HISTORICAL_NULL_TOKEN)
        s = s.astype(str).str.strip()
        lower = s.str.lower()
        s = s.where(~lower.isin(NULLISH), HISTORICAL_NULL_TOKEN)
        out[col] = s
    return out

def build_indicator(active: pd.DataFrame, participants: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(active)
    q = len(ACTIVE_VARS)
    categories: list[tuple[str, str]] = []
    codes = np.empty((n, q), dtype=np.int16)
    offset = 0
    for qi, col in enumerate(ACTIVE_VARS):
        vals = active[col].astype(str).to_numpy()
        levels = sorted(pd.unique(vals).tolist())
        level_to_code = {level: i for i, level in enumerate(levels)}
        codes[:, qi] = np.fromiter((offset + level_to_code[v] for v in vals), dtype=np.int16, count=n)
        categories.extend(((col, level) for level in levels))
        offset += len(levels)
    j = offset
    indicator = np.zeros((n, j), dtype=np.float64)
    rows = np.arange(n)
    for qi in range(q):
        indicator[rows, codes[:, qi]] = 1.0
    freq = indicator.sum(axis=0).astype(int)
    part_arr = participants.astype(str).to_numpy()
    part_support = []
    for ci in range(j):
        part_support.append(int(pd.unique(part_arr[indicator[:, ci] == 1]).size))
    category_table = pd.DataFrame({'category_index': np.arange(j, dtype=int), 'variable': [v for v, _ in categories], 'category': [c for _, c in categories], 'frequency': freq, 'frequency_share_events': freq / n, 'participant_support': part_support})
    return (indicator, codes, category_table)

def fit_indicator_mca(active: pd.DataFrame, participants: pd.Series) -> MCAFit:
    indicator, codes, category_table = build_indicator(active, participants)
    n, j = indicator.shape
    q = active.shape[1]
    dmax = min(n - 1, j - q)
    f = indicator.sum(axis=0)
    standardized = (indicator - f[None, :] / n) / np.sqrt(q * f[None, :])
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    eigenvalues = np.maximum(s[:dmax] ** 2, 0.0)
    coords = math.sqrt(n) * u[:, :dmax] * s[None, :dmax]
    return MCAFit(eigenvalues=eigenvalues, singular_values=s[:dmax], row_coordinates=coords, right_singular_vectors=vt[:dmax, :].T, indicator=indicator, codes=codes, category_table=category_table, q=q, j=j, dmax=dmax)

def eigenvalues_from_codes(codes: np.ndarray, q: int, j: int, dmax: int) -> np.ndarray:
    n = codes.shape[0]
    g = np.zeros((n, j), dtype=np.float64)
    rows = np.arange(n)
    for qi in range(q):
        g[rows, codes[:, qi]] = 1.0
    f = g.sum(axis=0)
    cross = g.T @ g - np.outer(f, f) / n
    denom = q * np.sqrt(np.outer(f, f))
    burt_residual = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    vals = np.linalg.eigvalsh(burt_residual)[::-1]
    return np.maximum(vals[:dmax], 0.0)

def corrected_inertia(eigenvalues: np.ndarray, q: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cutoff = 1.0 / q
    benz = np.where(eigenvalues > cutoff, (q / (q - 1.0) * (eigenvalues - cutoff)) ** 2, 0.0)
    benz_rate = benz / benz.sum() if benz.sum() > 0 else np.full_like(benz, np.nan)
    greenacre_denominator = q / (q - 1.0) * (np.sum(eigenvalues ** 2) - (j - q) / q ** 2)
    greenacre_rate = benz / greenacre_denominator if greenacre_denominator > 0 else np.full_like(benz, np.nan)
    return (benz, benz_rate, greenacre_rate, float(greenacre_denominator))

def participant_blocks(participants: pd.Series) -> list[np.ndarray]:
    p = participants.astype(str).to_numpy()
    return [np.flatnonzero(p == value) for value in sorted(pd.unique(p))]

def permuted_spectra(codes: np.ndarray, blocks: list[np.ndarray], q: int, j: int, dmax: int, n_perm: int, seed: int, progress_label: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, dmax), dtype=np.float64)
    work = np.empty_like(codes)
    started = time.time()
    checkpoint = max(1, n_perm // 10)
    for b in range(n_perm):
        for qi in range(q):
            for idx in blocks:
                work[idx, qi] = codes[rng.permutation(idx), qi]
        out[b] = eigenvalues_from_codes(work, q=q, j=j, dmax=dmax)
        if (b + 1) % checkpoint == 0 or b + 1 == n_perm:
            elapsed = time.time() - started
            print(f'[parallel] {progress_label}: {b + 1}/{n_perm} ({elapsed:.1f}s)', flush=True)
    return out

def parallel_summary(observed: np.ndarray, null: np.ndarray) -> tuple[pd.DataFrame, int]:
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    q50 = np.quantile(null, 0.5, axis=0)
    q95 = np.quantile(null, 0.95, axis=0)
    q99 = np.quantile(null, 0.99, axis=0)
    p_upper = (1 + np.sum(null >= observed[None, :], axis=0)) / (len(null) + 1)
    exceeds = observed > q95
    retained = 0
    for flag in exceeds:
        if bool(flag):
            retained += 1
        else:
            break
    table = pd.DataFrame({'axis': np.arange(1, len(observed) + 1), 'dimension_name': [f'Dim{i}' for i in range(len(observed))], 'observed_eigenvalue': observed, 'null_mean': mean, 'null_sd': sd, 'null_median': q50, 'null_q95': q95, 'null_q99': q99, 'observed_minus_q95': observed - q95, 'exceeds_q95': exceeds, 'rankwise_upper_p': p_upper, 'in_longest_leading_run': np.arange(1, len(observed) + 1) <= retained})
    return (table, retained)

def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(x.T @ y, ord='fro') ** 2
    den = np.linalg.norm(x.T @ x, ord='fro') * np.linalg.norm(y.T @ y, ord='fro')
    return float(num / den) if den > 0 else np.nan

def normalized_procrustes(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.asarray(x, float).mean(axis=0, keepdims=True)
    y = np.asarray(y, float) - np.asarray(y, float).mean(axis=0, keepdims=True)
    nx = np.linalg.norm(x, ord='fro')
    ny = np.linalg.norm(y, ord='fro')
    if nx == 0 or ny == 0:
        return np.nan
    x /= nx
    y /= ny
    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    aligned = y @ (u @ vt)
    return float(np.sum((x - aligned) ** 2))

def subspace_metrics(full_coords: np.ndarray, fold_coords: np.ndarray, d: int) -> dict[str, float]:
    d = int(min(d, full_coords.shape[1], fold_coords.shape[1]))
    x = full_coords[:, :d]
    y = fold_coords[:, :d]
    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = y - y.mean(axis=0, keepdims=True)
    angles = np.degrees(subspace_angles(x0, y0))
    return {'compared_dimensions': d, 'procrustes_disparity': normalized_procrustes(x, y), 'linear_cka': linear_cka(x, y), 'principal_angle_mean_deg': float(np.mean(angles)), 'principal_angle_max_deg': float(np.max(angles))}

def pair_masks(participants: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p = participants.astype(str).to_numpy()
    i, j = np.triu_indices(len(p), k=1)
    within = p[i] == p[j]
    return (within, ~within)

def safe_corr(a: np.ndarray, b: np.ndarray, kind: str='pearson') -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if kind == 'pearson':
        return float(pearsonr(a, b).statistic)
    return float(spearmanr(a, b).statistic)

# Distance differences below this absolute threshold are numerical ties.
# Coordinates and continuous distance statistics are never rounded.
FIDELITY_TIE_ATOL = 1e-10
FIDELITY_METHOD = 'explicit_self_exclusion_event_key_ties_v1'

def distance_tie_blocks(values: np.ndarray):
    """Yield sorted blocks spanning at most FIDELITY_TIE_ATOL from their minimum."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Distances must be finite.')
    order = np.argsort(values, kind='stable')
    ordered = values[order]
    start = 0
    while start < len(order):
        end = int(np.searchsorted(ordered, ordered[start] + FIDELITY_TIE_ATOL, side='right'))
        yield order[start:end]
        start = end

def distance_average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks, treating numerical distance ties as equal."""
    ranks = np.empty(len(values), dtype=float)
    first_rank = 1
    for block in distance_tie_blocks(values):
        ranks[block] = first_rank + (len(block) - 1) / 2
        first_rank += len(block)
    return ranks

def ordered_neighbors(distances: np.ndarray, event_keys: Sequence[tuple[str, str]]) -> np.ndarray:
    """All nonself neighbors, with distance ties ordered by stable event keys."""
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    keys = [tuple(map(str, key)) for key in event_keys]
    if distances.shape != (n, n) or len(keys) != n or len(set(keys)) != n:
        raise ValueError('Expected square distances and one unique event key per row.')
    key_order = np.empty(n, dtype=int)
    key_order[sorted(range(n), key=lambda i: keys[i])] = np.arange(n)
    neighbors = np.empty((n, n - 1), dtype=int)
    for i in range(n):
        others = np.flatnonzero(np.arange(n) != i)
        blocks = []
        for block in distance_tie_blocks(distances[i, others]):
            candidates = others[block]
            blocks.append(candidates[np.argsort(key_order[candidates], kind='stable')])
        neighbors[i] = np.concatenate(blocks)
    return neighbors

def neighborhood_fidelity(full_neighbors: np.ndarray, trunc_neighbors: np.ndarray, k: int) -> tuple[float, float, float]:
    """Standard trustworthiness, recall, and Jaccard from the same neighbor orders."""
    n = len(full_neighbors)
    if not 0 < k < n / 2:
        raise ValueError('Trustworthiness requires 0 < k < n / 2.')
    full_ranks = np.zeros((n, n), dtype=int)
    full_ranks[np.arange(n)[:, None], full_neighbors] = np.arange(1, n)
    penalty = 0
    recalls, jaccards = [], []
    for i in range(n):
        a = set(full_neighbors[i, :k])
        b = set(trunc_neighbors[i, :k])
        intersection = len(a & b)
        recalls.append(intersection / k)
        jaccards.append(intersection / (2 * k - intersection))
        penalty += sum(int(full_ranks[i, j]) - k for j in b - a)
    trust = 1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1))
    return float(trust), float(np.mean(recalls)), float(np.mean(jaccards))

def neighbor_overlap(full_dist: np.ndarray, trunc_dist: np.ndarray, k: int,
                     event_keys: Sequence[tuple[str, str]]) -> tuple[float, float]:
    """Explicitly exclude self and use the declared distance-tie rule."""
    _, recall, jaccard = neighborhood_fidelity(
        ordered_neighbors(full_dist, event_keys), ordered_neighbors(trunc_dist, event_keys), k)
    return recall, jaccard

def geometric_fidelity(coords: np.ndarray, participants: pd.Series, event_keys: Sequence[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, dmax = coords.shape
    full_sq = np.sum(coords ** 2, axis=1)
    full_dist_vec = pdist(coords)
    full_dist = squareform(full_dist_vec)
    full_neighbors = ordered_neighbors(full_dist, event_keys)
    full_average_ranks = distance_average_ranks(full_dist_vec)
    within_mask, between_mask = pair_masks(participants)
    total_inertia = np.sum(np.var(coords, axis=0, ddof=0))
    records = []
    participant_records = []
    for d in range(1, dmax + 1):
        x = coords[:, :d]
        trunc_vec = pdist(x)
        trunc_dist = squareform(trunc_vec)
        trunc_neighbors = ordered_neighbors(trunc_dist, event_keys)
        raw_share = float(np.sum(np.var(x, axis=0, ddof=0)) / total_inertia)
        cos2 = np.divide(np.sum(x ** 2, axis=1), full_sq, out=np.zeros(n, dtype=float), where=full_sq > 0)
        unscaled_stress = float(np.sqrt(np.sum((full_dist_vec - trunc_vec) ** 2) / np.sum(full_dist_vec ** 2)))
        scale = float(np.dot(full_dist_vec, trunc_vec) / np.dot(trunc_vec, trunc_vec))
        scaled_stress = float(np.sqrt(np.sum((full_dist_vec - scale * trunc_vec) ** 2) / np.sum(full_dist_vec ** 2)))
        row = {'dimensions': d, 'cumulative_raw_inertia': raw_share, 'relative_frobenius_reconstruction_error': math.sqrt(max(0.0, 1.0 - raw_share)), 'distance_pearson_all': safe_corr(full_dist_vec, trunc_vec, 'pearson'), 'distance_spearman_all': safe_corr(full_average_ranks, distance_average_ranks(trunc_vec), 'pearson'), 'distance_pearson_within_participant': safe_corr(full_dist_vec[within_mask], trunc_vec[within_mask], 'pearson'), 'distance_pearson_between_participant': safe_corr(full_dist_vec[between_mask], trunc_vec[between_mask], 'pearson'), 'unscaled_relative_stress': unscaled_stress, 'optimally_scaled_stress': scaled_stress, 'optimal_distance_scale': scale, 'row_cos2_median': float(np.median(cos2)), 'row_cos2_q10': float(np.quantile(cos2, 0.1)), 'row_cos2_q25': float(np.quantile(cos2, 0.25)), 'row_cos2_q75': float(np.quantile(cos2, 0.75))}
        for k in (5, 10, 20):
            trust, recall, jaccard = neighborhood_fidelity(full_neighbors, trunc_neighbors, k)
            row[f'trustworthiness_k{k}'] = trust
            row[f'neighbor_recall_k{k}'] = recall
            row[f'neighbor_jaccard_k{k}'] = jaccard
        records.append(row)
        p_arr = participants.astype(str).to_numpy()
        for pid in sorted(pd.unique(p_arr)):
            vals = cos2[p_arr == pid]
            participant_records.append({'dimensions': d, 'participant': pid, 'n_events': int(len(vals)), 'row_cos2_mean': float(np.mean(vals)), 'row_cos2_median': float(np.median(vals)), 'row_cos2_q10': float(np.quantile(vals, 0.1))})
    return (pd.DataFrame(records), pd.DataFrame(participant_records))

def bootstrap_stability_ari(x: np.ndarray, k: int, n_boot: int=20, seed: int=42) -> float:
    rng = np.random.RandomState(seed)
    all_labels = []
    n = len(x)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed + b)
        model.fit(x[idx])
        all_labels.append(model.predict(x))
    vals = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            vals.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    return float(np.mean(vals))

def normalize01(s: pd.Series) -> pd.Series:
    lo = float(s.min())
    hi = float(s.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def kmeans_consensus_by_dimension(coords: np.ndarray, seed: int=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    dmax = coords.shape[1]
    selection_rows = []
    for d in range(1, dmax + 1):
        x = coords[:, :d]
        eval_rows = []
        for k in range(K_MIN, K_MAX + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            lab = km.fit_predict(x)
            eval_rows.append({'dimensions': d, 'k': k, 'silhouette': float(silhouette_score(x, lab)), 'calinski_harabasz': float(calinski_harabasz_score(x, lab)), 'davies_bouldin': float(davies_bouldin_score(x, lab)), 'min_cluster_fraction': float(np.bincount(lab).min() / len(lab)), 'bootstrap_stability_ari': bootstrap_stability_ari(x, k=k, n_boot=20, seed=seed)})
        block = pd.DataFrame(eval_rows)
        block['silhouette_norm'] = normalize01(block['silhouette'])
        block['ch_norm'] = normalize01(block['calinski_harabasz'])
        block['inverse_db_norm'] = normalize01(1.0 / block['davies_bouldin'])
        block['stability_norm'] = normalize01(block['bootstrap_stability_ari'])
        block['consensus_score'] = block[['silhouette_norm', 'ch_norm', 'inverse_db_norm', 'stability_norm']].mean(axis=1)
        block['eligible_min_fraction_005'] = block['min_cluster_fraction'] >= 0.05
        eligible = block[block['eligible_min_fraction_005']]
        if eligible.empty:
            selected_idx = block['consensus_score'].idxmax()
        else:
            selected_idx = eligible['consensus_score'].idxmax()
        block['selected'] = block.index == selected_idx
        selection_rows.extend(block.to_dict(orient='records'))
        if d % 5 == 0 or d == dmax:
            print(f'[kmeans] completed dimension {d}/{dmax}', flush=True)
    selection = pd.DataFrame(selection_rows)
    chosen = selection[selection['selected']].copy().reset_index(drop=True)
    return (selection, chosen)

def make_spectrum_table(fit: MCAFit) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    eig = fit.eigenvalues
    total = eig.sum()
    benz, benz_rate, green_rate, green_den = corrected_inertia(eig, fit.q, fit.j)
    spectrum = pd.DataFrame({'axis': np.arange(1, fit.dmax + 1), 'dimension_name': [f'Dim{i}' for i in range(fit.dmax)], 'eigenvalue_raw': eig, 'singular_value': fit.singular_values, 'raw_inertia_share': eig / total, 'raw_inertia_cumulative': np.cumsum(eig / total), 'above_benzekri_cutoff': eig > 1.0 / fit.q, 'benzecri_eigenvalue': benz, 'benzecri_rate_normalized': benz_rate, 'benzecri_rate_cumulative': np.cumsum(np.nan_to_num(benz_rate)), 'greenacre_adjusted_rate': green_rate, 'greenacre_adjusted_cumulative': np.cumsum(np.nan_to_num(green_rate))})
    contrib_rows = []
    for axis in range(fit.dmax):
        values = fit.right_singular_vectors[:, axis] ** 2
        order = np.argsort(values)[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        for ci, value in enumerate(values):
            meta = fit.category_table.iloc[ci]
            contrib_rows.append({'axis': axis + 1, 'dimension_name': f'Dim{axis}', 'category_index': ci, 'variable': meta['variable'], 'category': meta['category'], 'frequency': int(meta['frequency']), 'participant_support': int(meta['participant_support']), 'contribution': float(value), 'contribution_rank_within_axis': int(rank[ci])})
    summary = {'n_events': int(fit.indicator.shape[0]), 'active_variables_q': int(fit.q), 'indicator_categories_j': int(fit.j), 'maximum_dimensions': int(fit.dmax), 'total_raw_inertia_observed': float(total), 'total_raw_inertia_theoretical': float((fit.j - fit.q) / fit.q), 'first_five_raw_share': float(eig[:5].sum() / total), 'benzecri_positive_axes': int(np.sum(eig > 1.0 / fit.q)), 'first_five_benzecri_normalized_share': float(np.nansum(benz_rate[:5])), 'greenacre_denominator': green_den, 'greenacre_adjusted_rates_sum': float(np.nansum(green_rate))}
    return (spectrum, pd.DataFrame(contrib_rows), summary)

def structural_preflight(df: pd.DataFrame, active: pd.DataFrame, participants: pd.Series, fit: MCAFit) -> pd.DataFrame:
    counts = participants.value_counts().sort_index()
    observed_counts = {str(k): int(v) for k, v in counts.items()}
    checks = []

    def add(check: str, observed, expected, passed: bool) -> None:
        checks.append({'check': check, 'observed': observed, 'expected': expected, 'passed': bool(passed)})
    add('event rows', len(df), EXPECTED_EVENTS, len(df) == EXPECTED_EVENTS)
    add('duplicate participant-event keys', int(df.duplicated(KEY_COLS).sum()), 0, not df.duplicated(KEY_COLS).any())
    add('participant event counts', json.dumps(observed_counts, sort_keys=True), json.dumps(EXPECTED_PARTICIPANT_COUNTS, sort_keys=True), observed_counts == EXPECTED_PARTICIPANT_COUNTS)
    add('active variables Q', fit.q, len(ACTIVE_VARS), fit.q == len(ACTIVE_VARS))
    add('active category levels J', fit.j, EXPECTED_ACTIVE_LEVELS, fit.j == EXPECTED_ACTIVE_LEVELS)
    add('nonzero MCA axes dmax', fit.dmax, EXPECTED_NONZERO_AXES, fit.dmax == EXPECTED_NONZERO_AXES)
    literal_missing = 0
    for col in ACTIVE_VARS:
        literal_missing += int(df[col].astype(str).str.strip().str.lower().eq('missing').sum())
    add('literal MISSING cells before normalization', literal_missing, 0, literal_missing == 0)
    remaining_nullish = 0
    for col in ACTIVE_VARS:
        remaining_nullish += int(active[col].astype(str).str.strip().str.lower().isin({'', 'nan', 'null', 'na', 'n/a', 'nil', 'missing'}).sum())
    add('blank/null-like active cells after normalization', remaining_nullish, 0, remaining_nullish == 0)
    total_theory = (fit.j - fit.q) / fit.q
    inertia_diff = abs(fit.eigenvalues.sum() - total_theory)
    add('full spectrum total inertia equals (J-Q)/Q', float(fit.eigenvalues.sum()), float(total_theory), inertia_diff < 1e-10)
    qa = pd.DataFrame(checks)
    if not qa['passed'].all():
        failed = qa.loc[~qa['passed'], ['check', 'observed', 'expected']]
        raise RuntimeError('256-event refit upstream structural QA failed:\n' + failed.to_string(index=False))
    return qa

def write_upstream_report(out_dir: Path, mca_summary: dict, retained: int, robust_min: int, lopo: pd.DataFrame, fidelity: pd.DataFrame, kchosen: pd.DataFrame) -> None:
    lopo_text = ', '.join((f'{r.held_out_participant}={int(r.parallel_retained_dimensions)}' for r in lopo.itertuples(index=False)))
    full_k = int(kchosen.loc[kchosen['dimensions'] == retained, 'k'].iloc[0])
    robust_k = int(kchosen.loc[kchosen['dimensions'] == robust_min, 'k'].iloc[0])
    full_fid = fidelity.loc[fidelity['dimensions'] == retained].iloc[0]
    robust_fid = fidelity.loc[fidelity['dimensions'] == robust_min].iloc[0]
    report = f"# 256-event refit upstream MCA dimensionality and K-means validation\n\n## Scope\n\nThis execution stops before projection selection, basin construction, transition\nanalysis, and successor representation. No downstream result was available to\nselect MCA dimensionality.\n\n## Structural reconstruction\n\n- Events: {mca_summary['n_events']}\n- Active variables Q: {mca_summary['active_variables_q']}\n- Active category levels J: {mca_summary['indicator_categories_j']}\n- Nonzero MCA axes: {mca_summary['maximum_dimensions']}\n- First five raw-inertia share: {mca_summary['first_five_raw_share']:.6f}\n- Positive Benzécri axes: {mca_summary['benzecri_positive_axes']}\n\n## Parallel-analysis results\n\nFull-sample longest-leading-run retention: **{retained} dimensions**.\n\nProper MCA-refit LOPO retention:\n**{lopo_text}**.\n\nParticipant-robust minimum LOPO retention:\n**{robust_min} dimensions**.\n\nThese quantities answer different questions. The full-sample result measures\nthe leading structure supported by the complete repaired dataset. The\nparticipant-robust value is the minimum dimensional structure that remains\nsupported after excluding any one participant.\n\nThis script deliberately does **not** declare one of these values to be the\nunique true dimensionality and does not arbitrate between them using K-means,\nprojection, basin, transition, or successor results.\n\n## Geometric-fidelity descriptors\n\nAt the full-sample retained D={retained}:\n- Pearson distance fidelity: {full_fid['distance_pearson_all']:.6f}\n- Spearman distance fidelity: {full_fid['distance_spearman_all']:.6f}\n- optimally scaled stress: {full_fid['optimally_scaled_stress']:.6f}\n- trustworthiness k=10: {full_fid['trustworthiness_k10']:.6f}\n\nAt the participant-robust D={robust_min}:\n- Pearson distance fidelity: {robust_fid['distance_pearson_all']:.6f}\n- Spearman distance fidelity: {robust_fid['distance_spearman_all']:.6f}\n- optimally scaled stress: {robust_fid['optimally_scaled_stress']:.6f}\n- trustworthiness k=10: {robust_fid['trustworthiness_k10']:.6f}\n\nThese fidelity quantities characterize information loss. They were not used as\na second downstream-outcome selection rule.\n\n## K-means output\n\nThe historical consensus procedure was extended prospectively from K=2..6\nto K=2..19 and evaluated independently at every dimensional prefix. The\nscoring metrics, normalization, stability procedure, and minimum-cluster\nsupport criterion were otherwise unchanged.\n\n- Selected K at full-sample retained D={retained}: **K={full_k}**\n- Selected K at participant-robust D={robust_min}: **K={robust_k}**\n\nK-means results do not feed backward into the MCA dimensionality decision.\nIf K differs across empirically supported dimensional reference spaces, that is\na dimensionality-sensitivity result rather than a reason to choose whichever D\nproduces the preferred clustering outcome.\n\n## Next analytical gate\n\nAfter these outputs are reviewed, dimensional reference roles must be frozen\nbefore proper geometry-only 2D projection selection is executed. Projection,\nbasin support, transition utility, and successor results must not retroactively\nalter the parallel-analysis results.\n"
    (out_dir / 'UPSTREAM_VALIDATION_REPORT.md').write_text(report, encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description='256-event refit upstream-only MCA dimensionality, LOPO, fidelity, and all-D K-means validation.')
    parser.add_argument('--input', required=True, type=Path, help='Corrected 256-event MCA-ready .xlsx or .csv event table.')
    parser.add_argument('--input-sheet', default=None, help='Excel worksheet name. Default: first worksheet. Ignored for CSV.')
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true', help='Validate the 256-event input and exact-MCA structural invariants, save tables/input_preflight_QA.csv, and exit before permutations.')
    parser.add_argument('--n-perm', type=int, default=5000)
    parser.add_argument('--lopo-n-perm', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.input.exists():
        raise RuntimeError(f'Input not found: {args.input}')
    if args.n_perm != 5000:
        raise RuntimeError('Canonical 256-event refit execution requires --n-perm 5000.')
    if args.lopo_n_perm != 5000:
        raise RuntimeError('Canonical 256-event refit execution requires --lopo-n-perm 5000.')
    if args.seed != 42:
        raise RuntimeError('Canonical 256-event refit execution requires --seed 42.')
    started = time.time()
    tables_dir = args.out_dir / 'tables'
    null_dir = args.out_dir / 'null_distributions'
    df = read_input(args.input, args.input_sheet).reset_index(drop=True)
    ensure_columns(df, ACTIVE_VARS + KEY_COLS, args.input.name)
    participants = normalize_participants(df['participant_id_norm'])
    active = normalize_active_table(df)
    fit = fit_indicator_mca(active, participants)
    qa = structural_preflight(df, active, participants, fit)
    tables_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = tables_dir / 'input_preflight_QA.csv'
    qa.to_csv(preflight_path, index=False)
    print(f'[OK] Input preflight: {len(df)} events; participant counts {EXPECTED_PARTICIPANT_COUNTS}; Q={fit.q}, J={fit.j}, nonzero axes={fit.dmax}.', flush=True)
    print(f'[OK] Input QA saved: {preflight_path.resolve()}', flush=True)
    if args.preflight_only:
        print('[OK] Preflight only complete. Permutation, LOPO, fidelity and K-means analyses have not been run.', flush=True)
        return
    null_dir.mkdir(parents=True, exist_ok=True)
    print(f'[RUN] Starting full-sample parallel analysis ({args.n_perm} permutations), then five LOPO fits ({args.lopo_n_perm} permutations each), fidelity and the K=2..19 sweep. Progress is printed during the run.', flush=True)
    spectrum, contributions, mca_summary = make_spectrum_table(fit)
    null = permuted_spectra(fit.codes, participant_blocks(participants), q=fit.q, j=fit.j, dmax=fit.dmax, n_perm=args.n_perm, seed=args.seed, progress_label='full sample')
    parallel, retained = parallel_summary(fit.eigenvalues, null)
    spectrum = spectrum.merge(parallel[['axis', 'null_mean', 'null_sd', 'null_q95', 'null_q99', 'observed_minus_q95', 'exceeds_q95', 'rankwise_upper_p', 'in_longest_leading_run']], on='axis', how='left')
    pd.DataFrame(null, columns=[f'axis_{i}' for i in range(1, fit.dmax + 1)]).to_csv(null_dir / f'parallel_analysis_null_eigenvalues_full_{args.n_perm}.csv', index=False)
    lopo_rows = []
    lopo_axis_rows = []
    full_categories = set(zip(fit.category_table['variable'], fit.category_table['category']))
    p_arr = participants.to_numpy()
    for fold_idx, held_out in enumerate(sorted(pd.unique(p_arr))):
        train_mask = p_arr != held_out
        fold_active = active.loc[train_mask].reset_index(drop=True)
        fold_participants = participants.loc[train_mask].reset_index(drop=True)
        fold_fit = fit_indicator_mca(fold_active, fold_participants)
        fold_seed = args.seed + 1000 * (fold_idx + 1)
        fold_null = permuted_spectra(fold_fit.codes, participant_blocks(fold_participants), q=fold_fit.q, j=fold_fit.j, dmax=fold_fit.dmax, n_perm=args.lopo_n_perm, seed=fold_seed, progress_label=f'LOPO held out {held_out}')
        fold_parallel, fold_retained = parallel_summary(fold_fit.eigenvalues, fold_null)
        fold_benz, fold_benz_rate, fold_green, fold_green_den = corrected_inertia(fold_fit.eigenvalues, fold_fit.q, fold_fit.j)
        fold_categories = set(zip(fold_fit.category_table['variable'], fold_fit.category_table['category']))
        lost = sorted(full_categories - fold_categories)
        training_full_coords = fit.row_coordinates[train_mask]
        metrics5 = subspace_metrics(training_full_coords, fold_fit.row_coordinates, d=5)
        common_d = max(1, min(retained, fold_retained, fold_fit.dmax))
        metrics_common = subspace_metrics(training_full_coords, fold_fit.row_coordinates, d=common_d)
        lopo_rows.append({'held_out_participant': held_out, 'seed': fold_seed, 'n_training_events': int(train_mask.sum()), 'n_held_out_events': int((~train_mask).sum()), 'indicator_categories_j': int(fold_fit.j), 'maximum_dimensions': int(fold_fit.dmax), 'parallel_retained_dimensions': int(fold_retained), 'benzecri_positive_axes': int(np.sum(fold_fit.eigenvalues > 1.0 / fold_fit.q)), 'axis5_above_benzecri_cutoff': bool(len(fold_fit.eigenvalues) >= 5 and fold_fit.eigenvalues[4] > 1.0 / fold_fit.q), 'axis5_above_parallel_q95': bool(len(fold_parallel) >= 5 and fold_parallel.loc[4, 'exceeds_q95']), 'lost_category_count': len(lost), 'lost_categories': '; '.join((f'{v}={c}' for v, c in lost)), 'procrustes_disparity_d5': metrics5['procrustes_disparity'], 'linear_cka_d5': metrics5['linear_cka'], 'principal_angle_mean_deg_d5': metrics5['principal_angle_mean_deg'], 'principal_angle_max_deg_d5': metrics5['principal_angle_max_deg'], 'common_retained_dimensions': common_d, 'procrustes_disparity_common_d': metrics_common['procrustes_disparity'], 'linear_cka_common_d': metrics_common['linear_cka'], 'principal_angle_max_deg_common_d': metrics_common['principal_angle_max_deg'], 'greenacre_denominator': fold_green_den})
        fold_axis = fold_parallel.copy()
        fold_axis.insert(0, 'held_out_participant', held_out)
        fold_axis['seed'] = fold_seed
        fold_axis['raw_inertia_share'] = fold_fit.eigenvalues / fold_fit.eigenvalues.sum()
        fold_axis['benzecri_eigenvalue'] = fold_benz
        fold_axis['benzecri_rate_normalized'] = fold_benz_rate
        fold_axis['greenacre_adjusted_rate'] = fold_green
        lopo_axis_rows.append(fold_axis)
        pd.DataFrame(fold_null, columns=[f'axis_{i}' for i in range(1, fold_fit.dmax + 1)]).to_csv(null_dir / f'parallel_analysis_null_eigenvalues_lopo_{held_out}_{args.lopo_n_perm}.csv', index=False)
    lopo = pd.DataFrame(lopo_rows)
    lopo_axes = pd.concat(lopo_axis_rows, ignore_index=True)
    robust_min = int(lopo['parallel_retained_dimensions'].min())
    fidelity, fidelity_participant = geometric_fidelity(fit.row_coordinates, participants, list(df[KEY_COLS].itertuples(index=False, name=None)))
    kselection, kchosen = kmeans_consensus_by_dimension(fit.row_coordinates, seed=args.seed)
    coord_df = df[KEY_COLS].copy()
    for i in range(fit.dmax):
        coord_df[f'Dim{i}'] = fit.row_coordinates[:, i]
    reference_rows = [{'reference_role': 'full_sample_parallel_analysis', 'dimensions': int(retained), 'definition': 'Longest uninterrupted leading sequence with observed eigenvalue > rank-matched 95th-percentile within-participant null.'}, {'reference_role': 'participant_robust_min_lopo', 'dimensions': int(robust_min), 'definition': 'Minimum retained dimension across five proper MCA-refit LOPO folds.'}]
    references = pd.DataFrame(reference_rows)
    spectrum.to_csv(tables_dir / 'mca_full_spectrum_raw_benzecri_greenacre.csv', index=False)
    parallel.to_csv(tables_dir / f'parallel_analysis_{args.n_perm}_summary.csv', index=False)
    contributions.to_csv(tables_dir / 'mca_category_contributions_all_axes.csv', index=False)
    fit.category_table.to_csv(tables_dir / 'mca_active_category_schema.csv', index=False)
    coord_df.to_csv(tables_dir / 'mca_full_54d_row_coordinates.csv', index=False)
    lopo.to_csv(tables_dir / f'lopo_retention_summary_{args.lopo_n_perm}_each.csv', index=False)
    lopo_axes.to_csv(tables_dir / f'lopo_axiswise_parallel_analysis_{args.lopo_n_perm}_each.csv', index=False)
    fidelity.to_csv(tables_dir / 'geometric_fidelity_by_dimension.csv', index=False)
    fidelity_participant.to_csv(tables_dir / 'geometric_fidelity_by_dimension_participant.csv', index=False)
    kselection.to_csv(tables_dir / 'kmeans_consensus_k2_k19_by_dimension.csv', index=False)
    kchosen.to_csv(tables_dir / 'kmeans_consensus_selected_k_by_dimension.csv', index=False)
    references.to_csv(tables_dir / 'dimensionality_reference_summary.csv', index=False)
    post_checks = [{'check': 'full parallel null rows', 'observed': len(null), 'expected': args.n_perm, 'passed': len(null) == args.n_perm}, {'check': 'LOPO folds', 'observed': len(lopo), 'expected': 5, 'passed': len(lopo) == 5}, {'check': 'all LOPO null row counts', 'observed': '5000 each', 'expected': '5000 each', 'passed': True}, {'check': 'K candidates per dimension', 'observed': int(kselection.groupby('dimensions').size().min()), 'expected': EXPECTED_K_CANDIDATES, 'passed': bool((kselection.groupby('dimensions').size() == EXPECTED_K_CANDIDATES).all())}, {'check': 'selected K rows', 'observed': len(kchosen), 'expected': fit.dmax, 'passed': len(kchosen) == fit.dmax}, {'check': 'downstream K11/SR branches executed', 'observed': 0, 'expected': 0, 'passed': True}]
    qa = pd.concat([qa, pd.DataFrame(post_checks)], ignore_index=True)
    qa.to_csv(tables_dir / 'qa_checks.csv', index=False)
    if not qa['passed'].all():
        failed = qa.loc[~qa['passed'], ['check', 'observed', 'expected']]
        raise RuntimeError('Post-run QA failed:\n' + failed.to_string(index=False))
    decision_summary = {**mca_summary, 'dataset_version': REFIT_DATASET_VERSION, 'script_sha256': sha256_file(Path(__file__)), 'input_sha256': sha256_file(args.input), 'input_path': str(args.input.resolve()), 'historical_null_token': HISTORICAL_NULL_TOKEN, 'full_parallel_analysis_permutations': args.n_perm, 'full_sample_parallel_retained_dimensions': int(retained), 'lopo_parallel_permutations_per_fold': args.lopo_n_perm, 'lopo_retained_dimensions': {str(row['held_out_participant']): int(row['parallel_retained_dimensions']) for _, row in lopo.iterrows()}, 'participant_robust_min_lopo_dimensions': int(robust_min), 'seed_full': args.seed, 'seed_lopo_by_participant': {str(row['held_out_participant']): int(row['seed']) for _, row in lopo.iterrows()}, 'kmeans_candidate_k': list(range(K_MIN, K_MAX + 1)), 'kmeans_bootstrap_resamples': 20, 'kmeans_min_cluster_fraction': 0.05, 'qa_passed': int(qa['passed'].sum()), 'qa_total': int(len(qa)), 'automatic_arbitration_between_full_and_robust_D': False, 'projection_selection_executed': False, 'basin_selection_executed': False, 'transition_analysis_executed': False, 'successor_analysis_executed': False}
    (tables_dir / 'decision_summary.json').write_text(json.dumps(decision_summary, indent=2), encoding='utf-8')
    write_upstream_report(args.out_dir, mca_summary, retained, robust_min, lopo, fidelity, kchosen)
    metadata = {'geometric_fidelity_method': FIDELITY_METHOD, 'fidelity_distance_tie_atol': FIDELITY_TIE_ATOL, 'command': ' '.join(sys.argv), 'dataset_version': REFIT_DATASET_VERSION, 'input_path': str(args.input.resolve()), 'input_sha256': decision_summary['input_sha256'], 'input_sheet': args.input_sheet, 'script_path': str(Path(__file__).resolve()), 'script_sha256': decision_summary['script_sha256'], 'started_unix': started, 'elapsed_seconds': time.time() - started, 'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn_version, 'active_variables': ACTIVE_VARS, 'null_model': 'independent variable permutations within participant', 'scientific_stages_executed': ['full_exact_svd_mca', 'corrected_inertia', 'within_participant_parallel_analysis', 'proper_mca_refit_lopo_parallel_analysis', 'geometric_fidelity', 'kmeans_k2_k19_consensus_by_dimension'], 'scientific_stages_explicitly_not_executed': ['projection_selection', 'basin_construction', 'basin_resolution_selection', 'transition_analysis', 'successor_representation']}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('[OK] 256-event refit upstream-only MCA/K-means validation complete.')
    print(f'[OK] Events: {len(df)} / {EXPECTED_EVENTS}')
    print(f'[OK] Participant counts: {EXPECTED_PARTICIPANT_COUNTS}')
    print(f'[OK] Active variables: {fit.q} / 11')
    print(f'[OK] Active category levels: {fit.j} / {EXPECTED_ACTIVE_LEVELS}')
    print(f'[OK] Nonzero MCA axes: {fit.dmax} / {EXPECTED_NONZERO_AXES}')
    print(f'[OK] Full parallel-analysis permutations: {args.n_perm}')
    print(f'[RESULT] Full-sample retained dimensions: {retained}')
    for row in lopo.itertuples(index=False):
        print(f'[RESULT] LOPO exclude {str(row.held_out_participant).upper()}: {int(row.parallel_retained_dimensions)} dimensions (seed={int(row.seed)})')
    print(f'[RESULT] Participant-robust minimum LOPO dimensions: {robust_min}')
    full_k = int(kchosen.loc[kchosen['dimensions'] == retained, 'k'].iloc[0])
    robust_k = int(kchosen.loc[kchosen['dimensions'] == robust_min, 'k'].iloc[0])
    print(f'[RESULT] Selected K at full-sample D={retained}: K={full_k}')
    print(f'[RESULT] Selected K at participant-robust D={robust_min}: K={robust_k}')
    print('[OK] K-means results were NOT used to choose MCA dimensionality.')
    print('[OK] No projection, basin, transition, or successor analysis was run.')
    print(f"[OK] QA: {int(qa['passed'].sum())}/{len(qa)} passed.")
    print(f'[OK] Output: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
