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
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, mutual_info_score, silhouette_score
KEY = ['participant_id_norm', 'hippo_id_norm']
FULL_ACTIVE = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
CONSTRUCT_FAMILIES: dict[str, list[str]] = {'context': ['context_type_primary', 'context_type_secondary'], 'hippocampal_mechanism': ['primary_hipp_mech', 'secondary_hipp_mech'], 'anchor_landmark': ['anchor_type', 'landmark'], 'representational_organization': ['frame', 'orient', 'perspective'], 'semantic_content': ['event_meta_superordinate_theme_primary'], 'affective_content': ['event_sentiment_primary']}
VARIABLE_TO_FAMILY = {variable: family for family, variables in CONSTRUCT_FAMILIES.items() for variable in variables}
D_VALUES = list(range(2, 14))
K_VALUES = list(range(2, 7))
NULLISH = {'', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing'}
NULL_TOKEN = 'none'
EXPECTED_EVENTS = 256
EXPECTED_PARTICIPANTS = 5
EXPECTED_FULL_LEVELS = 65
EXPECTED_FULL_DMAX = 54
EXPECTED_PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
HISTORICAL_MODEL = (5, 4)
D4K4_MODEL = (4, 4)
FULL_REFERENCE_MODEL = None
ROBUST_REFERENCE_MODEL = None
DATASET_VERSION = 'chronological_256_v2'
MIN_CLUSTER_FRACTION = 0.05
MIN_PARTICIPANT_SUPPORT_PER_CLUSTER = 2
PARETO_METRICS = ['silhouette', 'bootstrap_stability_ari', 'heldout_family_mean_null_corrected_R2', 'heldout_family_count_global_q05', 'lopo_ari_mean', 'adjacent_D_ari_mean']

@dataclass
class MCAFit:
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    row_coordinates: np.ndarray
    category_table: pd.DataFrame
    q: int
    j: int
    dmax: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--event-table', required=True, type=Path)
    ap.add_argument('--event-sheet', default='Sheet1')
    ap.add_argument('--full-coordinates', required=True, type=Path)
    ap.add_argument('--upstream-selected-k', type=Path, required=True)
    ap.add_argument('--upstream-dir', type=Path, default=None, help="Completed MCA run; defaults to the parent of the coordinates' tables folder.")
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-bootstrap', type=int, default=20)
    ap.add_argument('--permutation-chunk', type=int, default=250)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.n_perm < 1000:
        ap.error('--n-perm must be at least 1000; canonical execution uses 10000.')
    if args.n_bootstrap < 2:
        ap.error('--n-bootstrap must be at least 2; canonical execution uses 20.')
    if args.permutation_chunk < 1:
        ap.error('--permutation-chunk must be positive.')
    if args.seed != 42:
        ap.error('Canonical execution requires --seed 42.')
    return args

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_table(path: Path, sheet: str | None=None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path, low_memory=False)
    if suffix in {'.xlsx', '.xlsm', '.xls'}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    raise RuntimeError(f'Unsupported file type: {path}')

def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f'{label} missing required columns: {missing}')

def normalize_category_series(series: pd.Series) -> pd.Series:
    out = series.where(~series.isna(), NULL_TOKEN)
    out = out.astype(str).str.strip()
    lower = out.str.lower()
    out = out.where(~lower.isin(NULLISH), NULL_TOKEN)
    return out

def normalize_active(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: normalize_category_series(df[column]) for column in variables}, index=df.index)

def bh_adjust(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    candidate = ranked * m / np.arange(1, m + 1)
    candidate = np.minimum.accumulate(candidate[::-1])[::-1]
    candidate = np.minimum(candidate, 1.0)
    reordered = np.empty_like(candidate)
    reordered[order] = candidate
    adjusted[finite_indices] = reordered
    return adjusted

def normalize01(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)

def model_id(d: int, k: int) -> str:
    return f'D{d}_K{k}'

def entropy_from_codes(codes: np.ndarray) -> float:
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))

def build_indicator(active: pd.DataFrame, variables: Sequence[str], participants: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(active)
    blocks: list[np.ndarray] = []
    categories: list[tuple[str, str]] = []
    for column in variables:
        values = active[column].astype(str).to_numpy()
        levels = sorted(pd.unique(values).tolist())
        lookup = {value: index for index, value in enumerate(levels)}
        codes = np.fromiter((lookup[value] for value in values), dtype=np.int16, count=n)
        block = np.zeros((n, len(levels)), dtype=float)
        block[np.arange(n), codes] = 1.0
        blocks.append(block)
        categories.extend(((column, level) for level in levels))
    indicator = np.concatenate(blocks, axis=1)
    frequencies = indicator.sum(axis=0).astype(int)
    participant_values = participants.astype(str).to_numpy()
    supports = [int(pd.unique(participant_values[indicator[:, index] == 1]).size) for index in range(indicator.shape[1])]
    category_table = pd.DataFrame({'category_index': np.arange(indicator.shape[1]), 'variable': [variable for variable, _ in categories], 'category': [category for _, category in categories], 'frequency': frequencies, 'participant_support': supports})
    return (indicator, category_table)

def fit_mca(active: pd.DataFrame, variables: Sequence[str], participants: pd.Series) -> MCAFit:
    indicator, category_table = build_indicator(active, variables, participants)
    n, j = indicator.shape
    q = len(variables)
    dmax = min(n - 1, j - q)
    frequency = indicator.sum(axis=0)
    standardized = (indicator - frequency[None, :] / n) / np.sqrt(q * frequency[None, :])
    u, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    singular_values = singular_values[:dmax]
    eigenvalues = np.maximum(singular_values ** 2, 0.0)
    row_coordinates = math.sqrt(n) * u[:, :dmax] * singular_values[None, :]
    return MCAFit(eigenvalues=eigenvalues, singular_values=singular_values, row_coordinates=row_coordinates, category_table=category_table, q=q, j=j, dmax=dmax)

def spectrum_table(fit: MCAFit) -> pd.DataFrame:
    total = fit.eigenvalues.sum()
    return pd.DataFrame({'axis': np.arange(1, fit.dmax + 1), 'dimension_name': [f'Dim{i}' for i in range(fit.dmax)], 'eigenvalue_raw': fit.eigenvalues, 'raw_inertia_share': fit.eigenvalues / total, 'raw_inertia_cumulative': np.cumsum(fit.eigenvalues / total)})

def fit_partition(coordinates: np.ndarray, dimensions: int, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coordinates[:, :dimensions])

def bootstrap_stability_ari(x: np.ndarray, k: int, n_bootstrap: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    all_labels: list[np.ndarray] = []
    n = len(x)
    for bootstrap_index in range(n_bootstrap):
        sampled_rows = rng.randint(0, n, size=n)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed + bootstrap_index)
        model.fit(x[sampled_rows])
        all_labels.append(model.predict(x))
    values = [adjusted_rand_score(all_labels[i], all_labels[j]) for i in range(len(all_labels)) for j in range(i + 1, len(all_labels))]
    return float(np.mean(values))

def bias_corrected_cramers_v_from_table(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = int(table.sum())
    rows, columns = table.shape
    if n <= 1 or rows < 2 or columns < 2:
        return np.nan
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    valid = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (columns - 1) * (rows - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))

def categorical_codes(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    normalized = normalize_category_series(series)
    levels = sorted(normalized.unique().tolist())
    lookup = {level: index for index, level in enumerate(levels)}
    codes = np.fromiter((lookup[value] for value in normalized), dtype=np.int16, count=len(normalized))
    return (codes, levels)

def contingency(cluster_labels: np.ndarray, target_codes: np.ndarray, n_clusters: int, n_levels: int) -> np.ndarray:
    table = np.zeros((n_clusters, n_levels), dtype=int)
    np.add.at(table, (cluster_labels, target_codes), 1)
    return table

def participant_support_metrics(labels: np.ndarray, participants: np.ndarray, d: int, k: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    minimum_support = 10 ** 9
    maximum_participant_fraction = 0.0
    for cluster in range(k):
        mask = labels == cluster
        cluster_participants = pd.Series(participants[mask])
        counts = cluster_participants.value_counts()
        n_events = int(mask.sum())
        represented = int(len(counts))
        max_fraction = float(counts.max() / n_events) if n_events > 0 else np.nan
        minimum_support = min(minimum_support, represented)
        maximum_participant_fraction = max(maximum_participant_fraction, max_fraction if np.isfinite(max_fraction) else 0.0)
        rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'cluster': cluster + 1, 'n_events': n_events, 'participants_represented': represented, 'maximum_single_participant_fraction': max_fraction, 'participant_counts': json.dumps({str(index): int(value) for index, value in counts.items()}, sort_keys=True)})
    participant_codes, participant_levels = pd.factorize(participants, sort=True)
    table = contingency(labels, participant_codes.astype(np.int16), k, len(participant_levels))
    participant_v = bias_corrected_cramers_v_from_table(table)
    return (rows, {'min_participant_support_per_cluster': int(minimum_support), 'max_single_participant_fraction_any_cluster': float(maximum_participant_fraction), 'participant_cluster_bias_corrected_cramers_v': participant_v})

def build_full_macro_grid(coordinates: np.ndarray, participants: np.ndarray, seed: int, n_bootstrap: int) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray], pd.DataFrame]:
    rows: list[dict] = []
    partitions: dict[tuple[int, int], np.ndarray] = {}
    participant_rows: list[dict] = []
    for d in D_VALUES:
        x = coordinates[:, :d]
        for k in K_VALUES:
            labels = fit_partition(coordinates, d, k, seed)
            partitions[d, k] = labels
            sizes = np.bincount(labels, minlength=k)
            support_rows, support_summary = participant_support_metrics(labels, participants, d, k)
            participant_rows.extend(support_rows)
            rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'silhouette': float(silhouette_score(x, labels)), 'calinski_harabasz': float(calinski_harabasz_score(x, labels)), 'davies_bouldin': float(davies_bouldin_score(x, labels)), 'bootstrap_stability_ari': bootstrap_stability_ari(x, k, n_bootstrap=n_bootstrap, seed=seed), 'min_cluster_size': int(sizes.min()), 'max_cluster_size': int(sizes.max()), 'min_cluster_fraction': float(sizes.min() / len(labels)), 'max_cluster_fraction': float(sizes.max() / len(labels)), **support_summary})
        print(f'[full-grid] completed D={d}', flush=True)
    grid = pd.DataFrame(rows)
    consensus_blocks = []
    for d, block in grid.groupby('dimensions', sort=True):
        block = block.copy()
        block['silhouette_norm'] = normalize01(block['silhouette'])
        block['ch_norm'] = normalize01(block['calinski_harabasz'])
        block['inverse_db_norm'] = normalize01(1.0 / block['davies_bouldin'])
        block['stability_norm'] = normalize01(block['bootstrap_stability_ari'])
        block['macro_k2_k6_consensus_score'] = block[['silhouette_norm', 'ch_norm', 'inverse_db_norm', 'stability_norm']].mean(axis=1)
        block['macro_consensus_eligible_min_fraction_005'] = block['min_cluster_fraction'] >= MIN_CLUSTER_FRACTION
        eligible = block[block['macro_consensus_eligible_min_fraction_005']]
        selected_index = block['macro_k2_k6_consensus_score'].idxmax() if eligible.empty else eligible['macro_k2_k6_consensus_score'].idxmax()
        block['macro_k2_k6_consensus_selected'] = block.index == selected_index
        consensus_blocks.append(block)
    grid = pd.concat(consensus_blocks, ignore_index=True)
    return (grid, partitions, pd.DataFrame(participant_rows))

def adjusted_standardized_residuals(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=float)
    n = table.sum()
    row = table.sum(axis=1, keepdims=True)
    column = table.sum(axis=0, keepdims=True)
    expected = row @ column / n
    row_proportion = row / n
    column_proportion = column / n
    denominator = np.sqrt(expected * (1.0 - row_proportion) * (1.0 - column_proportion))
    return np.divide(table - expected, denominator, out=np.full_like(expected, np.nan, dtype=float), where=denominator > 0)

def characterize_full_features(events: pd.DataFrame, partitions: dict[tuple[int, int], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    association_rows: list[dict] = []
    cell_rows: list[dict] = []
    normalized = normalize_active(events, FULL_ACTIVE)
    for d in D_VALUES:
        for k in K_VALUES:
            labels = partitions[d, k]
            for variable in FULL_ACTIVE:
                target_codes, levels = categorical_codes(normalized[variable])
                table = contingency(labels, target_codes, k, len(levels))
                chi2, p_value, _, expected = chi2_contingency(table, correction=False)
                residuals = adjusted_standardized_residuals(table)
                h_target = entropy_from_codes(target_codes)
                info_fraction = float(mutual_info_score(labels, target_codes) / h_target) if h_target > 0 else np.nan
                association_rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'variable': variable, 'construct_family': VARIABLE_TO_FAMILY[variable], 'n_levels': len(levels), 'chi2': float(chi2), 'chi2_p': float(p_value), 'bias_corrected_cramers_v': bias_corrected_cramers_v_from_table(table), 'adjusted_mutual_information': float(adjusted_mutual_info_score(target_codes, labels)), 'information_fraction': info_fraction, 'inference_status': 'DESCRIPTIVE_FIELD_IS_ACTIVE_MCA_INPUT'})
                for cluster in range(k):
                    for level_index, level in enumerate(levels):
                        cell_rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'cluster': cluster + 1, 'variable': variable, 'construct_family': VARIABLE_TO_FAMILY[variable], 'level': level, 'observed': int(table[cluster, level_index]), 'expected': float(expected[cluster, level_index]), 'adjusted_standardized_residual': float(residuals[cluster, level_index]), 'cluster_fraction': float(table[cluster, level_index] / table[cluster].sum()) if table[cluster].sum() > 0 else np.nan})
    associations = pd.DataFrame(association_rows)
    associations['bh_q_within_model'] = np.nan
    for _, indices in associations.groupby('model_id').groups.items():
        associations.loc[indices, 'bh_q_within_model'] = bh_adjust(associations.loc[indices, 'chi2_p'])
    associations['bh_q_global'] = bh_adjust(associations['chi2_p'])
    return (associations, pd.DataFrame(cell_rows))

def equal_weight_family_target_matrix(events: pd.DataFrame, variables: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    normalized = normalize_active(events, variables)
    blocks: list[np.ndarray] = []
    schema_rows: list[dict] = []
    for variable in variables:
        values = normalized[variable].astype(str).to_numpy()
        levels = sorted(pd.unique(values).tolist())
        lookup = {level: index for index, level in enumerate(levels)}
        codes = np.fromiter((lookup[value] for value in values), dtype=np.int16, count=len(values))
        one_hot = np.zeros((len(values), len(levels)), dtype=float)
        one_hot[np.arange(len(values)), codes] = 1.0
        centered = one_hot - one_hot.mean(axis=0, keepdims=True)
        total_ss = float(np.sum(centered ** 2))
        if total_ss <= 0:
            raise RuntimeError(f'Held-out variable {variable} has zero categorical variance.')
        scaled = centered / math.sqrt(total_ss)
        blocks.append(scaled)
        for level_index, level in enumerate(levels):
            schema_rows.append({'variable': variable, 'level': level, 'level_index': level_index, 'frequency': int(np.sum(codes == level_index)), 'block_total_ss_before_scaling': total_ss, 'block_weighting': 'centered one-hot block Frobenius-scaled to total SS=1'})
    target = np.concatenate(blocks, axis=1)
    return (target, pd.DataFrame(schema_rows))

def pseudo_f_r2_from_labels(target: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    target = np.asarray(target, dtype=float)
    grand = target.mean(axis=0)
    total = float(np.sum((target - grand) ** 2))
    between = 0.0
    for cluster in range(k):
        sub = target[labels == cluster]
        if len(sub) == 0:
            raise RuntimeError('Empty cluster in pseudo-F computation.')
        difference = sub.mean(axis=0) - grand
        between += float(len(sub) * np.sum(difference ** 2))
    within = total - between
    n = len(target)
    if within <= 0 or n <= k:
        return (np.nan, np.nan)
    pseudo_f = between / (k - 1) / (within / (n - k))
    r2 = between / total if total > 0 else np.nan
    return (float(pseudo_f), float(r2))

def make_within_participant_row_permutation_bank(participants: np.ndarray, n_permutations: int, seed: int) -> np.ndarray:
    participants = np.asarray(participants, dtype=str)
    blocks = [np.flatnonzero(participants == participant) for participant in sorted(np.unique(participants))]
    rng = np.random.default_rng(seed)
    n = len(participants)
    bank = np.empty((n_permutations, n), dtype=np.int16)
    for permutation_index in range(n_permutations):
        order = np.arange(n, dtype=np.int16)
        for indices in blocks:
            order[indices] = rng.permutation(indices)
        bank[permutation_index] = order
    return bank

def vectorized_family_permutation_test(target: np.ndarray, labels_by_model: dict[str, np.ndarray], model_k: dict[str, int], permutation_bank: np.ndarray, chunk_size: int) -> pd.DataFrame:
    target = np.asarray(target, dtype=float)
    grand = target.mean(axis=0)
    total_ss = float(np.sum((target - grand) ** 2))
    n = len(target)
    cache = {}
    result = {}
    for mid, labels in labels_by_model.items():
        k = model_k[mid]
        one_hot_cluster = np.zeros((n, k), dtype=float)
        one_hot_cluster[np.arange(n), labels] = 1.0
        sizes = one_hot_cluster.sum(axis=0)
        observed_f, observed_r2 = pseudo_f_r2_from_labels(target, labels, k)
        cache[mid] = (one_hot_cluster, sizes, k)
        result[mid] = {'observed_pseudo_F': observed_f, 'observed_R2': observed_r2, 'null_F_sum': 0.0, 'null_F_sumsq': 0.0, 'null_R2_sum': 0.0, 'null_R2_sumsq': 0.0, 'F_exceedances': 0, 'R2_exceedances': 0, 'n_null': 0}
    for start in range(0, len(permutation_bank), chunk_size):
        stop = min(start + chunk_size, len(permutation_bank))
        orders = permutation_bank[start:stop].astype(np.intp, copy=False)
        permuted_target = target[orders]
        batch_size = stop - start
        for mid, (cluster_matrix, sizes, k) in cache.items():
            group_sums = np.einsum('nk,bnp->bkp', cluster_matrix, permuted_target, optimize=True)
            means = group_sums / sizes[None, :, None]
            difference = means - grand[None, None, :]
            between = np.sum(sizes[None, :, None] * difference ** 2, axis=(1, 2))
            r2 = between / total_ss
            within = total_ss - between
            pseudo_f = between / (k - 1) / (within / (n - k))
            record = result[mid]
            record['null_F_sum'] += float(np.sum(pseudo_f))
            record['null_F_sumsq'] += float(np.sum(pseudo_f ** 2))
            record['null_R2_sum'] += float(np.sum(r2))
            record['null_R2_sumsq'] += float(np.sum(r2 ** 2))
            record['F_exceedances'] += int(np.count_nonzero(pseudo_f >= record['observed_pseudo_F']))
            record['R2_exceedances'] += int(np.count_nonzero(r2 >= record['observed_R2']))
            record['n_null'] += batch_size
    rows = []
    for mid, record in result.items():
        m = record['n_null']
        null_f_mean = record['null_F_sum'] / m
        null_r2_mean = record['null_R2_sum'] / m
        null_f_variance = max(0.0, (record['null_F_sumsq'] - m * null_f_mean ** 2) / max(1, m - 1))
        null_r2_variance = max(0.0, (record['null_R2_sumsq'] - m * null_r2_mean ** 2) / max(1, m - 1))
        null_f_sd = math.sqrt(null_f_variance)
        null_r2_sd = math.sqrt(null_r2_variance)
        rows.append({'model_id': mid, 'observed_pseudo_F': record['observed_pseudo_F'], 'observed_R2': record['observed_R2'], 'null_mean_pseudo_F': null_f_mean, 'null_sd_pseudo_F': null_f_sd, 'null_mean_R2': null_r2_mean, 'null_sd_R2': null_r2_sd, 'null_corrected_R2': record['observed_R2'] - null_r2_mean, 'z_vs_null_pseudo_F': (record['observed_pseudo_F'] - null_f_mean) / null_f_sd if null_f_sd > 0 else np.nan, 'permutation_p_pseudo_F': (record['F_exceedances'] + 1) / (m + 1), 'permutation_p_R2': (record['R2_exceedances'] + 1) / (m + 1), 'n_permutations': m})
    return pd.DataFrame(rows)

def variable_alignment_descriptive(events: pd.DataFrame, variables: Sequence[str], labels_by_model: dict[str, np.ndarray], model_dimensions: dict[str, int], model_k: dict[str, int], family: str) -> pd.DataFrame:
    rows = []
    normalized = normalize_active(events, variables)
    for mid, labels in labels_by_model.items():
        k = model_k[mid]
        d = model_dimensions[mid]
        for variable in variables:
            target_codes, levels = categorical_codes(normalized[variable])
            table = contingency(labels, target_codes, k, len(levels))
            h = entropy_from_codes(target_codes)
            rows.append({'construct_family': family, 'model_id': mid, 'dimensions': d, 'k': k, 'variable': variable, 'n_levels': len(levels), 'bias_corrected_cramers_v': bias_corrected_cramers_v_from_table(table), 'adjusted_mutual_information': float(adjusted_mutual_info_score(target_codes, labels)), 'information_fraction': float(mutual_info_score(labels, target_codes) / h) if h > 0 else np.nan, 'inference_status': 'DESCRIPTIVE_VARIABLE_COMPONENT_OF_HELD_OUT_FAMILY'})
    return pd.DataFrame(rows)

def evaluate_construct_blind_families(events: pd.DataFrame, participants: np.ndarray, full_partitions: dict[tuple[int, int], np.ndarray], permutation_bank: np.ndarray, output_coordinates: Path, output_spectra: Path, n_permutations: int, permutation_chunk: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_rows: list[pd.DataFrame] = []
    variable_rows: list[pd.DataFrame] = []
    family_schema_rows: list[pd.DataFrame] = []
    for family, held_out_variables in CONSTRUCT_FAMILIES.items():
        retained_variables = [variable for variable in FULL_ACTIVE if variable not in held_out_variables]
        blind_active = normalize_active(events, retained_variables)
        blind_fit = fit_mca(blind_active, retained_variables, pd.Series(participants))
        if blind_fit.dmax < max(D_VALUES):
            raise RuntimeError(f'{family}-blind MCA has only {blind_fit.dmax} dimensions; D13 is unavailable.')
        coordinates_out = events[KEY].copy()
        for dimension in range(blind_fit.dmax):
            coordinates_out[f'Dim{dimension}'] = blind_fit.row_coordinates[:, dimension]
        coordinates_out.to_csv(output_coordinates / f'{family}_blind_coordinates.csv', index=False)
        spectrum_table(blind_fit).to_csv(output_spectra / f'{family}_blind_mca_spectrum.csv', index=False)
        blind_fit.category_table.to_csv(output_spectra / f'{family}_blind_category_schema.csv', index=False)
        labels_by_model: dict[str, np.ndarray] = {}
        model_dimensions: dict[str, int] = {}
        model_k: dict[str, int] = {}
        similarity_rows = []
        for d in D_VALUES:
            for k in K_VALUES:
                mid = model_id(d, k)
                labels = fit_partition(blind_fit.row_coordinates, d, k, seed)
                labels_by_model[mid] = labels
                model_dimensions[mid] = d
                model_k[mid] = k
                full_labels = full_partitions[d, k]
                similarity_rows.append({'model_id': mid, 'dimensions': d, 'k': k, 'ari_full_vs_construct_blind': float(adjusted_rand_score(full_labels, labels)), 'ami_full_vs_construct_blind': float(adjusted_mutual_info_score(full_labels, labels))})
        target, target_schema = equal_weight_family_target_matrix(events, held_out_variables)
        target_schema.insert(0, 'construct_family', family)
        family_schema_rows.append(target_schema)
        permutation_results = vectorized_family_permutation_test(target=target, labels_by_model=labels_by_model, model_k=model_k, permutation_bank=permutation_bank, chunk_size=permutation_chunk)
        permutation_results['construct_family'] = family
        permutation_results['held_out_variables'] = ','.join(held_out_variables)
        permutation_results['retained_mca_variables'] = ','.join(retained_variables)
        permutation_results['blind_mca_q'] = blind_fit.q
        permutation_results['blind_mca_j'] = blind_fit.j
        permutation_results['blind_mca_dmax'] = blind_fit.dmax
        permutation_results['permutation_contract'] = 'entire held-out family profile jointly permuted within participant'
        permutation_results = permutation_results.merge(pd.DataFrame(similarity_rows), on='model_id', how='left', validate='one_to_one')
        family_rows.append(permutation_results)
        variable_rows.append(variable_alignment_descriptive(events, held_out_variables, labels_by_model, model_dimensions, model_k, family))
        print(f'[construct-blind] {family}: Q={blind_fit.q}, J={blind_fit.j}, dmax={blind_fit.dmax}, models={len(labels_by_model)}', flush=True)
    family_alignment = pd.concat(family_rows, ignore_index=True)
    variable_alignment = pd.concat(variable_rows, ignore_index=True)
    family_schema = pd.concat(family_schema_rows, ignore_index=True)
    if 'dimensions' not in family_alignment.columns:
        family_alignment['dimensions'] = family_alignment['model_id'].str.extract('D(\\d+)_').astype(int)
    if 'k' not in family_alignment.columns:
        family_alignment['k'] = family_alignment['model_id'].str.extract('_K(\\d+)').astype(int)
    family_alignment['bh_q_within_model_across_families'] = np.nan
    for _, indices in family_alignment.groupby('model_id').groups.items():
        family_alignment.loc[indices, 'bh_q_within_model_across_families'] = bh_adjust(family_alignment.loc[indices, 'permutation_p_pseudo_F'])
    family_alignment['bh_q_global_across_model_family_grid'] = bh_adjust(family_alignment['permutation_p_pseudo_F'])
    return (family_alignment, variable_alignment, family_schema)

def lopo_partition_stability(events: pd.DataFrame, participants: np.ndarray, full_partitions: dict[tuple[int, int], np.ndarray], seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    participant_values = sorted(np.unique(participants))
    for held_out in participant_values:
        retained_mask = participants != held_out
        train = events.loc[retained_mask].reset_index(drop=True)
        train_participants = train['participant_id_norm'].astype(str).str.strip().str.lower()
        fit = fit_mca(normalize_active(train, FULL_ACTIVE), FULL_ACTIVE, train_participants)
        if fit.dmax < max(D_VALUES):
            raise RuntimeError(f'LOPO fold {held_out}: only {fit.dmax} dimensions available.')
        for d in D_VALUES:
            for k in K_VALUES:
                fold_labels = fit_partition(fit.row_coordinates, d, k, seed)
                reference = full_partitions[d, k][retained_mask]
                rows.append({'held_out_participant': held_out, 'n_training_events': int(retained_mask.sum()), 'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'ari_vs_full_partition_on_retained_events': float(adjusted_rand_score(reference, fold_labels)), 'ami_vs_full_partition_on_retained_events': float(adjusted_mutual_info_score(reference, fold_labels)), 'mca_refit_within_fold': True, 'interpretation': 'participant-influence analysis; full MCA refit before D-prefix K-means'})
        print(f'[lopo] excluded {held_out}: n={len(train)}, dmax={fit.dmax}', flush=True)
    return pd.DataFrame(rows)

def neighboring_partition_tables(partitions: dict[tuple[int, int], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimensional_rows = []
    for d in range(min(D_VALUES), max(D_VALUES)):
        for k in K_VALUES:
            a = partitions[d, k]
            b = partitions[d + 1, k]
            dimensional_rows.append({'k': k, 'dimensions_a': d, 'dimensions_b': d + 1, 'model_a': model_id(d, k), 'model_b': model_id(d + 1, k), 'ari': float(adjusted_rand_score(a, b)), 'ami': float(adjusted_mutual_info_score(a, b))})
    k_rows = []
    for d in D_VALUES:
        for k in range(min(K_VALUES), max(K_VALUES)):
            a = partitions[d, k]
            b = partitions[d, k + 1]
            k_rows.append({'dimensions': d, 'k_a': k, 'k_b': k + 1, 'model_a': model_id(d, k), 'model_b': model_id(d, k + 1), 'ari': float(adjusted_rand_score(a, b)), 'ami': float(adjusted_mutual_info_score(a, b))})
    return (pd.DataFrame(dimensional_rows), pd.DataFrame(k_rows))

def aggregate_model_summary(full_grid: pd.DataFrame, family_alignment: pd.DataFrame, full_feature_associations: pd.DataFrame, participant_support: pd.DataFrame, lopo: pd.DataFrame, neighboring_d: pd.DataFrame, neighboring_k: pd.DataFrame, upstream_selected_k: pd.DataFrame | None) -> pd.DataFrame:
    family_agg = family_alignment.groupby(['model_id', 'dimensions', 'k'], as_index=False).agg(heldout_family_mean_R2=('observed_R2', 'mean'), heldout_family_median_R2=('observed_R2', 'median'), heldout_family_min_R2=('observed_R2', 'min'), heldout_family_mean_null_corrected_R2=('null_corrected_R2', 'mean'), heldout_family_median_null_corrected_R2=('null_corrected_R2', 'median'), heldout_family_min_null_corrected_R2=('null_corrected_R2', 'min'), heldout_family_count_within_model_q05=('bh_q_within_model_across_families', lambda x: int(np.sum(np.asarray(x) < 0.05))), heldout_family_count_global_q05=('bh_q_global_across_model_family_grid', lambda x: int(np.sum(np.asarray(x) < 0.05))), heldout_family_mean_partition_ari_full_vs_blind=('ari_full_vs_construct_blind', 'mean'), heldout_family_min_partition_ari_full_vs_blind=('ari_full_vs_construct_blind', 'min'))
    feature_agg = full_feature_associations.groupby(['model_id', 'dimensions', 'k'], as_index=False).agg(full_feature_median_bias_corrected_v=('bias_corrected_cramers_v', 'median'), full_feature_mean_bias_corrected_v=('bias_corrected_cramers_v', 'mean'), full_feature_count_q05_within_model=('bh_q_within_model', lambda x: int(np.sum(np.asarray(x) < 0.05))), full_feature_count_q05_global=('bh_q_global', lambda x: int(np.sum(np.asarray(x) < 0.05))))
    significant_families = []
    for (mid, d, k), block in full_feature_associations.groupby(['model_id', 'dimensions', 'k'], sort=False):
        significant = block[block['bh_q_within_model'] < 0.05]
        significant_families.append({'model_id': mid, 'dimensions': d, 'k': k, 'full_feature_significant_family_count': int(significant['construct_family'].nunique())})
    significant_families = pd.DataFrame(significant_families)
    lopo_agg = lopo.groupby(['model_id', 'dimensions', 'k'], as_index=False).agg(lopo_ari_mean=('ari_vs_full_partition_on_retained_events', 'mean'), lopo_ari_min=('ari_vs_full_partition_on_retained_events', 'min'), lopo_ami_mean=('ami_vs_full_partition_on_retained_events', 'mean'), lopo_ami_min=('ami_vs_full_partition_on_retained_events', 'min'))
    adjacent_d_rows = []
    for d in D_VALUES:
        for k in K_VALUES:
            values = neighboring_d.loc[((neighboring_d['dimensions_a'] == d) | (neighboring_d['dimensions_b'] == d)) & (neighboring_d['k'] == k)]
            adjacent_d_rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'adjacent_D_ari_mean': float(values['ari'].mean()), 'adjacent_D_ari_min': float(values['ari'].min()), 'adjacent_D_ami_mean': float(values['ami'].mean()), 'adjacent_D_ami_min': float(values['ami'].min())})
    adjacent_d = pd.DataFrame(adjacent_d_rows)
    adjacent_k_rows = []
    for d in D_VALUES:
        for k in K_VALUES:
            values = neighboring_k.loc[(neighboring_k['dimensions'] == d) & ((neighboring_k['k_a'] == k) | (neighboring_k['k_b'] == k))]
            adjacent_k_rows.append({'model_id': model_id(d, k), 'dimensions': d, 'k': k, 'adjacent_K_ari_mean': float(values['ari'].mean()), 'adjacent_K_ami_mean': float(values['ami'].mean())})
    adjacent_k = pd.DataFrame(adjacent_k_rows)
    summary = full_grid.merge(family_agg, on=['model_id', 'dimensions', 'k'], validate='one_to_one').merge(feature_agg, on=['model_id', 'dimensions', 'k'], validate='one_to_one').merge(significant_families, on=['model_id', 'dimensions', 'k'], validate='one_to_one').merge(lopo_agg, on=['model_id', 'dimensions', 'k'], validate='one_to_one').merge(adjacent_d, on=['model_id', 'dimensions', 'k'], validate='one_to_one').merge(adjacent_k, on=['model_id', 'dimensions', 'k'], validate='one_to_one')
    del participant_support
    summary['eligible_min_cluster_fraction_005'] = summary['min_cluster_fraction'] >= MIN_CLUSTER_FRACTION
    summary['eligible_min_two_participants_per_cluster'] = summary['min_participant_support_per_cluster'] >= MIN_PARTICIPANT_SUPPORT_PER_CLUSTER
    summary['macro_interpretability_eligible'] = summary['eligible_min_cluster_fraction_005'] & summary['eligible_min_two_participants_per_cluster']
    summary['historical_D5_K4'] = (summary['dimensions'] == HISTORICAL_MODEL[0]) & (summary['k'] == HISTORICAL_MODEL[1])
    summary['candidate_D4_K4'] = (summary['dimensions'] == D4K4_MODEL[0]) & (summary['k'] == D4K4_MODEL[1])
    summary['full_sample_agnostic_reference'] = (summary['dimensions'] == FULL_REFERENCE_MODEL[0]) & (summary['k'] == FULL_REFERENCE_MODEL[1])
    summary['participant_robust_agnostic_reference'] = (summary['dimensions'] == ROBUST_REFERENCE_MODEL[0]) & (summary['k'] == ROBUST_REFERENCE_MODEL[1])
    if upstream_selected_k is not None and (not upstream_selected_k.empty):
        reference = upstream_selected_k[['dimensions', 'k']].rename(columns={'k': 'upstream_wideK_selected_k'})
        summary = summary.merge(reference, on='dimensions', how='left', validate='many_to_one')
        summary['matches_upstream_wideK_selected_k'] = summary['k'] == summary['upstream_wideK_selected_k']
    return summary

def mark_pareto_nondominated(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output['pareto_nondominated'] = False
    eligible_indices = output.index[output['macro_interpretability_eligible']].tolist()
    for i in eligible_indices:
        row_i = output.loc[i, PARETO_METRICS].astype(float).to_numpy()
        if not np.isfinite(row_i).all():
            continue
        dominated = False
        for j in eligible_indices:
            if i == j:
                continue
            row_j = output.loc[j, PARETO_METRICS].astype(float).to_numpy()
            if not np.isfinite(row_j).all():
                continue
            weakly_better = np.all(row_j >= row_i - 1e-12)
            strictly_better = np.any(row_j > row_i + 1e-12)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            output.loc[i, 'pareto_nondominated'] = True
    return output

def d4_d5_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    wanted = summary[(summary['dimensions'] == 4) & (summary['k'] == 4) | (summary['dimensions'] == 5) & (summary['k'] == 4)].copy()
    columns = ['model_id', 'dimensions', 'k', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'bootstrap_stability_ari', 'min_cluster_fraction', 'min_participant_support_per_cluster', 'max_single_participant_fraction_any_cluster', 'heldout_family_mean_R2', 'heldout_family_mean_null_corrected_R2', 'heldout_family_count_within_model_q05', 'heldout_family_count_global_q05', 'heldout_family_mean_partition_ari_full_vs_blind', 'full_feature_median_bias_corrected_v', 'full_feature_count_q05_within_model', 'full_feature_significant_family_count', 'lopo_ari_mean', 'lopo_ari_min', 'adjacent_D_ari_mean', 'adjacent_D_ari_min', 'pareto_nondominated']
    wanted = wanted[columns].sort_values('dimensions')
    if len(wanted) == 2:
        d4 = wanted.iloc[0]
        d5 = wanted.iloc[1]
        delta = {'model_id': 'D5_MINUS_D4_AT_K4', 'dimensions': np.nan, 'k': 4}
        for column in columns:
            if column in {'model_id', 'dimensions', 'k', 'pareto_nondominated'}:
                continue
            if pd.api.types.is_numeric_dtype(wanted[column]):
                delta[column] = float(d5[column]) - float(d4[column]) if pd.notna(d5[column]) and pd.notna(d4[column]) else np.nan
        delta['pareto_nondominated'] = np.nan
        wanted = pd.concat([wanted, pd.DataFrame([delta])], ignore_index=True)
    return wanted

def build_report(summary: pd.DataFrame, family_alignment: pd.DataFrame, d4d5: pd.DataFrame, qa: pd.DataFrame) -> str:
    pareto = summary[summary['pareto_nondominated']].sort_values(['dimensions', 'k'])
    lines = ['# Construct-aligned macro grid — REFIT 256', '', '## Scientific role', '', 'This analysis evaluates a coarse, interpretable macro representation separately from MCA dimensionality retention. D13 remains the full-sample geometric reference and D10 remains the participant-robust geometric reference.', f'The upstream agnostic partitions are {model_id(*FULL_REFERENCE_MODEL)} and {model_id(*ROBUST_REFERENCE_MODEL)}. They are comparison references, not automatic macro-model selections.', '', 'No weighted interpretability score is computed and no final D/K winner is automatically declared.', '', '## Candidate grid', '', '- D = 2 through 13', '- K = 2 through 6', f'- minimum cluster-fraction eligibility = {MIN_CLUSTER_FRACTION:.2f}', f'- minimum participant support per cluster = {MIN_PARTICIPANT_SUPPORT_PER_CLUSTER}', '', '## Construct families', '']
    for family, variables in CONSTRUCT_FAMILIES.items():
        lines.append(f"- **{family}**: {', '.join(variables)}")
    lines.extend(['', 'Each family is held out in turn before MCA. The complete family profile is then permuted jointly within participant for inference.', '', '## Pareto-nondominated eligible models', ''])
    if pareto.empty:
        lines.append('- None. Review eligibility and QA.')
    else:
        for row in pareto.itertuples(index=False):
            lines.append(f'- {row.model_id}: silhouette={row.silhouette:.4f}; bootstrap ARI={row.bootstrap_stability_ari:.4f}; mean null-corrected held-out R²={row.heldout_family_mean_null_corrected_R2:.4f}; globally significant held-out families={int(row.heldout_family_count_global_q05)}/6; LOPO ARI={row.lopo_ari_mean:.4f}; adjacent-D ARI={row.adjacent_D_ari_mean:.4f}.')
    lines.extend(['', 'Pareto membership is a screening device, not a final selection rule. A model is nondominated only if no other eligible model is at least as good on all six prespecified Pareto criteria and strictly better on at least one.', '', '## D4/K4 versus D5/K4', ''])
    compare = d4d5[d4d5['model_id'].isin(['D4_K4', 'D5_K4'])]
    for row in compare.itertuples(index=False):
        lines.append(f'- {row.model_id}: silhouette={row.silhouette:.4f}; bootstrap ARI={row.bootstrap_stability_ari:.4f}; mean held-out null-corrected R²={row.heldout_family_mean_null_corrected_R2:.4f}; held-out families global q<.05={int(row.heldout_family_count_global_q05)}/6; descriptive feature families represented={int(row.full_feature_significant_family_count)}/6; LOPO ARI={row.lopo_ari_mean:.4f}; adjacent-D ARI={row.adjacent_D_ari_mean:.4f}; Pareto={bool(row.pareto_nondominated)}.')
    lines.extend(['', 'D4/K4 and D5/K4 receive no historical bonus. If they are practically indistinguishable across stability, held-out construct recovery, participant influence, and profile distinctiveness, the lower-dimensional representation is the parsimonious candidate. If D5 adds a reproducible construct regime or materially improves the prespecified evidence, that provides a repaired-corpus justification for retaining D5.', '', '## Held-out construct inference', ''])
    for family, block in family_alignment.groupby('construct_family', sort=False):
        best = block.sort_values(['null_corrected_R2', 'permutation_p_pseudo_F'], ascending=[False, True]).iloc[0]
        lines.append(f"- {family}: largest null-corrected R² at {best['model_id']} ({best['null_corrected_R2']:.4f}; p={best['permutation_p_pseudo_F']:.6f}; global q={best['bh_q_global_across_model_family_grid']:.6f}).")
    lines.extend(['', 'These family-specific maxima are descriptive grid results and must not be used individually to select D/K. The macro representation should reflect broad, stable multiconstruct organization rather than optimization for one construct family.', '', '## Interpretation constraints', '', '1. This analysis does not change the upstream D13/D10 MCA conclusions.', '2. Context is not privileged over the other construct families.', '3. Full-geometry feature associations are descriptive because those features participated in MCA construction.', '4. Held-out construct recovery is the independent construct-alignment branch.', '5. Transition, basin, and successor evidence are absent by design.', '6. Final macro selection should be made only after reviewing the complete Pareto set, D4/K4 versus D5/K4, neighboring-dimensional persistence, and cluster residual profiles.', '', '## QA', '', f"{int(qa['passed'].sum())}/{len(qa)} checks passed.", ''])
    return '\n'.join(lines)

def validate_refit_inputs(args, events, coordinates, participants, matrix):
    global FULL_REFERENCE_MODEL, ROBUST_REFERENCE_MODEL
    counts = pd.Series(participants).value_counts().sort_index().to_dict()
    if len(events) != EXPECTED_EVENTS or counts != EXPECTED_PARTICIPANT_COUNTS:
        raise RuntimeError(f'Expected 256 chronological events with counts {EXPECTED_PARTICIPANT_COUNTS}; got {len(events)}, {counts}.')
    if len(coordinates) != len(events):
        raise RuntimeError('Coordinate table must contain exactly the same event set.')
    if not np.isfinite(matrix).all():
        raise RuntimeError('Nonfinite full MCA coordinates.')
    upstream_dir = args.upstream_dir or args.full_coordinates.parent.parent
    paths = {'metadata': upstream_dir / 'RUN_METADATA.json', 'decision': upstream_dir / 'tables/decision_summary.json', 'qa': upstream_dir / 'tables/qa_checks.csv', 'coordinates': upstream_dir / 'tables/mca_full_54d_row_coordinates.csv', 'selected_k': upstream_dir / 'tables/kmeans_consensus_selected_k_by_dimension.csv'}
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f'Missing completed upstream input: {path}')
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    decision = json.loads(paths['decision'].read_text(encoding='utf-8'))
    input_hash = sha256_file(args.event_table)
    for name, record in [('metadata', metadata), ('decision', decision)]:
        if record.get('input_sha256') != input_hash:
            raise RuntimeError(f'Event workbook does not match upstream {name}.')
        if record.get('dataset_version') != DATASET_VERSION:
            raise RuntimeError(f'Upstream {name} is not the chronological 256-event refit.')
    if metadata.get('input_sheet') != args.event_sheet:
        raise RuntimeError('Event sheet must match the completed upstream run.')
    if metadata.get('active_variables') != FULL_ACTIVE:
        raise RuntimeError('Upstream active variables differ from the macro-grid schema.')
    for key, expected in {'n_events': EXPECTED_EVENTS, 'active_variables_q': len(FULL_ACTIVE), 'indicator_categories_j': EXPECTED_FULL_LEVELS, 'maximum_dimensions': EXPECTED_FULL_DMAX, 'seed_full': args.seed}.items():
        if decision.get(key) != expected:
            raise RuntimeError(f'Upstream {key} must be {expected}.')
    upstream_qa = pd.read_csv(paths['qa'])
    ensure_columns(upstream_qa, ['passed'], 'Upstream QA')
    if upstream_qa.empty or not upstream_qa['passed'].eq(True).all() or decision.get('qa_passed') != len(upstream_qa) or (decision.get('qa_total') != len(upstream_qa)):
        raise RuntimeError('Upstream QA is incomplete or inconsistent.')
    for supplied, key in [(args.full_coordinates, 'coordinates'), (args.upstream_selected_k, 'selected_k')]:
        if sha256_file(supplied) != sha256_file(paths[key]):
            raise RuntimeError(f'Supplied {key} differs from the completed upstream run.')
    full_d = int(decision['full_sample_parallel_retained_dimensions'])
    robust_d = int(decision['participant_robust_min_lopo_dimensions'])
    if (full_d, robust_d) != (13, 10):
        raise RuntimeError('This inspected REFIT grid expects upstream D13/D10. Review the search range before using different retention results.')
    selected = pd.read_csv(args.upstream_selected_k)
    ensure_columns(selected, ['dimensions', 'k'], 'Upstream selected K')
    if selected['dimensions'].duplicated().any():
        raise RuntimeError('Duplicate dimensions in upstream selected-K table.')
    selected = selected.set_index('dimensions')
    if not set(D_VALUES).issubset(selected.index):
        raise RuntimeError('Upstream selected-K table does not cover D2 through D13.')
    k_values = pd.to_numeric(selected.loc[D_VALUES, 'k'], errors='raise').to_numpy(float)
    if not (np.isfinite(k_values).all() and np.equal(k_values, np.floor(k_values)).all() and ((k_values >= 2) & (k_values <= 19)).all()):
        raise RuntimeError('Upstream selected K values must be integers in 2..19.')
    FULL_REFERENCE_MODEL = (full_d, int(selected.loc[full_d, 'k']))
    ROBUST_REFERENCE_MODEL = (robust_d, int(selected.loc[robust_d, 'k']))
    fit = fit_mca(normalize_active(events, FULL_ACTIVE), FULL_ACTIVE, pd.Series(participants))
    if fit.j != EXPECTED_FULL_LEVELS or fit.dmax != EXPECTED_FULL_DMAX:
        raise RuntimeError('Refitted active schema does not match upstream dimensions.')
    max_gram_error = 0.0
    for d in D_VALUES + [EXPECTED_FULL_DMAX]:
        supplied_gram = matrix[:, :d] @ matrix[:, :d].T
        refit_gram = fit.row_coordinates[:, :d] @ fit.row_coordinates[:, :d].T
        max_gram_error = max(max_gram_error, float(np.max(abs(supplied_gram - refit_gram))))
        if not np.allclose(supplied_gram, refit_gram, rtol=1e-08, atol=1e-09):
            raise RuntimeError(f'Supplied D{d} geometry does not match this event workbook.')
    print(f'[OK] Matched 256-event upstream input; agnostic references {model_id(*FULL_REFERENCE_MODEL)}, {model_id(*ROBUST_REFERENCE_MODEL)}', flush=True)
    return {'directory': str(upstream_dir.resolve()), 'files': {name: {'path': str(path.resolve()), 'sha256': sha256_file(path)} for name, path in paths.items()}, 'input_sha256': input_hash, 'coordinate_prefix_gram_max_absolute_error': max_gram_error, 'coordinate_prefixes_checked': D_VALUES + [EXPECTED_FULL_DMAX]}

def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / 'tables'
    coordinates_dir = args.out_dir / 'coordinates'
    spectra_dir = args.out_dir / 'spectra'
    tables_dir.mkdir(parents=True, exist_ok=True)
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    events = read_table(args.event_table, args.event_sheet).reset_index(drop=True)
    coordinates = read_table(args.full_coordinates).reset_index(drop=True)
    ensure_columns(events, KEY + FULL_ACTIVE, args.event_table.name)
    ensure_columns(coordinates, KEY + [f'Dim{i}' for i in range(EXPECTED_FULL_DMAX)], args.full_coordinates.name)
    if events.duplicated(KEY).any():
        raise RuntimeError('Duplicate event keys in event table.')
    if coordinates.duplicated(KEY).any():
        raise RuntimeError('Duplicate event keys in full coordinate table.')
    merged = events.merge(coordinates[KEY + [f'Dim{i}' for i in range(EXPECTED_FULL_DMAX)]], on=KEY, how='left', validate='one_to_one', indicator=True)
    if not (merged['_merge'] == 'both').all():
        bad = merged.loc[merged['_merge'] != 'both', KEY + ['_merge']]
        raise RuntimeError('Coordinate merge incomplete:\n' + bad.to_string(index=False))
    merged = merged.drop(columns='_merge')
    participants = merged['participant_id_norm'].astype(str).str.strip().str.lower().to_numpy()
    observed_participant_counts = pd.Series(participants).value_counts().sort_index().to_dict()
    full_coordinate_matrix = merged[[f'Dim{i}' for i in range(EXPECTED_FULL_DMAX)]].apply(pd.to_numeric, errors='raise').to_numpy(float)
    upstream_provenance = validate_refit_inputs(args, merged, coordinates, participants, full_coordinate_matrix)
    family_variables = [variable for variables in CONSTRUCT_FAMILIES.values() for variable in variables]
    if sorted(family_variables) != sorted(FULL_ACTIVE):
        raise RuntimeError('Construct families do not cover the 11 active variables exactly once.')
    if len(family_variables) != len(set(family_variables)):
        raise RuntimeError('At least one active variable appears in multiple families.')
    print('[stage] Full macro grid', flush=True)
    full_grid, full_partitions, participant_support = build_full_macro_grid(full_coordinate_matrix, participants, seed=args.seed, n_bootstrap=args.n_bootstrap)
    print('[stage] Full-geometry multifeature characterization', flush=True)
    feature_associations, feature_cells = characterize_full_features(merged, full_partitions)
    print('[stage] Common participant-block permutation bank', flush=True)
    permutation_bank = make_within_participant_row_permutation_bank(participants, n_permutations=args.n_perm, seed=args.seed)
    print('[stage] Construct-blind held-out recovery', flush=True)
    family_alignment, variable_alignment, family_schema = evaluate_construct_blind_families(events=merged, participants=participants, full_partitions=full_partitions, permutation_bank=permutation_bank, output_coordinates=coordinates_dir, output_spectra=spectra_dir, n_permutations=args.n_perm, permutation_chunk=args.permutation_chunk, seed=args.seed)
    print('[stage] Proper LOPO participant influence', flush=True)
    lopo = lopo_partition_stability(merged, participants, full_partitions, seed=args.seed)
    print('[stage] Neighboring-model persistence', flush=True)
    neighboring_d, neighboring_k = neighboring_partition_tables(full_partitions)
    upstream_reference = None
    if args.upstream_selected_k is not None:
        upstream = read_table(args.upstream_selected_k)
        ensure_columns(upstream, ['dimensions', 'k'], args.upstream_selected_k.name)
        upstream = upstream[upstream['dimensions'].isin(D_VALUES)][['dimensions', 'k']].copy()
        if upstream['dimensions'].duplicated().any():
            raise RuntimeError('Upstream selected-K table has duplicate dimensional rows.')
        upstream_reference = pd.DataFrame({'dimensions': D_VALUES}).merge(upstream, on='dimensions', how='left')
        if upstream_reference['k'].isna().any():
            missing_d = upstream_reference.loc[upstream_reference['k'].isna(), 'dimensions'].tolist()
            raise RuntimeError(f'Upstream selected-K table missing D values: {missing_d}')
    print('[stage] Model summary / Pareto screen', flush=True)
    summary = aggregate_model_summary(full_grid=full_grid, family_alignment=family_alignment, full_feature_associations=feature_associations, participant_support=participant_support, lopo=lopo, neighboring_d=neighboring_d, neighboring_k=neighboring_k, upstream_selected_k=upstream_reference)
    summary = mark_pareto_nondominated(summary)
    pareto = summary[summary['pareto_nondominated']].copy()
    d4d5 = d4_d5_comparison(summary)
    qa_rows: list[dict] = [{'check': 'refit upstream identity and D-prefix geometry', 'observed': 'matched', 'expected': 'matched', 'passed': True}]

    def qa(check: str, observed, expected, passed: bool) -> None:
        qa_rows.append({'check': check, 'observed': observed, 'expected': expected, 'passed': bool(passed)})
    qa('event rows', len(merged), EXPECTED_EVENTS, len(merged) == EXPECTED_EVENTS)
    qa('participant count', len(np.unique(participants)), EXPECTED_PARTICIPANTS, len(np.unique(participants)) == EXPECTED_PARTICIPANTS)
    qa('participant event counts', json.dumps(observed_participant_counts, sort_keys=True), json.dumps(EXPECTED_PARTICIPANT_COUNTS, sort_keys=True), observed_participant_counts == EXPECTED_PARTICIPANT_COUNTS)
    qa('full active variables', len(FULL_ACTIVE), 11, len(FULL_ACTIVE) == 11)
    qa('construct families', len(CONSTRUCT_FAMILIES), 6, len(CONSTRUCT_FAMILIES) == 6)
    qa('construct family schema exact coverage', len(family_variables), len(FULL_ACTIVE), sorted(family_variables) == sorted(FULL_ACTIVE) and len(family_variables) == len(set(family_variables)))
    qa('full coordinate dimensions', full_coordinate_matrix.shape[1], EXPECTED_FULL_DMAX, full_coordinate_matrix.shape[1] == EXPECTED_FULL_DMAX)
    qa('full macro models', len(full_grid), len(D_VALUES) * len(K_VALUES), len(full_grid) == len(D_VALUES) * len(K_VALUES))
    qa('construct-blind model-family tests', len(family_alignment), len(D_VALUES) * len(K_VALUES) * len(CONSTRUCT_FAMILIES), len(family_alignment) == len(D_VALUES) * len(K_VALUES) * len(CONSTRUCT_FAMILIES))
    qa('held-out variable descriptive rows', len(variable_alignment), len(D_VALUES) * len(K_VALUES) * len(FULL_ACTIVE), len(variable_alignment) == len(D_VALUES) * len(K_VALUES) * len(FULL_ACTIVE))
    qa('full feature association rows', len(feature_associations), len(D_VALUES) * len(K_VALUES) * len(FULL_ACTIVE), len(feature_associations) == len(D_VALUES) * len(K_VALUES) * len(FULL_ACTIVE))
    qa('proper LOPO rows', len(lopo), EXPECTED_PARTICIPANTS * len(D_VALUES) * len(K_VALUES), len(lopo) == EXPECTED_PARTICIPANTS * len(D_VALUES) * len(K_VALUES))
    qa('adjacent-dimensionality comparisons', len(neighboring_d), (len(D_VALUES) - 1) * len(K_VALUES), len(neighboring_d) == (len(D_VALUES) - 1) * len(K_VALUES))
    qa('adjacent-K comparisons', len(neighboring_k), len(D_VALUES) * (len(K_VALUES) - 1), len(neighboring_k) == len(D_VALUES) * (len(K_VALUES) - 1))
    qa('participant-block permutations', len(permutation_bank), args.n_perm, len(permutation_bank) == args.n_perm)
    qa('permutation p values complete', int(family_alignment['permutation_p_pseudo_F'].notna().sum()), len(family_alignment), family_alignment['permutation_p_pseudo_F'].notna().all())
    qa('D4/K4 and D5/K4 both present', int(summary['model_id'].isin(['D4_K4', 'D5_K4']).sum()), 2, summary['model_id'].isin(['D4_K4', 'D5_K4']).sum() == 2)
    qa_table = pd.DataFrame(qa_rows)
    if not qa_table['passed'].all():
        failed = qa_table.loc[~qa_table['passed']]
        raise RuntimeError('Construct-aligned macro-grid QA failed:\n' + failed.to_string(index=False))
    full_grid.to_csv(tables_dir / 'full_geometry_macro_grid.csv', index=False)
    family_alignment.to_csv(tables_dir / 'heldout_construct_family_alignment.csv', index=False)
    variable_alignment.to_csv(tables_dir / 'heldout_construct_variable_alignment.csv', index=False)
    feature_associations.to_csv(tables_dir / 'full_geometry_feature_associations.csv', index=False)
    feature_cells.to_csv(tables_dir / 'full_geometry_cluster_feature_cells.csv', index=False)
    participant_support.to_csv(tables_dir / 'participant_support_by_model.csv', index=False)
    lopo.to_csv(tables_dir / 'lopo_partition_stability.csv', index=False)
    neighboring_d.to_csv(tables_dir / 'neighboring_dimension_partition_similarity.csv', index=False)
    neighboring_k.to_csv(tables_dir / 'neighboring_k_partition_similarity.csv', index=False)
    summary.to_csv(tables_dir / 'macro_model_summary.csv', index=False)
    pareto.to_csv(tables_dir / 'pareto_nondominated_models.csv', index=False)
    d4d5.to_csv(tables_dir / 'd4_k4_vs_d5_k4_comparison.csv', index=False)
    family_schema.to_csv(tables_dir / 'construct_family_schema.csv', index=False)
    if upstream_reference is not None:
        upstream_reference.to_csv(tables_dir / 'upstream_selected_k_reference.csv', index=False)
    qa_table.to_csv(tables_dir / 'qa_checks.csv', index=False)
    report = build_report(summary=summary, family_alignment=family_alignment, d4d5=d4d5, qa=qa_table)
    (args.out_dir / 'MACRO_INTERPRETABILITY_REPORT.md').write_text(report, encoding='utf-8')
    metadata = {'analysis': 'construct-aligned macro grid', 'dataset_version': DATASET_VERSION, 'n_events': len(merged), 'participant_event_counts': observed_participant_counts, 'input_sheet': args.event_sheet, 'script_sha256': sha256_file(Path(__file__)), 'upstream_reference': upstream_provenance, 'full_sample_agnostic_reference': model_id(*FULL_REFERENCE_MODEL), 'participant_robust_agnostic_reference': model_id(*ROBUST_REFERENCE_MODEL), 'qa_passed': int(qa_table['passed'].sum()), 'qa_total': len(qa_table), 'command': ' '.join(sys.argv), 'elapsed_seconds': time.time() - started, 'input_event_table': str(args.event_table.resolve()), 'input_event_table_sha256': sha256_file(args.event_table), 'input_full_coordinates': str(args.full_coordinates.resolve()), 'input_full_coordinates_sha256': sha256_file(args.full_coordinates), 'upstream_selected_k': str(args.upstream_selected_k.resolve()) if args.upstream_selected_k is not None else None, 'upstream_selected_k_sha256': sha256_file(args.upstream_selected_k) if args.upstream_selected_k is not None else None, 'dimensions_evaluated': D_VALUES, 'k_evaluated': K_VALUES, 'full_active_variables': FULL_ACTIVE, 'construct_families': CONSTRUCT_FAMILIES, 'historical_model': 'D5_K4', 'explicit_neighbor_candidate': 'D4_K4', 'full_sample_geometry_reference': 'D13', 'participant_robust_geometry_reference': 'D10', 'n_permutations': args.n_perm, 'permutation_scheme': 'common participant-block row-permutation bank; each held-out construct family profile permuted jointly within participant', 'n_bootstrap': args.n_bootstrap, 'seed': args.seed, 'macro_cluster_support_rule': {'min_cluster_fraction': MIN_CLUSTER_FRACTION, 'min_participants_per_cluster': MIN_PARTICIPANT_SUPPORT_PER_CLUSTER}, 'pareto_metrics': PARETO_METRICS, 'weighted_interpretability_score': False, 'automatic_final_model_selection': False, 'context_privileged': False, 'projection_analysis_run': False, 'basin_analysis_run': False, 'transition_analysis_run': False, 'successor_analysis_run': False, 'software': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn_version}}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('\n[OK] Construct-aligned macro-grid analysis complete.')
    print(f'[OK] Events: {len(merged)} / {EXPECTED_EVENTS}')
    print(f'[OK] Participants: {len(np.unique(participants))} / {EXPECTED_PARTICIPANTS}')
    print(f'[OK] Full active variables: {len(FULL_ACTIVE)} / 11')
    print(f'[OK] Construct families: {len(CONSTRUCT_FAMILIES)} / 6')
    print(f'[OK] Full macro models: {len(full_grid)} / {len(D_VALUES) * len(K_VALUES)}')
    print(f'[OK] Construct-blind model-family tests: {len(family_alignment)} / {len(D_VALUES) * len(K_VALUES) * len(CONSTRUCT_FAMILIES)}')
    print(f'[OK] Proper LOPO model fits: {len(lopo)} / {EXPECTED_PARTICIPANTS * len(D_VALUES) * len(K_VALUES)}')
    print(f'[OK] Participant-block permutations: {len(permutation_bank)}')
    print('[OK] No projection, basin, transition, or SR analysis was run.')
    print(f"[OK] QA: {int(qa_table['passed'].sum())}/{len(qa_table)} passed.")
    print(f'[OK] Output: {args.out_dir.resolve()}')
    print('\n[RESULT] Pareto-nondominated eligible macro models:')
    if pareto.empty:
        print('  None')
    else:
        for row in pareto.sort_values(['dimensions', 'k']).itertuples(index=False):
            print(f'  {row.model_id}: sil={row.silhouette:.4f}; bootARI={row.bootstrap_stability_ari:.4f}; heldoutΔR2={row.heldout_family_mean_null_corrected_R2:.4f}; families_q05={int(row.heldout_family_count_global_q05)}/6; LOPO_ARI={row.lopo_ari_mean:.4f}; adjD_ARI={row.adjacent_D_ari_mean:.4f}')
    print('\n[RESULT] D4/K4 vs D5/K4:')
    compare = d4d5[d4d5['model_id'].isin(['D4_K4', 'D5_K4'])]
    print(compare[['model_id', 'silhouette', 'bootstrap_stability_ari', 'heldout_family_mean_null_corrected_R2', 'heldout_family_count_global_q05', 'lopo_ari_mean', 'adjacent_D_ari_mean', 'pareto_nondominated']].to_string(index=False))
if __name__ == '__main__':
    main()
