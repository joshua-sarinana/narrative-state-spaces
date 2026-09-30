"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
KEY = ['participant_id_norm', 'hippo_id_norm']
D_VALUES = list(range(2, 14))
K_VALUES = list(range(2, 7))
EXPECTED_EVENTS = 256
EXPECTED_PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
DATASET_VERSION = 'chronological_256_v2'
EXPECTED_MODELS = len(D_VALUES) * len(K_VALUES)
EXPECTED_ADJACENT_D = (len(D_VALUES) - 1) * len(K_VALUES)
EXPECTED_ADJACENT_K = len(D_VALUES) * (len(K_VALUES) - 1)
EXPECTED_PROFILE_CELLS = 12 * sum(K_VALUES) * 65
FULL_ACTIVE = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
CONSTRUCT_FAMILIES = {'context': ['context_type_primary', 'context_type_secondary'], 'hippocampal_mechanism': ['primary_hipp_mech', 'secondary_hipp_mech'], 'anchor_landmark': ['anchor_type', 'landmark'], 'representational_organization': ['frame', 'orient', 'perspective'], 'semantic_content': ['event_meta_superordinate_theme_primary'], 'affective_content': ['event_sentiment_primary']}
VARIABLE_TO_FAMILY = {variable: family for family, variables in CONSTRUCT_FAMILIES.items() for variable in variables}
FOCAL_MODEL_REASONS = {'D2_K2': 'low-complexity two-regime candidate', 'D3_K3': 'low-complexity three-regime candidate', 'D4_K3': 'direct K3 parent for D4/K4 interpretive-resolution audit', 'D4_K4': 'previously selected D4/K4, reassessed as a candidate', 'D5_K4': 'historical repaired D5/K4 comparison', 'D6_K4': 'D6 four-regime dimensional-persistence comparison', 'D7_K4': 'D7 four-regime dimensional-persistence comparison', 'D8_K4': 'higher-D K4 sensitivity / continuation of K4 path'}
LOW_COMPLEXITY_PAIRS = [('D2_K2', 'D3_K3', 'low-complexity K2 -> K3 representative comparison'), ('D3_K3', 'D4_K4', 'low-complexity K3 -> K4 representative comparison'), ('D3_K3', 'D6_K4', 'K3 versus D6/K4 dimensional comparison'), ('D4_K4', 'D5_K4', 'neighboring four-regime comparison'), ('D4_K4', 'D6_K4', 'D4/K4 versus D6/K4 dimensional comparison'), ('D6_K4', 'D7_K4', 'D6/K4 versus D7/K4 dimensional persistence')]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-coordinates', required=True, type=Path)
    parser.add_argument('--macro-tables-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        parser.error('Canonical post-processing execution requires --seed 42.')
    return args

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f'Required input does not exist: {path}')
    return pd.read_csv(path, low_memory=False)

def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f'{label} missing required columns: {missing}')

def model_id(d: int, k: int) -> str:
    return f'D{d}_K{k}'

def parse_model_id(mid: str) -> tuple[int, int]:
    try:
        d_text, k_text = str(mid).split('_')
        return (int(d_text[1:]), int(k_text[1:]))
    except Exception as exc:
        raise RuntimeError(f'Cannot parse model_id={mid!r}') from exc

def reconstruct_partitions(coordinates: np.ndarray, seed: int) -> dict[tuple[int, int], np.ndarray]:
    partitions: dict[tuple[int, int], np.ndarray] = {}
    for d in D_VALUES:
        for k in K_VALUES:
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coordinates[:, :d])
            partitions[d, k] = labels.astype(np.int16)
        print(f'[reconstruct] completed D={d}', flush=True)
    return partitions

def membership_table(keys: pd.DataFrame, partitions: dict[tuple[int, int], np.ndarray]) -> pd.DataFrame:
    out = keys.copy()
    for d in D_VALUES:
        for k in K_VALUES:
            out[model_id(d, k)] = partitions[d, k] + 1
    return out

def profile_keys(cells: pd.DataFrame) -> list[tuple[str, str]]:
    keys = cells[['variable', 'level']].drop_duplicates().sort_values(['variable', 'level'])
    return list(keys.itertuples(index=False, name=None))

def model_profile_matrix(cells: pd.DataFrame, mid: str, expected_k: int, ordered_keys: list[tuple[str, str]]) -> tuple[np.ndarray, list[int]]:
    block = cells[cells['model_id'] == mid].copy()
    if block.empty:
        raise RuntimeError(f'No residual-profile cells found for {mid}.')
    clusters = sorted(block['cluster'].astype(int).unique().tolist())
    if clusters != list(range(1, expected_k + 1)):
        raise RuntimeError(f'{mid}: unexpected cluster labels {clusters}; expected 1..{expected_k}.')
    lookup = {(int(row.cluster), str(row.variable), str(row.level)): float(row.adjusted_standardized_residual) for row in block.itertuples(index=False)}
    matrix = np.empty((expected_k, len(ordered_keys)), dtype=float)
    for cluster_index, cluster in enumerate(clusters):
        for feature_index, (variable, level) in enumerate(ordered_keys):
            key = (cluster, variable, level)
            if key not in lookup:
                raise RuntimeError(f'{mid}: missing residual cell cluster={cluster}, variable={variable}, level={level}.')
            matrix[cluster_index, feature_index] = lookup[key]
    return (matrix, clusters)

def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float(np.sum(a_centered ** 2)) * float(np.sum(b_centered ** 2)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(a_centered * b_centered) / denominator)

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = math.sqrt(float(np.sum(a ** 2)) * float(np.sum(b ** 2)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(a * b) / denominator)

def pairwise_profile_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.empty((matrix_a.shape[0], matrix_b.shape[0]), dtype=float)
    cosine = np.empty_like(pearson)
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[0]):
            pearson[i, j] = safe_pearson(matrix_a[i], matrix_b[j])
            cosine[i, j] = safe_cosine(matrix_a[i], matrix_b[j])
    return (pearson, cosine)

def hungarian_maximize(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(similarity, dtype=float).copy()
    score[~np.isfinite(score)] = -1000000000.0
    row_index, column_index = linear_sum_assignment(-score)
    return (row_index, column_index)

def contingency_labels(labels_a: np.ndarray, labels_b: np.ndarray, k_a: int, k_b: int) -> np.ndarray:
    table = np.zeros((k_a, k_b), dtype=int)
    np.add.at(table, (labels_a, labels_b), 1)
    return table

def jaccard_for_clusters(labels_a: np.ndarray, cluster_a: int, labels_b: np.ndarray, cluster_b: int) -> float:
    a = labels_a == cluster_a
    b = labels_b == cluster_b
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return float(intersection / union) if union else np.nan

def direct_partition_similarity(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, float]:
    return (float(adjusted_rand_score(labels_a, labels_b)), float(adjusted_mutual_info_score(labels_a, labels_b)))

def crosscheck_neighboring_d(archived: pd.DataFrame, partitions: dict[tuple[int, int], np.ndarray], tolerance: float=1e-12) -> pd.DataFrame:
    ensure_columns(archived, ['k', 'dimensions_a', 'dimensions_b', 'model_a', 'model_b', 'ari', 'ami'], 'neighboring_dimension_partition_similarity.csv')
    rows = []
    for row in archived.itertuples(index=False):
        d_a = int(row.dimensions_a)
        d_b = int(row.dimensions_b)
        k = int(row.k)
        ari, ami = direct_partition_similarity(partitions[d_a, k], partitions[d_b, k])
        rows.append({'model_a': row.model_a, 'model_b': row.model_b, 'archived_ari': float(row.ari), 'reconstructed_ari': ari, 'abs_diff_ari': abs(ari - float(row.ari)), 'archived_ami': float(row.ami), 'reconstructed_ami': ami, 'abs_diff_ami': abs(ami - float(row.ami)), 'passed': abs(ari - float(row.ari)) <= tolerance and abs(ami - float(row.ami)) <= tolerance})
    return pd.DataFrame(rows)

def crosscheck_neighboring_k(archived: pd.DataFrame, partitions: dict[tuple[int, int], np.ndarray], tolerance: float=1e-12) -> pd.DataFrame:
    ensure_columns(archived, ['dimensions', 'k_a', 'k_b', 'model_a', 'model_b', 'ari', 'ami'], 'neighboring_k_partition_similarity.csv')
    rows = []
    for row in archived.itertuples(index=False):
        d = int(row.dimensions)
        k_a = int(row.k_a)
        k_b = int(row.k_b)
        ari, ami = direct_partition_similarity(partitions[d, k_a], partitions[d, k_b])
        rows.append({'model_a': row.model_a, 'model_b': row.model_b, 'archived_ari': float(row.ari), 'reconstructed_ari': ari, 'abs_diff_ari': abs(ari - float(row.ari)), 'archived_ami': float(row.ami), 'reconstructed_ami': ami, 'abs_diff_ami': abs(ami - float(row.ami)), 'passed': abs(ari - float(row.ari)) <= tolerance and abs(ami - float(row.ami)) <= tolerance})
    return pd.DataFrame(rows)

def same_k_dimension_persistence(partitions: dict[tuple[int, int], np.ndarray], cells: pd.DataFrame, ordered_profile_keys: list[tuple[str, str]], archived_neighboring_d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    matched_rows = []
    archived_lookup = {(int(row.dimensions_a), int(row.dimensions_b), int(row.k)): (float(row.ari), float(row.ami)) for row in archived_neighboring_d.itertuples(index=False)}
    for d in range(min(D_VALUES), max(D_VALUES)):
        for k in K_VALUES:
            mid_a = model_id(d, k)
            mid_b = model_id(d + 1, k)
            labels_a = partitions[d, k]
            labels_b = partitions[d + 1, k]
            overlap = contingency_labels(labels_a, labels_b, k, k)
            event_rows, event_cols = linear_sum_assignment(-overlap)
            event_mapping = {int(a): int(b) for a, b in zip(event_rows, event_cols)}
            event_match_count = int(sum((overlap[a, b] for a, b in zip(event_rows, event_cols))))
            profile_a, clusters_a = model_profile_matrix(cells, mid_a, k, ordered_profile_keys)
            profile_b, clusters_b = model_profile_matrix(cells, mid_b, k, ordered_profile_keys)
            pearson, cosine = pairwise_profile_similarity(profile_a, profile_b)
            profile_rows, profile_cols = hungarian_maximize(pearson)
            profile_mapping = {int(a): int(b) for a, b in zip(profile_rows, profile_cols)}
            profile_pearsons = []
            profile_cosines = []
            mapping_agreements = 0
            for a_index in range(k):
                b_profile = profile_mapping[a_index]
                b_event = event_mapping[a_index]
                if b_profile == b_event:
                    mapping_agreements += 1
                count = int(overlap[a_index, b_profile])
                size_a = int(np.sum(labels_a == a_index))
                size_b = int(np.sum(labels_b == b_profile))
                union = size_a + size_b - count
                jaccard = float(count / union) if union else np.nan
                event_fraction_a = float(count / size_a) if size_a else np.nan
                event_fraction_b = float(count / size_b) if size_b else np.nan
                p = float(pearson[a_index, b_profile])
                c = float(cosine[a_index, b_profile])
                profile_pearsons.append(p)
                profile_cosines.append(c)
                matched_rows.append({'k': k, 'dimensions_a': d, 'dimensions_b': d + 1, 'model_a': mid_a, 'model_b': mid_b, 'cluster_a': clusters_a[a_index], 'profile_matched_cluster_b': clusters_b[b_profile], 'event_matched_cluster_b': clusters_b[b_event], 'profile_event_matching_agree': b_profile == b_event, 'profile_pearson': p, 'profile_cosine': c, 'event_overlap_count_for_profile_match': count, 'cluster_a_size': size_a, 'cluster_b_size': size_b, 'fraction_cluster_a_retained_in_profile_match': event_fraction_a, 'fraction_cluster_b_from_profile_match': event_fraction_b, 'jaccard_for_profile_match': jaccard})
            archived_ari, archived_ami = archived_lookup[d, d + 1, k]
            summary_rows.append({'k': k, 'dimensions_a': d, 'dimensions_b': d + 1, 'model_a': mid_a, 'model_b': mid_b, 'ari': archived_ari, 'ami': archived_ami, 'hungarian_event_match_fraction': event_match_count / len(labels_a), 'profile_pearson_mean': float(np.nanmean(profile_pearsons)), 'profile_pearson_min': float(np.nanmin(profile_pearsons)), 'profile_cosine_mean': float(np.nanmean(profile_cosines)), 'profile_cosine_min': float(np.nanmin(profile_cosines)), 'profile_event_mapping_agreement_fraction': mapping_agreements / k})
    return (pd.DataFrame(summary_rows), pd.DataFrame(matched_rows))

def family_profile_difference(vector_a: np.ndarray, vector_b: np.ndarray, ordered_keys: list[tuple[str, str]]) -> list[dict]:
    by_family: dict[str, list[int]] = {family: [] for family in CONSTRUCT_FAMILIES}
    for index, (variable, _level) in enumerate(ordered_keys):
        family = VARIABLE_TO_FAMILY[variable]
        by_family[family].append(index)
    rows = []
    for family, indices in by_family.items():
        diff = vector_a[indices] - vector_b[indices]
        a = vector_a[indices]
        b = vector_b[indices]
        opposite_notable = (a >= 2.0) & (b <= -2.0) | (a <= -2.0) & (b >= 2.0)
        rows.append({'construct_family': family, 'rms_residual_difference': float(math.sqrt(np.mean(diff ** 2))), 'mean_absolute_residual_difference': float(np.mean(np.abs(diff))), 'max_absolute_residual_difference': float(np.max(np.abs(diff))), 'n_cells_abs_difference_ge_2': int(np.sum(np.abs(diff) >= 2.0)), 'n_cells_opposite_notable_sign': int(np.sum(opposite_notable))})
    return rows

def k_resolution_hierarchy(partitions: dict[tuple[int, int], np.ndarray], cells: pd.DataFrame, ordered_profile_keys: list[tuple[str, str]], neighboring_k: pd.DataFrame, participant_support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_rows = []
    profile_rows = []
    hierarchy_rows = []
    daughter_family_rows = []
    archived_lookup = {(int(row.dimensions), int(row.k_a), int(row.k_b)): (float(row.ari), float(row.ami)) for row in neighboring_k.itertuples(index=False)}
    support_lookup = {(str(row.model_id), int(row.cluster)): int(row.participants_represented) for row in participant_support.itertuples(index=False)}
    for d in D_VALUES:
        for parent_k in range(min(K_VALUES), max(K_VALUES)):
            child_k = parent_k + 1
            parent_mid = model_id(d, parent_k)
            child_mid = model_id(d, child_k)
            parent_labels = partitions[d, parent_k]
            child_labels = partitions[d, child_k]
            table = contingency_labels(parent_labels, child_labels, parent_k, child_k)
            parent_sizes = table.sum(axis=1)
            child_sizes = table.sum(axis=0)
            parent_profile, parent_clusters = model_profile_matrix(cells, parent_mid, parent_k, ordered_profile_keys)
            child_profile, child_clusters = model_profile_matrix(cells, child_mid, child_k, ordered_profile_keys)
            pearson, cosine = pairwise_profile_similarity(parent_profile, child_profile)
            for parent_index in range(parent_k):
                for child_index in range(child_k):
                    count = int(table[parent_index, child_index])
                    parent_fraction = float(count / parent_sizes[parent_index])
                    child_fraction = float(count / child_sizes[child_index])
                    split_rows.append({'dimensions': d, 'parent_k': parent_k, 'child_k': child_k, 'parent_model': parent_mid, 'child_model': child_mid, 'parent_cluster': parent_clusters[parent_index], 'child_cluster': child_clusters[child_index], 'event_count': count, 'parent_fraction_to_child': parent_fraction, 'child_fraction_from_parent': child_fraction})
                    profile_rows.append({'dimensions': d, 'parent_k': parent_k, 'child_k': child_k, 'parent_model': parent_mid, 'child_model': child_mid, 'parent_cluster': parent_clusters[parent_index], 'child_cluster': child_clusters[child_index], 'event_count': count, 'parent_fraction_to_child': parent_fraction, 'child_fraction_from_parent': child_fraction, 'profile_pearson': float(pearson[parent_index, child_index]), 'profile_cosine': float(cosine[parent_index, child_index])})
            child_dominant_fractions = table.max(axis=0) / child_sizes
            child_parent_purity = float(table.max(axis=0).sum() / table.sum())
            parent_dominant_fractions = table.max(axis=1) / parent_sizes
            parent_child_purity = float(table.max(axis=1).sum() / table.sum())
            parent_fraction_matrix = table / parent_sizes[:, None]
            sorted_child_order = np.argsort(parent_fraction_matrix, axis=1)[:, ::-1]
            second_largest = np.array([parent_fraction_matrix[parent_index, sorted_child_order[parent_index, 1]] for parent_index in range(parent_k)])
            split_parent_index = int(np.argmax(second_largest))
            daughter_indices = sorted_child_order[split_parent_index, :2]
            daughter_a = int(daughter_indices[0])
            daughter_b = int(daughter_indices[1])
            daughter_a_fraction = float(parent_fraction_matrix[split_parent_index, daughter_a])
            daughter_b_fraction = float(parent_fraction_matrix[split_parent_index, daughter_b])
            daughter_profile_pearson = safe_pearson(child_profile[daughter_a], child_profile[daughter_b])
            daughter_profile_cosine = safe_cosine(child_profile[daughter_a], child_profile[daughter_b])
            family_differences = family_profile_difference(child_profile[daughter_a], child_profile[daughter_b], ordered_profile_keys)
            families_any_abs2 = int(sum((row['n_cells_abs_difference_ge_2'] > 0 for row in family_differences)))
            families_opposite_notable = int(sum((row['n_cells_opposite_notable_sign'] > 0 for row in family_differences)))
            for family_row in family_differences:
                daughter_family_rows.append({'dimensions': d, 'parent_k': parent_k, 'child_k': child_k, 'parent_model': parent_mid, 'child_model': child_mid, 'candidate_split_parent_cluster': parent_clusters[split_parent_index], 'daughter_a_cluster': child_clusters[daughter_a], 'daughter_b_cluster': child_clusters[daughter_b], 'daughter_a_parent_fraction': daughter_a_fraction, 'daughter_b_parent_fraction': daughter_b_fraction, 'daughter_profile_pearson': daughter_profile_pearson, 'daughter_profile_cosine': daughter_profile_cosine, **family_row})
            ari, ami = archived_lookup[d, parent_k, child_k]
            hierarchy_rows.append({'dimensions': d, 'parent_k': parent_k, 'child_k': child_k, 'parent_model': parent_mid, 'child_model': child_mid, 'ari': ari, 'ami': ami, 'child_parent_purity': child_parent_purity, 'mean_child_dominant_parent_fraction': float(np.mean(child_dominant_fractions)), 'min_child_dominant_parent_fraction': float(np.min(child_dominant_fractions)), 'parent_child_purity': parent_child_purity, 'mean_parent_dominant_child_fraction': float(np.mean(parent_dominant_fractions)), 'candidate_split_parent_cluster': parent_clusters[split_parent_index], 'split_parent_second_child_fraction': float(second_largest[split_parent_index]), 'split_parent_two_child_fraction': daughter_a_fraction + daughter_b_fraction, 'daughter_a_cluster': child_clusters[daughter_a], 'daughter_b_cluster': child_clusters[daughter_b], 'daughter_a_parent_fraction': daughter_a_fraction, 'daughter_b_parent_fraction': daughter_b_fraction, 'daughter_profile_pearson': daughter_profile_pearson, 'daughter_profile_cosine': daughter_profile_cosine, 'daughter_a_participants_represented': support_lookup[child_mid, child_clusters[daughter_a]], 'daughter_b_participants_represented': support_lookup[child_mid, child_clusters[daughter_b]], 'families_with_any_residual_difference_ge_2': families_any_abs2, 'families_with_opposite_notable_residual_sign': families_opposite_notable, 'split_parent_definition': 'parent cluster with largest second-largest event fraction across K+1 child clusters'})
    return (pd.DataFrame(split_rows), pd.DataFrame(profile_rows), pd.DataFrame(hierarchy_rows), pd.DataFrame(daughter_family_rows))

def build_candidate_registry(macro_summary: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    available = set(macro_summary['model_id'].astype(str))
    reasons: dict[str, list[str]] = {}
    for mid, reason in FOCAL_MODEL_REASONS.items():
        if mid not in available:
            raise RuntimeError(f'Prespecified focal model missing: {mid}')
        reasons.setdefault(mid, []).append(reason)
    for mid in pareto['model_id'].astype(str):
        if mid not in available:
            raise RuntimeError(f'Pareto model {mid} absent from macro_model_summary.csv')
        reasons.setdefault(mid, []).append('Pareto-nondominated in prospective macro-grid screen')
    rows = []
    for mid in sorted(reasons, key=lambda value: parse_model_id(value)):
        d, k = parse_model_id(mid)
        rows.append({'model_id': mid, 'dimensions': d, 'k': k, 'candidate_reason': ' | '.join(reasons[mid]), 'pareto_nondominated': bool(mid in set(pareto['model_id'].astype(str)))})
    return pd.DataFrame(rows)

def candidate_profiles(cells: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mids = set(registry['model_id'].astype(str))
    out = cells[cells['model_id'].astype(str).isin(mids)].copy()
    out['absolute_standardized_residual'] = out['adjusted_standardized_residual'].abs()
    out['notable_abs_asr_ge_2'] = out['absolute_standardized_residual'] >= 2.0
    out['direction'] = np.where(out['adjusted_standardized_residual'] >= 0, 'enriched', 'depleted')
    out['construct_family'] = out['variable'].map(VARIABLE_TO_FAMILY)
    out['abs_residual_rank_within_cluster'] = out.groupby(['model_id', 'cluster'])['absolute_standardized_residual'].rank(method='first', ascending=False)
    summary_rows = []
    for (mid, cluster), block in out.groupby(['model_id', 'cluster'], sort=True):
        positive = block.sort_values('adjusted_standardized_residual', ascending=False).head(8)
        negative = block.sort_values('adjusted_standardized_residual', ascending=True).head(8)

        def format_cells(part: pd.DataFrame) -> str:
            return ' | '.join((f'{row.variable}={row.level} (ASR={row.adjusted_standardized_residual:.2f})' for row in part.itertuples(index=False)))
        notable = block[block['notable_abs_asr_ge_2']]
        summary_rows.append({'model_id': mid, 'dimensions': int(block['dimensions'].iloc[0]), 'k': int(block['k'].iloc[0]), 'cluster': int(cluster), 'n_notable_abs_asr_ge_2': int(len(notable)), 'n_construct_families_with_notable_cells': int(notable['construct_family'].nunique()), 'top_positive_residual_cells': format_cells(positive), 'top_negative_residual_cells': format_cells(negative)})
    return (out, pd.DataFrame(summary_rows))

def build_macro_resolution_summary(registry: pd.DataFrame, macro_summary: pd.DataFrame, within_k: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    summary = registry.merge(macro_summary, on=['model_id', 'dimensions', 'k'], how='left', validate='one_to_one', suffixes=('', '_macro'))
    previous_d_rows = []
    next_d_rows = []
    for row in within_k.itertuples(index=False):
        next_d_rows.append({'model_id': row.model_a, 'same_k_next_D_model': row.model_b, 'same_k_next_D_ari': row.ari, 'same_k_next_D_ami': row.ami, 'same_k_next_D_profile_pearson_mean': row.profile_pearson_mean, 'same_k_next_D_profile_pearson_min': row.profile_pearson_min})
        previous_d_rows.append({'model_id': row.model_b, 'same_k_previous_D_model': row.model_a, 'same_k_previous_D_ari': row.ari, 'same_k_previous_D_ami': row.ami, 'same_k_previous_D_profile_pearson_mean': row.profile_pearson_mean, 'same_k_previous_D_profile_pearson_min': row.profile_pearson_min})
    summary = summary.merge(pd.DataFrame(previous_d_rows), on='model_id', how='left', validate='one_to_one').merge(pd.DataFrame(next_d_rows), on='model_id', how='left', validate='one_to_one')
    parent_rows = []
    child_rows = []
    for row in hierarchy.itertuples(index=False):
        child_rows.append({'model_id': row.child_model, 'same_D_parent_resolution_model': row.parent_model, 'same_D_parent_resolution_ari': row.ari, 'same_D_parent_resolution_ami': row.ami, 'same_D_parent_child_nestedness': row.child_parent_purity, 'same_D_candidate_split_parent_cluster': row.candidate_split_parent_cluster, 'same_D_split_daughter_a_cluster': row.daughter_a_cluster, 'same_D_split_daughter_b_cluster': row.daughter_b_cluster, 'same_D_split_daughter_profile_pearson': row.daughter_profile_pearson, 'same_D_split_daughter_profile_cosine': row.daughter_profile_cosine, 'same_D_split_daughter_min_participant_support': min(row.daughter_a_participants_represented, row.daughter_b_participants_represented), 'same_D_split_families_with_abs_resid_diff_ge_2': row.families_with_any_residual_difference_ge_2, 'same_D_split_families_with_opposite_notable_sign': row.families_with_opposite_notable_residual_sign})
        parent_rows.append({'model_id': row.parent_model, 'same_D_next_resolution_model': row.child_model, 'same_D_next_resolution_ari': row.ari, 'same_D_next_resolution_ami': row.ami, 'same_D_next_child_parent_nestedness': row.child_parent_purity})
    summary = summary.merge(pd.DataFrame(child_rows), on='model_id', how='left', validate='one_to_one').merge(pd.DataFrame(parent_rows), on='model_id', how='left', validate='one_to_one')
    return summary

def low_complexity_pair_similarity(partitions: dict[tuple[int, int], np.ndarray]) -> pd.DataFrame:
    rows = []
    for mid_a, mid_b, rationale in LOW_COMPLEXITY_PAIRS:
        d_a, k_a = parse_model_id(mid_a)
        d_b, k_b = parse_model_id(mid_b)
        a = partitions[d_a, k_a]
        b = partitions[d_b, k_b]
        ari, ami = direct_partition_similarity(a, b)
        table = contingency_labels(a, b, k_a, k_b)
        if k_a <= k_b:
            lower_labels, higher_labels = (a, b)
            lower_k, higher_k = (k_a, k_b)
            nesting = contingency_labels(lower_labels, higher_labels, lower_k, higher_k)
        else:
            lower_labels, higher_labels = (b, a)
            lower_k, higher_k = (k_b, k_a)
            nesting = contingency_labels(lower_labels, higher_labels, lower_k, higher_k)
        higher_sizes = nesting.sum(axis=0)
        child_parent_purity = float(nesting.max(axis=0).sum() / nesting.sum())
        rows.append({'model_a': mid_a, 'model_b': mid_b, 'rationale': rationale, 'dimensions_a': d_a, 'k_a': k_a, 'dimensions_b': d_b, 'k_b': k_b, 'ari': ari, 'ami': ami, 'lower_K_to_higher_K_child_parent_purity': child_parent_purity, 'contingency_counts_json': json.dumps(table.tolist())})
    return pd.DataFrame(rows)

def report_model_line(row: pd.Series) -> str:
    fields = [f"{row['model_id']}", f"D={int(row['dimensions'])}", f"K={int(row['k'])}"]
    for label, column in [('sil', 'silhouette'), ('bootARI', 'bootstrap_stability_ari'), ('heldoutΔR²', 'heldout_family_mean_null_corrected_R2'), ('LOPO', 'lopo_ari_mean'), ('adjD', 'adjacent_D_ari_mean')]:
        if column in row and pd.notna(row[column]):
            fields.append(f'{label}={float(row[column]):.4f}')
    return '; '.join(fields)

def build_report(macro_resolution_summary: pd.DataFrame, k4_path: pd.DataFrame, k3_to_k4: pd.DataFrame, low_pairs: pd.DataFrame, qa: pd.DataFrame) -> str:
    lines = ['# Macro-resolution hierarchy post-processing — REFIT 256', '', '## Scope', '', 'This report post-processes the completed construct-aligned macro grid. It does not rerun MCA retention, held-out construct permutations, LOPO MCA, projection selection, basin analysis, transitions, or successor representation.', '', 'The frozen geometric roles remain:', '', '- D13: full-sample MCA reference', '- D10: participant-robust MCA reference', '', 'No automatic macro D/K winner is declared.', '', '## Focused candidate registry', '']
    for _, row in macro_resolution_summary.sort_values(['dimensions', 'k']).iterrows():
        lines.append(f"- {report_model_line(row)}; reason={row['candidate_reason']}")
    lines.extend(['', '## K4 dimensional path: D3 through D8', ''])
    for row in k4_path.itertuples(index=False):
        lines.append(f'- {row.model_a} -> {row.model_b}: ARI={row.ari:.4f}; AMI={row.ami:.4f}; mean matched-profile r={row.profile_pearson_mean:.4f}; minimum matched-profile r={row.profile_pearson_min:.4f}; event-match fraction={row.hungarian_event_match_fraction:.4f}.')
    lines.extend(['', 'This path separates event-membership persistence from residual-profile persistence. A dimensional step can reorganize memberships while retaining recognizable macro-regime identities.', '', '## K3 -> K4 split across D', ''])
    for row in k3_to_k4.itertuples(index=False):
        lines.append(f'- D{int(row.dimensions)}: ARI={row.ari:.4f}; AMI={row.ami:.4f}; child-parent purity={row.child_parent_purity:.4f}; candidate split parent=C{int(row.candidate_split_parent_cluster)} -> C{int(row.daughter_a_cluster)} + C{int(row.daughter_b_cluster)}; daughter profile r={row.daughter_profile_pearson:.4f}; family residual differences >=2 in {int(row.families_with_any_residual_difference_ge_2)}/6 families; opposite notable signs in {int(row.families_with_opposite_notable_residual_sign)}/6 families.')
    lines.extend(['', 'The split-parent label is descriptive: it is the K3 parent sending the largest second-largest fraction of its events to two K4 daughters.', '', '## Low-complexity representative comparisons', ''])
    for row in low_pairs.itertuples(index=False):
        lines.append(f'- {row.model_a} vs {row.model_b}: ARI={row.ari:.4f}; AMI={row.ami:.4f}; lower-K -> higher-K child-parent purity={row.lower_K_to_higher_K_child_parent_purity:.4f}. ({row.rationale})')
    lines.extend(['', 'These cross-D/cross-K comparisons are descriptive and are not treated as formal nesting tests. Same-D K -> K+1 tables are the primary resolution hierarchy evidence.', '', '## Adjudication framework', '', 'The outputs support the following prospective interpretation rules:', '', '1. Within fixed K, prefer the earliest D at which macro-regime identities become persistent if later dimensions mainly refine the same profiles.', '2. Retain K+1 as a meaningful finer resolution only when its added daughter regimes are multiconstruct-distinct, adequately participant-supported, and not merely a narrow or unstable subdivision.', '3. If K2, K3, and K4 form coherent nested refinements, report hierarchical macro resolution rather than forcing one uniquely true K.', '4. Existing construct-recovery, bootstrap, LOPO, and geometric metrics are constraints on interpretation; they are not recomputed or collapsed into a new weighted score here.', '', '## QA', '', f"{int(qa['passed'].sum())}/{len(qa)} checks passed.", ''])
    return '\n'.join(lines)

def validate_refit_provenance(args, coordinates_df, tables):
    metadata_path = args.macro_tables_dir.parent / 'RUN_METADATA.json'
    if not metadata_path.is_file():
        raise RuntimeError(f'Missing macro-grid metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('dataset_version') != DATASET_VERSION:
        raise RuntimeError('Macro-grid inputs are not the chronological 256-event refit.')
    if metadata.get('n_events') != EXPECTED_EVENTS or len(coordinates_df) != EXPECTED_EVENTS:
        raise RuntimeError('Macro grid and coordinates must both contain 256 events.')
    participants = coordinates_df['participant_id_norm'].astype(str).str.strip().str.lower()
    counts = participants.value_counts().sort_index().to_dict()
    if counts != EXPECTED_PARTICIPANT_COUNTS or metadata.get('participant_event_counts') != counts:
        raise RuntimeError(f'Participant event counts do not match the refit: {counts}')
    if coordinates_df[KEY].isna().any().any():
        raise RuntimeError('Missing event identifiers in coordinates.')
    coordinate_values = coordinates_df[[f'Dim{i}' for i in range(54)]].to_numpy(float)
    if not np.isfinite(coordinate_values).all():
        raise RuntimeError('Nonfinite full MCA coordinates.')
    coordinate_hash = sha256_file(args.full_coordinates)
    if coordinate_hash != metadata.get('input_full_coordinates_sha256'):
        raise RuntimeError('Coordinates do not match the completed macro-grid run.')
    if metadata.get('seed') != args.seed:
        raise RuntimeError('Reconstruction seed must match the macro-grid run.')
    if metadata.get('dimensions_evaluated') != D_VALUES or metadata.get('k_evaluated') != K_VALUES:
        raise RuntimeError('Macro-grid candidate range differs from this hierarchy analysis.')
    if metadata.get('full_active_variables') != FULL_ACTIVE or metadata.get('construct_families') != CONSTRUCT_FAMILIES:
        raise RuntimeError('Macro-grid variables or construct families differ from this hierarchy schema.')
    if (metadata.get('full_sample_geometry_reference'), metadata.get('participant_robust_geometry_reference')) != ('D13', 'D10'):
        raise RuntimeError('Review hierarchy reference labels for changed MCA dimensionality.')
    upstream = metadata.get('upstream_reference', {})
    if upstream.get('input_sha256') != metadata.get('input_event_table_sha256') or not metadata.get('input_event_table_sha256'):
        raise RuntimeError('Macro-grid event-input provenance is inconsistent.')
    if upstream.get('files', {}).get('coordinates', {}).get('sha256') != coordinate_hash:
        raise RuntimeError('Macro-grid upstream coordinate provenance is inconsistent.')
    qa = tables['upstream_qa_checks']
    ensure_columns(qa, ['passed'], 'Macro-grid QA')
    if qa.empty or not qa['passed'].astype(str).str.lower().isin({'true', '1'}).all() or metadata.get('qa_passed') != len(qa) or (metadata.get('qa_total') != len(qa)):
        raise RuntimeError('Macro-grid QA is incomplete or inconsistent with its metadata.')
    expected_models = {model_id(d, k) for d in D_VALUES for k in K_VALUES}
    for name in ['full_geometry_macro_grid', 'macro_model_summary']:
        frame = tables[name]
        ensure_columns(frame, ['model_id', 'dimensions', 'k'], name)
        if frame['model_id'].duplicated().any() or set(frame['model_id']) != expected_models:
            raise RuntimeError(f'Incomplete or duplicated macro models in {name}.')
        if any((row.model_id != model_id(int(row.dimensions), int(row.k)) for row in frame.itertuples())):
            raise RuntimeError(f'Model identifiers disagree with D/K in {name}.')
    summary = tables['macro_model_summary']
    ensure_columns(summary, ['pareto_nondominated'], 'Macro-model summary')
    expected_pareto = set(summary.loc[summary['pareto_nondominated'].astype(str).str.lower().isin({'true', '1'}), 'model_id'])
    pareto = tables['pareto_nondominated_models']
    if pareto['model_id'].duplicated().any() or set(pareto['model_id']) != expected_pareto:
        raise RuntimeError('Pareto table disagrees with the macro-model summary.')
    print('[OK] Matched 256-event macro grid and full MCA coordinates', flush=True)
    return {'macro_metadata_path': str(metadata_path.resolve()), 'macro_metadata_sha256': sha256_file(metadata_path), 'macro_script_sha256': metadata.get('script_sha256'), 'input_event_table_sha256': metadata['input_event_table_sha256'], 'full_coordinates_sha256': coordinate_hash, 'participant_event_counts': counts, 'full_sample_agnostic_reference': metadata.get('full_sample_agnostic_reference'), 'participant_robust_agnostic_reference': metadata.get('participant_robust_agnostic_reference')}

def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_out = args.out_dir / 'tables'
    tables_out.mkdir(parents=True, exist_ok=True)
    paths = {'full_geometry_macro_grid': args.macro_tables_dir / 'full_geometry_macro_grid.csv', 'heldout_construct_family_alignment': args.macro_tables_dir / 'heldout_construct_family_alignment.csv', 'full_geometry_feature_associations': args.macro_tables_dir / 'full_geometry_feature_associations.csv', 'full_geometry_cluster_feature_cells': args.macro_tables_dir / 'full_geometry_cluster_feature_cells.csv', 'participant_support_by_model': args.macro_tables_dir / 'participant_support_by_model.csv', 'lopo_partition_stability': args.macro_tables_dir / 'lopo_partition_stability.csv', 'neighboring_dimension_partition_similarity': args.macro_tables_dir / 'neighboring_dimension_partition_similarity.csv', 'neighboring_k_partition_similarity': args.macro_tables_dir / 'neighboring_k_partition_similarity.csv', 'macro_model_summary': args.macro_tables_dir / 'macro_model_summary.csv', 'pareto_nondominated_models': args.macro_tables_dir / 'pareto_nondominated_models.csv', 'upstream_qa_checks': args.macro_tables_dir / 'qa_checks.csv'}
    coordinates_df = read_csv_required(args.full_coordinates)
    tables = {name: read_csv_required(path) for name, path in paths.items()}
    ensure_columns(coordinates_df, KEY + [f'Dim{i}' for i in range(54)], args.full_coordinates.name)
    if coordinates_df.duplicated(KEY).any():
        raise RuntimeError('Duplicate event keys in full-coordinate table.')
    refit_provenance = validate_refit_provenance(args, coordinates_df, tables)
    macro_grid = tables['full_geometry_macro_grid']
    family_alignment = tables['heldout_construct_family_alignment']
    feature_associations = tables['full_geometry_feature_associations']
    cells = tables['full_geometry_cluster_feature_cells']
    participant_support = tables['participant_support_by_model']
    lopo = tables['lopo_partition_stability']
    neighboring_d = tables['neighboring_dimension_partition_similarity']
    neighboring_k = tables['neighboring_k_partition_similarity']
    macro_summary = tables['macro_model_summary']
    pareto = tables['pareto_nondominated_models']
    upstream_qa = tables['upstream_qa_checks']
    ensure_columns(macro_summary, ['model_id', 'dimensions', 'k'], 'macro_model_summary.csv')
    ensure_columns(cells, ['model_id', 'dimensions', 'k', 'cluster', 'variable', 'construct_family', 'level', 'observed', 'expected', 'adjusted_standardized_residual', 'cluster_fraction'], 'full_geometry_cluster_feature_cells.csv')
    ensure_columns(participant_support, ['model_id', 'dimensions', 'k', 'cluster', 'n_events', 'participants_represented'], 'participant_support_by_model.csv')
    ensure_columns(upstream_qa, ['check', 'passed'], 'qa_checks.csv')
    upstream_pass = upstream_qa['passed'].astype(str).str.lower().isin({'true', '1'})
    if not upstream_pass.all():
        failed = upstream_qa.loc[~upstream_pass]
        raise RuntimeError('Upstream macro-grid QA is not fully passing:\n' + failed.to_string(index=False))
    unknown_variables = sorted(set(cells['variable'].astype(str)) - set(FULL_ACTIVE))
    if unknown_variables:
        raise RuntimeError(f'Unexpected variables in residual profiles: {unknown_variables}')
    coordinates = coordinates_df[[f'Dim{i}' for i in range(54)]].apply(pd.to_numeric, errors='raise').to_numpy(float)
    print('[stage] Deterministic full-geometry membership reconstruction', flush=True)
    partitions = reconstruct_partitions(coordinates, seed=args.seed)
    memberships = membership_table(coordinates_df[KEY], partitions)
    print('[stage] Reconstruction cross-checks', flush=True)
    crosscheck_d = crosscheck_neighboring_d(neighboring_d, partitions)
    crosscheck_k = crosscheck_neighboring_k(neighboring_k, partitions)
    if not crosscheck_d['passed'].all():
        bad = crosscheck_d.loc[~crosscheck_d['passed']]
        raise RuntimeError('Adjacent-D reconstruction does not reproduce archived ARI/AMI:\n' + bad.to_string(index=False))
    if not crosscheck_k['passed'].all():
        bad = crosscheck_k.loc[~crosscheck_k['passed']]
        raise RuntimeError('Adjacent-K reconstruction does not reproduce archived ARI/AMI:\n' + bad.to_string(index=False))
    ordered_profile_keys = profile_keys(cells)
    if len(ordered_profile_keys) != 65:
        raise RuntimeError(f'Expected 65 full-geometry variable-level profile cells; found {len(ordered_profile_keys)}.')
    first_variable = FULL_ACTIVE[0]
    archived_cluster_sizes = cells[cells['variable'] == first_variable].groupby(['model_id', 'cluster'], as_index=False)['observed'].sum().rename(columns={'observed': 'archived_cluster_size'})
    reconstructed_size_rows = []
    for d in D_VALUES:
        for k in K_VALUES:
            labels = partitions[d, k]
            for cluster_zero in range(k):
                reconstructed_size_rows.append({'model_id': model_id(d, k), 'cluster': cluster_zero + 1, 'reconstructed_cluster_size': int(np.sum(labels == cluster_zero))})
    size_check = archived_cluster_sizes.merge(pd.DataFrame(reconstructed_size_rows), on=['model_id', 'cluster'], how='outer', validate='one_to_one')
    size_check['passed'] = size_check['archived_cluster_size'] == size_check['reconstructed_cluster_size']
    if not size_check['passed'].all():
        bad = size_check.loc[~size_check['passed']]
        raise RuntimeError('Reconstructed K-means cluster labels/sizes do not match archived residual-profile cluster labels:\n' + bad.to_string(index=False))
    print('[stage] Same-K dimensional persistence', flush=True)
    within_k, matched_profiles = same_k_dimension_persistence(partitions, cells, ordered_profile_keys, neighboring_d)
    print('[stage] Same-D K -> K+1 hierarchy', flush=True)
    split_matrix, split_profile_similarity, hierarchy, daughter_family_difference = k_resolution_hierarchy(partitions, cells, ordered_profile_keys, neighboring_k, participant_support)
    print('[stage] Candidate registry and profiles', flush=True)
    registry = build_candidate_registry(macro_summary, pareto)
    candidate_profile_cells, candidate_profile_summary = candidate_profiles(cells, registry)
    macro_resolution_summary = build_macro_resolution_summary(registry, macro_summary, within_k, hierarchy)
    k4_path = within_k[(within_k['k'] == 4) & (within_k['dimensions_a'] >= 3) & (within_k['dimensions_b'] <= 8)].copy()
    k3_to_k4 = hierarchy[(hierarchy['parent_k'] == 3) & (hierarchy['child_k'] == 4)].copy()
    low_pairs = low_complexity_pair_similarity(partitions)
    upstream_candidate_metrics = macro_summary[macro_summary['model_id'].isin(registry['model_id'])].copy()
    qa_rows = [{'check': '256-event macro-grid provenance and candidate coverage', 'observed': 'matched', 'expected': 'matched', 'passed': True}]

    def qa(check: str, observed, expected, passed: bool) -> None:
        qa_rows.append({'check': check, 'observed': observed, 'expected': expected, 'passed': bool(passed)})
    qa('event rows', len(coordinates_df), EXPECTED_EVENTS, len(coordinates_df) == EXPECTED_EVENTS)
    qa('full macro models available', macro_summary['model_id'].nunique(), EXPECTED_MODELS, macro_summary['model_id'].nunique() == EXPECTED_MODELS)
    qa('reconstructed K-means models', len(partitions), EXPECTED_MODELS, len(partitions) == EXPECTED_MODELS)
    qa('adjacent-D archived rows', len(neighboring_d), EXPECTED_ADJACENT_D, len(neighboring_d) == EXPECTED_ADJACENT_D)
    qa('adjacent-D membership cross-checks', int(crosscheck_d['passed'].sum()), EXPECTED_ADJACENT_D, len(crosscheck_d) == EXPECTED_ADJACENT_D and crosscheck_d['passed'].all())
    qa('adjacent-K archived rows', len(neighboring_k), EXPECTED_ADJACENT_K, len(neighboring_k) == EXPECTED_ADJACENT_K)
    qa('adjacent-K membership cross-checks', int(crosscheck_k['passed'].sum()), EXPECTED_ADJACENT_K, len(crosscheck_k) == EXPECTED_ADJACENT_K and crosscheck_k['passed'].all())
    qa('full residual-profile cells', len(cells), EXPECTED_PROFILE_CELLS, len(cells) == EXPECTED_PROFILE_CELLS)
    qa('unique residual-profile variable-level cells', len(ordered_profile_keys), 65, len(ordered_profile_keys) == 65)
    qa('cluster label/size reconstruction', int(size_check['passed'].sum()), len(size_check), size_check['passed'].all())
    qa('same-K dimensional persistence rows', len(within_k), EXPECTED_ADJACENT_D, len(within_k) == EXPECTED_ADJACENT_D)
    qa('same-D K->K+1 hierarchy rows', len(hierarchy), EXPECTED_ADJACENT_K, len(hierarchy) == EXPECTED_ADJACENT_K)
    qa('proper LOPO rows preserved', len(lopo), 300, len(lopo) == 300)
    qa('held-out construct-family rows preserved', len(family_alignment), 360, len(family_alignment) == 360)
    qa('full feature-association rows preserved', len(feature_associations), 660, len(feature_associations) == 660)
    qa('candidate registry contains D2/K2 D3/K3 D4/K4 D6/K4 D7/K4', int(registry['model_id'].isin(['D2_K2', 'D3_K3', 'D4_K4', 'D6_K4', 'D7_K4']).sum()), 5, registry['model_id'].isin(['D2_K2', 'D3_K3', 'D4_K4', 'D6_K4', 'D7_K4']).sum() == 5)
    qa('upstream macro-grid QA preserved', int(upstream_pass.sum()), len(upstream_qa), upstream_pass.all())
    qa_table = pd.DataFrame(qa_rows)
    if not qa_table['passed'].all():
        failed = qa_table.loc[~qa_table['passed']]
        raise RuntimeError('Macro-resolution hierarchy QA failed:\n' + failed.to_string(index=False))
    memberships.to_csv(tables_out / 'reconstructed_event_memberships.csv', index=False)
    within_k.to_csv(tables_out / 'within_k_dimension_persistence.csv', index=False)
    matched_profiles.to_csv(tables_out / 'matched_cluster_profiles_across_dimensions.csv', index=False)
    split_matrix.to_csv(tables_out / 'k_to_kplus1_event_split_matrix.csv', index=False)
    split_profile_similarity.to_csv(tables_out / 'k_to_kplus1_cluster_profile_similarity.csv', index=False)
    hierarchy.to_csv(tables_out / 'k_to_kplus1_hierarchy_summary.csv', index=False)
    daughter_family_difference.to_csv(tables_out / 'split_daughter_family_difference.csv', index=False)
    registry.to_csv(tables_out / 'candidate_model_registry.csv', index=False)
    candidate_profile_cells.to_csv(tables_out / 'candidate_macro_regime_profiles.csv', index=False)
    candidate_profile_summary.to_csv(tables_out / 'candidate_macro_regime_profile_summary.csv', index=False)
    macro_resolution_summary.to_csv(tables_out / 'macro_resolution_hierarchy_summary.csv', index=False)
    k4_path.to_csv(tables_out / 'k4_dimensional_path_D3_D8.csv', index=False)
    k3_to_k4.to_csv(tables_out / 'k3_to_k4_split_D2_D13.csv', index=False)
    low_pairs.to_csv(tables_out / 'low_complexity_candidate_pair_similarity.csv', index=False)
    upstream_candidate_metrics.to_csv(tables_out / 'upstream_candidate_metrics_REFERENCE_ONLY.csv', index=False)
    crosscheck_d.to_csv(tables_out / 'qa_reconstructed_neighboring_d_crosscheck.csv', index=False)
    crosscheck_k.to_csv(tables_out / 'qa_reconstructed_neighboring_k_crosscheck.csv', index=False)
    size_check.to_csv(tables_out / 'qa_reconstructed_cluster_size_crosscheck.csv', index=False)
    qa_table.to_csv(tables_out / 'qa_checks.csv', index=False)
    report = build_report(macro_resolution_summary, k4_path, k3_to_k4, low_pairs, qa_table)
    (args.out_dir / 'MACRO_RESOLUTION_HIERARCHY_REPORT.md').write_text(report, encoding='utf-8')
    metadata = {'analysis': 'macro-resolution hierarchy post-processing', 'dataset_version': DATASET_VERSION, 'n_events': len(coordinates_df), 'participant_event_counts': refit_provenance['participant_event_counts'], 'script_sha256': sha256_file(Path(__file__)), 'refit_provenance': refit_provenance, 'qa_passed': int(qa_table['passed'].sum()), 'qa_total': len(qa_table), 'scientific_role': 'post-process completed construct-aligned macro grid; no upstream inference rerun', 'command': ' '.join(sys.argv), 'elapsed_seconds': time.time() - started, 'full_coordinates': str(args.full_coordinates.resolve()), 'full_coordinates_sha256': sha256_file(args.full_coordinates), 'macro_tables_dir': str(args.macro_tables_dir.resolve()), 'macro_input_sha256': {name: sha256_file(path) for name, path in paths.items()}, 'dimensions_reconstructed': D_VALUES, 'k_reconstructed': K_VALUES, 'kmeans_reconstruction': {'n_init': 10, 'random_state': args.seed, 'purpose': 'event-membership reconstruction only; archived adjacent ARI/AMI must be reproduced exactly'}, 'construct_families': CONSTRUCT_FAMILIES, 'prespecified_focal_models': FOCAL_MODEL_REASONS, 'pareto_models_added_automatically': sorted(pareto['model_id'].astype(str).tolist()), 'automatic_macro_winner': False, 'weighted_score': False, 'rerun_flags': {'mca_retention': False, 'construct_blind_mca': False, 'construct_permutations': False, 'bootstrap_stability': False, 'lopo_mca': False, 'projection': False, 'basin': False, 'transition': False, 'successor': False}, 'software': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn_version}}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('\n[OK] Macro-resolution hierarchy post-processing complete.')
    print(f'[OK] Events: {len(coordinates_df)} / {EXPECTED_EVENTS}')
    print(f"[OK] Full macro models available: {macro_summary['model_id'].nunique()} / {EXPECTED_MODELS}")
    print(f'[OK] Reconstructed K-means models: {len(partitions)} / {EXPECTED_MODELS}')
    print(f"[OK] Adjacent-D membership cross-checks: {int(crosscheck_d['passed'].sum())} / {EXPECTED_ADJACENT_D}")
    print(f"[OK] Adjacent-K membership cross-checks: {int(crosscheck_k['passed'].sum())} / {EXPECTED_ADJACENT_K}")
    print(f'[OK] Full residual-profile cells: {len(cells)} / {EXPECTED_PROFILE_CELLS}')
    print('[OK] No MCA, permutation, LOPO, projection, basin, transition, or SR analysis was rerun.')
    print(f"[OK] QA: {int(qa_table['passed'].sum())}/{len(qa_table)} passed.")
    print(f'[OK] Output: {args.out_dir.resolve()}')
    print('\n[RESULT] Focused candidate models:')
    for row in registry.sort_values(['dimensions', 'k']).itertuples(index=False):
        print(f'  {row.model_id}: {row.candidate_reason}')
    print('\n[RESULT] K4 dimensional path D3->D8:')
    print(k4_path[['model_a', 'model_b', 'ari', 'ami', 'profile_pearson_mean', 'profile_pearson_min', 'hungarian_event_match_fraction']].to_string(index=False))
    print('\n[RESULT] K3->K4 resolution split by D:')
    print(k3_to_k4[['dimensions', 'ari', 'ami', 'child_parent_purity', 'candidate_split_parent_cluster', 'daughter_a_cluster', 'daughter_b_cluster', 'daughter_profile_pearson', 'families_with_any_residual_difference_ge_2', 'families_with_opposite_notable_residual_sign']].to_string(index=False))
    print('\n[NOTE] This script does not declare a final macro D/K winner. Review the hierarchy report and the candidate regime profiles.')
if __name__ == '__main__':
    main()
