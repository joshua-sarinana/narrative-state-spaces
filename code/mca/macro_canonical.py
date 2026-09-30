"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
DATASET_VERSION = 'chronological_256_v2'
EXPECTED_EVENTS = 256
EXPECTED_PARTICIPANTS = 5
MODEL_ID = 'D4_K4'
D = 4
K = 4
DIMS = ['Dim0', 'Dim1', 'Dim2', 'Dim3']
KEYS = ['participant_id_norm', 'hippo_id_norm']
ACTIVE_VARIABLES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
CONSTRUCT_FAMILIES = ['context', 'hippocampal_mechanism', 'anchor_landmark', 'representational_organization', 'semantic_content', 'affective_content']
STAGE_FILES = {'mca_dir': {'full_coordinates': 'mca_full_54d_row_coordinates.csv', 'spectrum': 'mca_full_spectrum_raw_benzecri_greenacre.csv', 'upstream_selected_k': 'kmeans_consensus_selected_k_by_dimension.csv'}, 'macro_dir': {'feature_associations': 'full_geometry_feature_associations.csv', 'feature_cells': 'full_geometry_cluster_feature_cells.csv', 'participant_support': 'participant_support_by_model.csv', 'full_macro_grid': 'full_geometry_macro_grid.csv', 'heldout_construct_alignment': 'heldout_construct_family_alignment.csv', 'macro_model_summary': 'macro_model_summary.csv', 'lopo_stability': 'lopo_partition_stability.csv'}, 'hierarchy_dir': {'hierarchy_memberships': 'reconstructed_event_memberships.csv', 'candidate_profile_summary': 'candidate_macro_regime_profile_summary.csv', 'within_k_persistence': 'within_k_dimension_persistence.csv', 'k3_to_k4_hierarchy': 'k3_to_k4_split_D2_D13.csv', 'low_complexity_pairs': 'low_complexity_candidate_pair_similarity.csv'}, 'projection_dir': {'projection_winners': 'full_sample_projection_winners_by_D.csv', 'projection_candidates': 'full_sample_projection_candidates_D2_D13.csv'}}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-coordinates', type=Path)
    parser.add_argument('--spectrum', type=Path)
    parser.add_argument('--upstream-selected-k', type=Path)
    parser.add_argument('--hierarchy-memberships', type=Path)
    parser.add_argument('--feature-associations', type=Path)
    parser.add_argument('--feature-cells', type=Path)
    parser.add_argument('--participant-support', type=Path)
    parser.add_argument('--full-macro-grid', type=Path)
    parser.add_argument('--heldout-construct-alignment', type=Path)
    parser.add_argument('--macro-model-summary', type=Path)
    parser.add_argument('--lopo-stability', type=Path)
    parser.add_argument('--candidate-profile-summary', type=Path)
    parser.add_argument('--within-k-persistence', type=Path)
    parser.add_argument('--k3-to-k4-hierarchy', type=Path)
    parser.add_argument('--low-complexity-pairs', type=Path)
    parser.add_argument('--projection-winners', type=Path)
    parser.add_argument('--projection-candidates', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    for stage in STAGE_FILES:
        parser.add_argument('--' + stage.replace('_', '-'), type=Path)
    args = parser.parse_args()
    for stage, files in STAGE_FILES.items():
        directory = getattr(args, stage)
        for key, filename in files.items():
            if getattr(args, key) is None:
                if directory is None:
                    parser.error('Supply --' + stage.replace('_', '-') + ' or --' + key.replace('_', '-'))
                setattr(args, key, directory / 'tables' / filename)
    if args.seed != 42:
        parser.error('Canonical execution requires --seed 42.')
    return args

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f'QA FAIL: {message}')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_csv(path: Path) -> pd.DataFrame:
    require(path.is_file(), f'input file not found: {path}')
    return pd.read_csv(path, low_memory=False)

def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    require(not missing, f'{label} missing columns: {missing}')

def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ensure_columns(out, KEYS, 'event table')
    for key in KEYS:
        out[key] = out[key].astype(str).str.strip().str.lower()
    return out

def find_dimension_column(df: pd.DataFrame, label: str) -> str:
    candidates = ['dimensions', 'dimension', 'reference_dimensions', 'reference_dimension']
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f'QA FAIL: {label} lacks a recognizable dimensionality column. Observed columns: {list(df.columns)}')

def find_k_column(df: pd.DataFrame, label: str) -> str:
    for candidate in ['k', 'selected_k']:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f'QA FAIL: {label} lacks k/selected_k. Observed columns: {list(df.columns)}')

def find_projection_column(df: pd.DataFrame, label: str) -> str:
    for candidate in ['projection', 'projection_name', 'pair']:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f'QA FAIL: {label} lacks a projection column. Observed columns: {list(df.columns)}')

def markdown_table(df: pd.DataFrame, max_rows: int | None=None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)

    def cell(value: object) -> str:
        if pd.isna(value):
            return ''
        if isinstance(value, (float, np.floating)):
            rendered = f'{float(value):.6f}'
        else:
            rendered = str(value)
        return rendered.replace('|', '\\|').replace('\n', ' ')
    headers = [str(column).replace('|', '\\|') for column in df.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join((cell(value) for value in row)) + ' |')
    return '\n'.join(lines)

def align_labels(fitted: np.ndarray, archived: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    fitted_ids = np.array(sorted(np.unique(fitted)))
    archived_ids = np.array(sorted(np.unique(archived)))
    require(len(fitted_ids) == K, f'reconstructed D4/K4 has {len(fitted_ids)} labels')
    require(len(archived_ids) == K, f'archived D4/K4 has {len(archived_ids)} labels')
    overlap = np.array([[int(np.sum((fitted == fitted_id) & (archived == archived_id))) for archived_id in archived_ids] for fitted_id in fitted_ids], dtype=int)
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {int(fitted_ids[row]): int(archived_ids[col]) for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping[int(label)] for label in fitted], dtype=int)
    return (aligned, mapping, overlap)

def build_profile_summary(cells: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    support_lookup = support.set_index('cluster')
    rows = []
    for cluster, block in cells.groupby('cluster', sort=True):
        cluster = int(cluster)
        notable = block[block['adjusted_standardized_residual'].abs() >= 2.0].copy()
        positive = notable[notable['adjusted_standardized_residual'] >= 2.0].sort_values('adjusted_standardized_residual', ascending=False)
        negative = notable[notable['adjusted_standardized_residual'] <= -2.0].sort_values('adjusted_standardized_residual', ascending=True)

        def format_rows(frame: pd.DataFrame, n: int=10) -> str:
            return ' | '.join((f'{row.variable}={row.level} (ASR={row.adjusted_standardized_residual:.2f})' for row in frame.head(n).itertuples(index=False)))
        support_row = support_lookup.loc[cluster]
        participant_counts = json.loads(str(support_row['participant_counts']))
        dominant_participant, dominant_n = max(participant_counts.items(), key=lambda item: (item[1], item[0]))
        rows.append({'model_id': MODEL_ID, 'dimensions': D, 'k': K, 'cluster': cluster, 'n_events': int(support_row['n_events']), 'participants_represented': int(support_row['participants_represented']), 'dominant_participant': dominant_participant, 'dominant_participant_n': int(dominant_n), 'maximum_single_participant_fraction': float(support_row['maximum_single_participant_fraction']), 'n_notable_abs_asr_ge_2': int(len(notable)), 'n_construct_families_with_notable_cells': int(notable['construct_family'].nunique()), 'construct_families_with_notable_cells': ' | '.join(sorted(notable['construct_family'].astype(str).unique())), 'top_positive_residual_cells': format_rows(positive), 'top_negative_residual_cells': format_rows(negative)})
    return pd.DataFrame(rows)

def validate_refit_sources(args, input_paths):
    anchors = {'mca_dir': args.full_coordinates, 'macro_dir': args.macro_model_summary, 'hierarchy_dir': args.hierarchy_memberships, 'projection_dir': args.projection_winners}
    directories = {stage: getattr(args, stage) or anchor.parent.parent for stage, anchor in anchors.items()}
    metadata = {}
    files = {}
    for stage, directory in directories.items():
        path = directory / 'RUN_METADATA.json'
        require(path.is_file(), f'Missing {stage} metadata: {path}')
        record = json.loads(path.read_text(encoding='utf-8'))
        require(record.get('dataset_version') == DATASET_VERSION, f'{stage} must be the chronological 256-event refit')
        qa_path = directory / 'tables' / 'qa_checks.csv'
        qa = read_csv(qa_path)
        require(not qa.empty and qa['passed'].astype(str).str.lower().isin({'true', '1'}).all(), f'{stage} QA has not passed')
        if stage != 'mca_dir':
            require(record.get('n_events') == EXPECTED_EVENTS, f'Wrong event count in {stage}')
            require(record.get('qa_passed') == len(qa) and record.get('qa_total') == len(qa), f'{stage} metadata and QA counts disagree')
        for key, filename in STAGE_FILES[stage].items():
            canonical = directory / 'tables' / filename
            require(canonical.is_file() and sha256_file(input_paths[key]) == sha256_file(canonical), f'{key} differs from its completed stage output')
        metadata[stage] = record
        files[stage] = {'path': str(path.resolve()), 'sha256': sha256_file(path), 'qa_sha256': sha256_file(qa_path)}
    mca, macro, hierarchy, projection = [metadata[k] for k in anchors]
    input_hash = mca.get('input_sha256')
    require(bool(input_hash) and input_hash == macro.get('input_event_table_sha256') == hierarchy.get('refit_provenance', {}).get('input_event_table_sha256') == projection.get('input_sha256'), 'Stage event-input fingerprints disagree')
    coordinate_hash = sha256_file(args.full_coordinates)
    require(coordinate_hash == macro.get('input_full_coordinates_sha256') == hierarchy.get('full_coordinates_sha256'), 'Stage coordinate fingerprints disagree')
    require(hierarchy['refit_provenance']['macro_metadata_sha256'] == files['macro_dir']['sha256'], 'Hierarchy belongs to a different macro-grid run')
    require(macro['upstream_reference']['files']['metadata']['sha256'] == files['mca_dir']['sha256'] == projection['upstream_reference']['files']['metadata']['sha256'], 'Macro-grid or projection uses a different MCA run')
    for key, filename in STAGE_FILES['macro_dir'].items():
        require(sha256_file(input_paths[key]) == hierarchy['macro_input_sha256'].get(Path(filename).stem), f'Hierarchy was computed using different {key}')
    decision_path = directories['mca_dir'] / 'tables/decision_summary.json'
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    require(decision.get('n_events') == EXPECTED_EVENTS and decision.get('input_sha256') == input_hash, 'MCA decision summary does not match the refit')
    require((decision.get('full_sample_parallel_retained_dimensions'), decision.get('participant_robust_min_lopo_dimensions')) == (13, 10), 'Review geometry labels for changed retained dimensions')
    require((projection.get('full_reference_D'), projection.get('participant_robust_reference_D')) == (13, 10), 'Projection reference dimensions disagree')
    return {'stage_metadata': files, 'input_event_table_sha256': input_hash, 'coordinate_sha256': coordinate_hash, 'full_sample_agnostic_reference': macro.get('full_sample_agnostic_reference'), 'participant_robust_agnostic_reference': macro.get('participant_robust_agnostic_reference'), 'full_sample_projection': projection.get('primary_projection'), 'participant_robust_projection': projection.get('robust_full_sample_projection')}

def main() -> None:
    args = parse_args()
    started = time.time()
    input_paths = {'full_coordinates': args.full_coordinates, 'spectrum': args.spectrum, 'upstream_selected_k': args.upstream_selected_k, 'hierarchy_memberships': args.hierarchy_memberships, 'feature_associations': args.feature_associations, 'feature_cells': args.feature_cells, 'participant_support': args.participant_support, 'full_macro_grid': args.full_macro_grid, 'heldout_construct_alignment': args.heldout_construct_alignment, 'macro_model_summary': args.macro_model_summary, 'lopo_stability': args.lopo_stability, 'candidate_profile_summary': args.candidate_profile_summary, 'within_k_persistence': args.within_k_persistence, 'k3_to_k4_hierarchy': args.k3_to_k4_hierarchy, 'low_complexity_pairs': args.low_complexity_pairs, 'projection_winners': args.projection_winners, 'projection_candidates': args.projection_candidates}
    for path in input_paths.values():
        require(path.is_file(), f'required input not found: {path}')
    refit_provenance = validate_refit_sources(args, input_paths)
    observed_coordinate_sha = sha256_file(args.full_coordinates)
    expected_coordinate_sha = refit_provenance['coordinate_sha256']
    coords = normalize_keys(read_csv(args.full_coordinates))
    spectrum = read_csv(args.spectrum)
    upstream_selected_k = read_csv(args.upstream_selected_k)
    hierarchy_memberships = normalize_keys(read_csv(args.hierarchy_memberships))
    feature_associations_all = read_csv(args.feature_associations)
    feature_cells_all = read_csv(args.feature_cells)
    participant_support_all = read_csv(args.participant_support)
    full_macro_grid = read_csv(args.full_macro_grid)
    heldout_all = read_csv(args.heldout_construct_alignment)
    model_summary_all = read_csv(args.macro_model_summary)
    lopo_all = read_csv(args.lopo_stability)
    archived_profile_summary_all = read_csv(args.candidate_profile_summary)
    within_k = read_csv(args.within_k_persistence)
    k3_to_k4 = read_csv(args.k3_to_k4_hierarchy)
    low_pairs = read_csv(args.low_complexity_pairs)
    projection_winners = read_csv(args.projection_winners)
    projection_candidates = read_csv(args.projection_candidates)
    ensure_columns(coords, KEYS + [f'Dim{i}' for i in range(54)], 'coordinates')
    ensure_columns(hierarchy_memberships, KEYS + [MODEL_ID], 'hierarchy memberships')
    require(len(coords) == EXPECTED_EVENTS, f'coordinates rows={len(coords)}')
    require(len(hierarchy_memberships) == EXPECTED_EVENTS, f'hierarchy rows={len(hierarchy_memberships)}')
    require(not coords.duplicated(KEYS).any(), 'duplicate coordinate event keys')
    require(not hierarchy_memberships.duplicated(KEYS).any(), 'duplicate hierarchy event keys')
    joined = hierarchy_memberships[KEYS + [MODEL_ID]].merge(coords[KEYS + [f'Dim{i}' for i in range(54)]], on=KEYS, how='left', validate='one_to_one', indicator=True)
    require(len(joined) == EXPECTED_EVENTS and (joined['_merge'] == 'both').all(), '256-event hierarchy/coordinate join failed')
    joined = joined.drop(columns='_merge')
    require(np.isfinite(joined[DIMS].to_numpy(float)).all(), 'D4 prefix contains missing/nonfinite coordinates')
    fitted = KMeans(n_clusters=K, n_init=10, random_state=args.seed).fit_predict(joined[DIMS].to_numpy(float))
    archived = joined[MODEL_ID].to_numpy(int)
    aligned, label_map, overlap = align_labels(fitted, archived)
    ari = float(adjusted_rand_score(archived, aligned))
    ami = float(adjusted_mutual_info_score(archived, aligned))
    exact_matches = int(np.sum(aligned == archived))
    require(exact_matches == EXPECTED_EVENTS, f'D4/K4 exact label matches={exact_matches}/{EXPECTED_EVENTS}')
    require(abs(ari - 1.0) < 1e-12 and abs(ami - 1.0) < 1e-12, f'D4/K4 reconstruction ARI={ari}, AMI={ami}; expected 1/1')
    d4_coordinates = joined[KEYS + DIMS].copy()
    d4_memberships = d4_coordinates.copy()
    d4_memberships['cluster_k4_d4'] = aligned
    centroid_rows = []
    cluster_size_rows = []
    for cluster in sorted(np.unique(aligned)):
        mask = aligned == cluster
        centroid = d4_memberships.loc[mask, DIMS].mean()
        centroid_rows.append({'cluster': int(cluster), 'n_events': int(mask.sum()), **{dim: float(centroid[dim]) for dim in DIMS}})
        cluster_size_rows.append({'cluster': int(cluster), 'n_events': int(mask.sum()), 'fraction_events': float(mask.mean())})
    centroids = pd.DataFrame(centroid_rows)
    cluster_sizes = pd.DataFrame(cluster_size_rows)
    if 'dimension_name' in spectrum.columns:
        d4_spectrum = spectrum[spectrum['dimension_name'].astype(str).isin(DIMS)].copy()
        if len(d4_spectrum) != 4:
            d4_spectrum = spectrum.iloc[:4].copy()
    else:
        d4_spectrum = spectrum.iloc[:4].copy()
    require(len(d4_spectrum) == 4, 'D4 spectrum prefix must contain exactly 4 rows')
    dcol_grid = find_dimension_column(full_macro_grid, 'full macro grid')
    ensure_columns(full_macro_grid, ['k'], 'full macro grid')
    d4_grid = full_macro_grid[pd.to_numeric(full_macro_grid[dcol_grid], errors='coerce').eq(D)].copy()
    require(len(d4_grid) == 5 and set(d4_grid['k'].astype(int)) == set(range(2, 7)), 'D4 macro grid must contain K=2..6 exactly once')
    ensure_columns(model_summary_all, ['model_id'], 'macro model summary')
    d4_model_metrics = model_summary_all[model_summary_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_model_metrics) == 1, 'expected one D4_K4 macro-model summary row')
    dcol_upstream = find_dimension_column(upstream_selected_k, 'upstream selected-K table')
    kcol_upstream = find_k_column(upstream_selected_k, 'upstream selected-K table')
    d4_selected_k_reference = upstream_selected_k[pd.to_numeric(upstream_selected_k[dcol_upstream], errors='coerce').eq(D)].copy()
    require(len(d4_selected_k_reference) == 1, 'expected exactly one upstream selected-K reference row at D4')
    ensure_columns(feature_associations_all, ['model_id', 'variable'], 'feature associations')
    d4_associations = feature_associations_all[feature_associations_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_associations) == 11 and set(d4_associations['variable'].astype(str)) == set(ACTIVE_VARIABLES), 'D4/K4 feature associations must contain all 11 active variables')
    ensure_columns(feature_cells_all, ['model_id', 'cluster', 'variable', 'level', 'construct_family', 'adjusted_standardized_residual'], 'feature cells')
    d4_cells = feature_cells_all[feature_cells_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_cells) == 260, 'D4/K4 must contain 260 residual-profile cells')
    ensure_columns(participant_support_all, ['model_id', 'cluster', 'n_events', 'participants_represented', 'maximum_single_participant_fraction', 'participant_counts'], 'participant support')
    d4_support = participant_support_all[participant_support_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_support) == 4 and int(d4_support['n_events'].sum()) == EXPECTED_EVENTS, 'D4/K4 participant support must have four clusters totaling 256 events')
    require(int(d4_support['participants_represented'].max()) <= EXPECTED_PARTICIPANTS, 'participant-support table contains impossible participant count')
    d4_profile_summary = build_profile_summary(d4_cells, d4_support)
    ensure_columns(archived_profile_summary_all, ['model_id', 'cluster', 'n_notable_abs_asr_ge_2', 'n_construct_families_with_notable_cells'], 'candidate profile summary')
    archived_d4_profile = archived_profile_summary_all[archived_profile_summary_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(archived_d4_profile) == 4, 'expected four archived D4/K4 profile rows')
    profile_check = d4_profile_summary.merge(archived_d4_profile[['cluster', 'n_notable_abs_asr_ge_2', 'n_construct_families_with_notable_cells']], on='cluster', suffixes=('_canonical', '_archive'), validate='one_to_one')
    require((profile_check['n_notable_abs_asr_ge_2_canonical'] == profile_check['n_notable_abs_asr_ge_2_archive']).all(), 'D4/K4 notable residual-cell counts disagree with hierarchy archive')
    require((profile_check['n_construct_families_with_notable_cells_canonical'] == profile_check['n_construct_families_with_notable_cells_archive']).all(), 'D4/K4 construct-family coverage disagrees with hierarchy archive')
    ensure_columns(heldout_all, ['model_id', 'construct_family'], 'held-out construct alignment')
    d4_heldout = heldout_all[heldout_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_heldout) == 6 and set(d4_heldout['construct_family'].astype(str)) == set(CONSTRUCT_FAMILIES), 'D4/K4 held-out construct alignment must contain all six construct families')
    ensure_columns(lopo_all, ['model_id', 'held_out_participant', 'ari_vs_full_partition_on_retained_events', 'ami_vs_full_partition_on_retained_events'], 'LOPO stability')
    d4_lopo = lopo_all[lopo_all['model_id'].astype(str).eq(MODEL_ID)].copy()
    require(len(d4_lopo) == EXPECTED_PARTICIPANTS, 'D4/K4 LOPO stability must contain five participant-exclusion folds')
    dcol_winners = find_dimension_column(projection_winners, 'projection winners')
    pcol_winners = find_projection_column(projection_winners, 'projection winners')
    d4_projection_winner = projection_winners[pd.to_numeric(projection_winners[dcol_winners], errors='coerce').eq(D)].copy()
    require(len(d4_projection_winner) == 1, 'expected exactly one full-sample projection winner at D4')
    projection_name = str(d4_projection_winner.iloc[0][pcol_winners])
    normalized_projection = projection_name.replace(' ', '').replace('×', 'x').replace('*', 'x')
    require(normalized_projection.lower() == 'dim0xdim3', f'D4 projection winner is {projection_name!r}, expected Dim0xDim3')
    dcol_candidates = find_dimension_column(projection_candidates, 'projection candidates')
    d4_projection_candidates = projection_candidates[pd.to_numeric(projection_candidates[dcol_candidates], errors='coerce').eq(D)].copy()
    require(len(d4_projection_candidates) >= 1, 'no projection candidates found for D4')
    ensure_columns(within_k, ['model_a', 'model_b', 'k', 'ari', 'ami'], 'within-K persistence')
    d4_neighboring = within_k[(within_k['k'].astype(int) == 4) & (within_k['model_a'].astype(str).eq('D4_K4') | within_k['model_b'].astype(str).eq('D4_K4'))].copy()
    require(set(zip(d4_neighboring['model_a'], d4_neighboring['model_b'])) == {('D3_K4', 'D4_K4'), ('D4_K4', 'D5_K4')}, 'D4/K4 neighboring-dimensional evidence must contain D3->D4 and D4->D5')
    ensure_columns(k3_to_k4, ['dimensions', 'parent_model', 'child_model'], 'K3->K4 hierarchy table')
    d4_k3k4 = k3_to_k4[pd.to_numeric(k3_to_k4['dimensions'], errors='coerce').eq(D)].copy()
    require(len(d4_k3k4) == 1 and str(d4_k3k4.iloc[0]['parent_model']) == 'D4_K3' and (str(d4_k3k4.iloc[0]['child_model']) == 'D4_K4'), 'expected exactly one D4/K3 -> D4/K4 hierarchy row')
    ensure_columns(low_pairs, ['model_a', 'model_b'], 'low-complexity pair table')
    d3k3_d4k4 = low_pairs[low_pairs['model_a'].astype(str).eq('D3_K3') & low_pairs['model_b'].astype(str).eq('D4_K4')].copy()
    require(len(d3k3_d4k4) == 1, 'expected one D3/K3 versus D4/K4 low-complexity hierarchy row')
    qa_rows: list[dict[str, object]] = [{'check': 'four-stage chronological 256-event provenance', 'observed': 'matched', 'expected': 'matched', 'passed': True}]

    def qa(check: str, observed, expected, passed: bool) -> None:
        qa_rows.append({'check': check, 'observed': observed, 'expected': expected, 'passed': bool(passed)})
    qa('coordinate SHA-256', observed_coordinate_sha, expected_coordinate_sha, observed_coordinate_sha == expected_coordinate_sha)
    qa('events', len(joined), EXPECTED_EVENTS, len(joined) == EXPECTED_EVENTS)
    qa('D4 prefix dimensions', ','.join(DIMS), 'Dim0,Dim1,Dim2,Dim3', DIMS == ['Dim0', 'Dim1', 'Dim2', 'Dim3'])
    qa('D4/K4 exact reconstructed labels', exact_matches, EXPECTED_EVENTS, exact_matches == EXPECTED_EVENTS)
    qa('D4/K4 reconstruction ARI', ari, 1.0, abs(ari - 1.0) < 1e-12)
    qa('D4/K4 reconstruction AMI', ami, 1.0, abs(ami - 1.0) < 1e-12)
    qa('D4 spectrum prefix rows', len(d4_spectrum), 4, len(d4_spectrum) == 4)
    qa('D4 K2-K6 grid rows', len(d4_grid), 5, len(d4_grid) == 5)
    qa('D4/K4 feature associations', len(d4_associations), 11, len(d4_associations) == 11)
    qa('D4/K4 residual cells', len(d4_cells), 260, len(d4_cells) == 260)
    qa('D4/K4 support clusters', len(d4_support), 4, len(d4_support) == 4)
    qa('D4/K4 held-out construct families', len(d4_heldout), 6, len(d4_heldout) == 6)
    qa('D4/K4 LOPO folds', len(d4_lopo), 5, len(d4_lopo) == 5)
    qa('D4 projection winner', projection_name, 'Dim0xDim3', normalized_projection.lower() == 'dim0xdim3')
    qa('D4 neighboring K4 dimensional rows', len(d4_neighboring), 2, len(d4_neighboring) == 2)
    qa('D4 K3->K4 hierarchy row', len(d4_k3k4), 1, len(d4_k3k4) == 1)
    qa('D3/K3->D4/K4 representative hierarchy row', len(d3k3_d4k4), 1, len(d3k3_d4k4) == 1)
    qa_table = pd.DataFrame(qa_rows)
    require(qa_table['passed'].all(), 'one or more canonical-package QA checks failed')
    tables_dir = args.out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    d4_coordinates.to_csv(tables_dir / 'd4_mca_prefix_row_coordinates_REFIT.csv', index=False)
    d4_spectrum.to_csv(tables_dir / 'd4_mca_prefix_spectrum_REFIT.csv', index=False)
    d4_memberships.to_csv(tables_dir / 'd4_k4_event_memberships_REFIT.csv', index=False)
    pd.DataFrame([{'raw_kmeans_label': raw, 'aligned_archive_cluster': archive_label} for raw, archive_label in sorted(label_map.items())]).to_csv(tables_dir / 'd4_k4_hungarian_label_map_REFIT.csv', index=False)
    pd.DataFrame(overlap, index=[f'raw_{value}' for value in sorted(label_map)], columns=[f'archive_{value}' for value in sorted(np.unique(archived))]).to_csv(tables_dir / 'd4_k4_raw_archive_overlap_REFIT.csv')
    centroids.to_csv(tables_dir / 'd4_k4_cluster_centroids_D4_REFIT.csv', index=False)
    cluster_sizes.to_csv(tables_dir / 'd4_k4_cluster_sizes_REFIT.csv', index=False)
    d4_grid.to_csv(tables_dir / 'd4_k2_k6_model_metrics_REFIT.csv', index=False)
    d4_selected_k_reference.to_csv(tables_dir / 'd4_upstream_selected_k_reference_REFIT.csv', index=False)
    d4_model_metrics.to_csv(tables_dir / 'd4_k4_model_metrics_REFIT.csv', index=False)
    d4_associations.to_csv(tables_dir / 'd4_k4_feature_associations_REFIT.csv', index=False)
    d4_cells.to_csv(tables_dir / 'd4_k4_cluster_feature_cells_REFIT.csv', index=False)
    d4_profile_summary.to_csv(tables_dir / 'd4_k4_cluster_profile_summary_REFIT.csv', index=False)
    d4_support.to_csv(tables_dir / 'd4_k4_participant_support_REFIT.csv', index=False)
    d4_heldout.to_csv(tables_dir / 'd4_k4_heldout_construct_alignment_REFIT.csv', index=False)
    d4_lopo.to_csv(tables_dir / 'd4_k4_lopo_stability_REFIT.csv', index=False)
    d4_projection_winner.to_csv(tables_dir / 'd4_projection_selection_REFIT.csv', index=False)
    d4_projection_candidates.to_csv(tables_dir / 'd4_projection_candidates_REFIT.csv', index=False)
    d4_neighboring.to_csv(tables_dir / 'd4_k4_neighboring_dimension_persistence_REFIT.csv', index=False)
    d4_k3k4.to_csv(tables_dir / 'd4_k3_to_k4_hierarchy_REFIT.csv', index=False)
    d3k3_d4k4.to_csv(tables_dir / 'd3k3_to_d4k4_hierarchy_REFIT.csv', index=False)
    qa_table.to_csv(tables_dir / 'qa_checks.csv', index=False)
    report_lines = ['# D4/K4 canonical REFIT package', '', '## Scientific role', '', 'This script exports the reviewed D4/K4 finer macro representation. Running it records an explicit export choice; it does not automatically select the model. D4 is the Dim0-Dim3 prefix of the authoritative full repaired MCA; it is not a separately fitted MCA and is not the statistically retained MCA dimensionality.', '', '- Full-sample retained geometry: D13', '- Participant-robust geometry reference: D10', '- Primary finer macro representation: D4/K4', '- D4-specific projection: Dim0 × Dim3', f"- D13 projection winner: {refit_provenance['full_sample_projection']}", f"- D10 projection winner: {refit_provenance['participant_robust_projection']}", '- This export does not select the downstream basin-construction plane.', '', '## Provenance reconstruction', '', f'- Authoritative coordinate SHA-256: `{observed_coordinate_sha}`', '- D4/K4 reconstruction: KMeans(n_clusters=4, n_init=10, random_state=42)', '- Clustering coordinates: Dim0, Dim1, Dim2, Dim3', f'- Exact event-label equality after Hungarian relabeling: {exact_matches}/{EXPECTED_EVENTS}', f'- ARI versus current hierarchy D4_K4: {ari:.6f}', f'- AMI versus current hierarchy D4_K4: {ami:.6f}', f'- Raw-to-hierarchy cluster-label map: `{json.dumps(label_map, sort_keys=True)}`', '', '## D4/K4 cluster sizes and support', '', markdown_table(d4_support[['cluster', 'n_events', 'participants_represented', 'maximum_single_participant_fraction', 'participant_counts']]), '', '## D4/K4 construct-alignment evidence', '', markdown_table(d4_heldout[[column for column in ['construct_family', 'observed_pseudo_F', 'observed_R2', 'null_mean_R2', 'null_corrected_R2', 'permutation_p_pseudo_F', 'bh_q_within_model_across_families', 'bh_q_global_across_model_family_grid', 'ari_full_vs_construct_blind'] if column in d4_heldout.columns]]), '', '## D4/K4 proper LOPO participant influence', '', markdown_table(d4_lopo[[column for column in ['held_out_participant', 'n_training_events', 'ari_vs_full_partition_on_retained_events', 'ami_vs_full_partition_on_retained_events'] if column in d4_lopo.columns]]), '', '## D4 projection', '', f'The current full-sample D4 projection winner is **{projection_name}**.', '', markdown_table(d4_projection_winner), '', '## D4/K4 hierarchy evidence', '', 'Neighboring-dimensional K4 persistence:', '', markdown_table(d4_neighboring), '', 'D4/K3 -> D4/K4 resolution split:', '', markdown_table(d4_k3k4), '', 'D3/K3 -> D4/K4 representative hierarchy comparison:', '', markdown_table(d3k3_d4k4), '', '## Descriptive cluster profiles', '', 'No semantic macro-regime names are frozen by this export script. The following profile summaries are derived directly from the archived D4/K4 adjusted-standardized residual cells and participant-support table.', '', markdown_table(d4_profile_summary[['cluster', 'n_events', 'participants_represented', 'n_construct_families_with_notable_cells', 'top_positive_residual_cells', 'top_negative_residual_cells']]), '', '## Scope exclusions', '', 'This package does not rerun MCA, parallel analysis, construct permutations, bootstrap stability, LOPO MCA, projection selection, basin analysis, transition analysis, or successor representation.', '', 'Projection-fidelity rendering and Figure 2A rendering are separate next steps.', '', '## QA', '', f"{int(qa_table['passed'].sum())}/{len(qa_table)} checks passed.", '']
    (args.out_dir / 'D4_K4_CANONICAL_REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')
    metadata = {'analysis': 'D4/K4 canonical REFIT export package', 'dataset_version': DATASET_VERSION, 'n_events': len(joined), 'script_sha256': sha256_file(Path(__file__)), 'refit_provenance': refit_provenance, 'model_choice': 'explicit D4/K4 export; no automatic selection performed', 'basin_projection_selected': False, 'qa_passed': int(qa_table['passed'].sum()), 'qa_total': len(qa_table), 'command': ' '.join(sys.argv), 'elapsed_seconds': time.time() - started, 'scientific_role': 'canonicalize already-completed D4/K4 evidence; no upstream inference rerun', 'coordinate_sha256': observed_coordinate_sha, 'd4_definition': {'type': 'prefix_of_full_repaired_mca', 'dimensions': DIMS, 'separately_refit_mca': False}, 'd4_k4_reconstruction': {'k': K, 'n_init': 10, 'random_state': args.seed, 'exact_event_matches': exact_matches, 'ari_vs_hierarchy_archive': ari, 'ami_vs_hierarchy_archive': ami, 'hungarian_raw_to_hierarchy_map': label_map}, 'projection': {'d4_specific_2d_plane': 'Dim0xDim3', 'full_sample_reference_winner': refit_provenance['full_sample_projection'], 'participant_robust_reference_winner': refit_provenance['participant_robust_projection'], 'd4_full_sample_projection_winner': projection_name, 'projection_selection_rerun': False}, 'export_geometry_roles': {'full_sample_reference': 'D13', 'participant_robust_reference': 'D10', 'primary_finer_macro': 'D4_K4'}, 'input_sha256': {name: sha256_file(path) for name, path in input_paths.items()}, 'rerun_flags': {'mca': False, 'parallel_analysis': False, 'lopo_dimensionality': False, 'construct_blind_mca': False, 'construct_permutations': False, 'bootstrap_stability': False, 'macro_selection': False, 'projection_selection': False, 'basin': False, 'transition': False, 'successor': False}, 'software': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn_version}}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('\n[OK] D4/K4 canonical REFIT package complete.')
    print('[OK] Coordinates match the completed 256-event refit stages.')
    print(f'[OK] Events: {len(joined)} / {EXPECTED_EVENTS}')
    print('[OK] D4 prefix coordinates: Dim0-Dim3')
    print(f'[OK] D4/K4 reconstructed labels: {exact_matches} / {EXPECTED_EVENTS} exact after relabeling')
    print(f'[OK] D4/K4 ARI vs current hierarchy archive: {ari:.6f}')
    print(f'[OK] D4/K4 AMI vs current hierarchy archive: {ami:.6f}')
    print(f'[OK] D4 full-grid K candidates: {len(d4_grid)} / 5')
    print(f'[OK] D4/K4 feature associations: {len(d4_associations)} / 11')
    print(f'[OK] D4/K4 residual-profile cells: {len(d4_cells)} / 260')
    print(f'[OK] D4/K4 held-out construct families: {len(d4_heldout)} / 6')
    print(f'[OK] D4/K4 LOPO folds: {len(d4_lopo)} / 5')
    print(f'[OK] D4 projection winner: {projection_name}')
    print('[OK] No upstream inference or downstream basin/transition/SR analysis was rerun.')
    print(f"[OK] QA: {int(qa_table['passed'].sum())}/{len(qa_table)} passed.")
    print(f'[OK] Output: {args.out_dir.resolve()}')
if __name__ == '__main__':
    main()
