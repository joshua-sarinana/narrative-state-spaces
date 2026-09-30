"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
DATASET_VERSION = 'chronological_256_v2'
N_EVENTS = 256
KEYS = ['participant_id_norm', 'hippo_id_norm']
EXPECTED_PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
ACTIVE_VARIABLES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f'QA FAIL: {message}')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key in KEYS:
        require(key in out.columns, f'missing key column {key}')
        out[key] = out[key].astype(str).str.strip().str.lower()
    return out

def align_labels(fitted: np.ndarray, archived: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    fit_ids = np.array(sorted(np.unique(fitted)))
    archive_ids = np.array(sorted(np.unique(archived)))
    require(len(fit_ids) == k and len(archive_ids) == k, f'expected {k} labels')
    overlap = np.array([[int(np.sum((fitted == fi) & (archived == ai))) for ai in archive_ids] for fi in fit_ids], dtype=int)
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {int(fit_ids[i]): int(archive_ids[j]) for i, j in zip(row_ind, col_ind)}
    aligned = np.array([mapping[int(x)] for x in fitted], dtype=int)
    return (aligned, mapping, overlap)

def residual_text(rows: pd.DataFrame, n: int=8) -> str:
    return ' | '.join((f'{r.variable}={r.level} (ASR={r.adjusted_standardized_residual:.2f})' for r in rows.head(n).itertuples()))

def table(df: pd.DataFrame) -> str:

    def cell(value: object) -> str:
        if pd.isna(value):
            return ''
        if isinstance(value, (float, np.floating)):
            rendered = f'{float(value):.3g}' if 0 < abs(float(value)) < 1e-06 else f'{float(value):.6f}'
        else:
            rendered = str(value)
        return rendered.replace('|', '\\|').replace('\n', ' ')
    headers = [str(column).replace('|', '\\|') for column in df.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join((cell(value) for value in row)) + ' |')
    return '\n'.join(lines)

def profile_summary(model_id: str, cells: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    records = []
    support_by_cluster = support.set_index('cluster')
    for cluster, group in cells.groupby('cluster', sort=True):
        notable = group[group['adjusted_standardized_residual'].abs() >= 2]
        positive = notable[notable['adjusted_standardized_residual'] >= 2].sort_values(['adjusted_standardized_residual', 'variable', 'level'], ascending=[False, True, True])
        negative = notable[notable['adjusted_standardized_residual'] <= -2].sort_values(['adjusted_standardized_residual', 'variable', 'level'], ascending=[True, True, True])
        s = support_by_cluster.loc[cluster]
        counts = json.loads(s['participant_counts'])
        dominant, dominant_n = max(counts.items(), key=lambda x: (x[1], x[0]))
        records.append({'model_id': model_id, 'cluster': int(cluster), 'descriptive_name': f'Cluster {int(cluster)}', 'n_events': int(s['n_events']), 'participants_represented': int(s['participants_represented']), 'dominant_participant': dominant, 'dominant_participant_n': int(dominant_n), 'maximum_single_participant_fraction': float(s['maximum_single_participant_fraction']), 'n_notable_abs_asr_ge_2': int(len(notable)), 'n_construct_families_with_notable_cells': int(notable['construct_family'].nunique()), 'construct_families_with_notable_cells': ' | '.join(sorted(notable['construct_family'].unique())), 'top_positive_residual_cells': residual_text(positive), 'top_negative_residual_cells': residual_text(negative)})
    return pd.DataFrame(records)

def pair_stats(labels: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = list(combinations(labels, 2))
    stats = []
    flows = []
    for a, b in pairs:
        la, lb = (labels[a], labels[b])
        stats.append({'model_a': a, 'model_b': b, 'ari': adjusted_rand_score(la, lb), 'ami': adjusted_mutual_info_score(la, lb)})
        tab = pd.crosstab(pd.Series(la, name='cluster_a'), pd.Series(lb, name='cluster_b'))
        for ca in tab.index:
            for cb in tab.columns:
                n = int(tab.loc[ca, cb])
                flows.append({'model_a': a, 'cluster_a': int(ca), 'model_b': b, 'cluster_b': int(cb), 'event_count': n, 'fraction_of_cluster_a': n / int((la == ca).sum()), 'fraction_of_cluster_b': n / int((lb == cb).sum())})
    return (pd.DataFrame(stats), pd.DataFrame(flows))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mca', 'macro', 'hierarchy', 'canonical', 'out'):
        parser.add_argument(f'--{name}-dir', type=Path, required=True)
    return parser.parse_args()

def check(rows, name, condition, observed='matched', expected='matched'):
    require(bool(condition), f'{name}: observed={observed}, expected={expected}')
    rows.append({'check': name, 'observed': observed, 'expected': expected, 'passed': True})

def validate_sources(args, qa):
    directories = {name: getattr(args, f'{name}_dir') for name in ('mca', 'macro', 'hierarchy', 'canonical')}
    metadata, stage_files = ({}, {})
    for name, directory in directories.items():
        path = directory / 'RUN_METADATA.json'
        qpath = directory / 'tables/qa_checks.csv'
        require(path.is_file() and qpath.is_file(), f'missing {name} metadata or QA')
        record = json.loads(path.read_text(encoding='utf-8'))
        checks = pd.read_csv(qpath)
        require(record.get('dataset_version') == DATASET_VERSION, f'wrong {name} dataset')
        passed = not checks.empty and checks['passed'].astype(str).str.lower().isin(['true', '1']).all()
        if name != 'mca':
            passed = passed and record.get('n_events') == N_EVENTS
            passed = passed and record.get('qa_passed') == record.get('qa_total') == len(checks)
        check(qa, f'{name} completed-stage QA', passed, int(checks['passed'].astype(str).str.lower().isin(['true', '1']).sum()), len(checks))
        metadata[name] = record
        stage_files[name] = {'path': str(path.resolve()), 'sha256': sha256(path), 'qa_sha256': sha256(qpath)}
    mca, macro, hierarchy, canonical = [metadata[n] for n in directories]
    previous = canonical['refit_provenance']['stage_metadata']
    for name in ('mca', 'macro', 'hierarchy'):
        require(stage_files[name]['sha256'] == previous[name + '_dir']['sha256'] and stage_files[name]['qa_sha256'] == previous[name + '_dir']['qa_sha256'], f'{name} stage differs from the validated D4/K4 package')
    paths = {'full_coordinates': args.mca_dir / 'tables/mca_full_54d_row_coordinates.csv', 'upstream_selected_k': args.mca_dir / 'tables/kmeans_consensus_selected_k_by_dimension.csv', 'hierarchy_memberships': args.hierarchy_dir / 'tables/reconstructed_event_memberships.csv', 'd4_memberships': args.canonical_dir / 'tables/d4_k4_event_memberships_REFIT.csv', 'feature_associations': args.macro_dir / 'tables/full_geometry_feature_associations.csv', 'feature_cells': args.macro_dir / 'tables/full_geometry_cluster_feature_cells.csv', 'participant_support': args.macro_dir / 'tables/participant_support_by_model.csv', 'macro_model_summary': args.macro_dir / 'tables/macro_model_summary.csv', 'decision_summary': args.mca_dir / 'tables/decision_summary.json'}
    for key, path in paths.items():
        require(path.is_file(), f'missing {key}: {path}')
    hashes = {key: sha256(path) for key, path in paths.items()}
    for key in canonical['input_sha256'].keys() & paths.keys():
        require(hashes[key] == canonical['input_sha256'][key], f'changed canonical source: {key}')
    for key in ('feature_associations', 'feature_cells', 'participant_support', 'macro_model_summary'):
        require(hashes[key] == hierarchy['macro_input_sha256'][paths[key].stem], f'changed hierarchy source: {key}')
    require(hashes['decision_summary'] == macro['upstream_reference']['files']['decision']['sha256'], 'changed MCA decision summary')
    require(hashes['full_coordinates'] == macro['input_full_coordinates_sha256'] == hierarchy['full_coordinates_sha256'] == canonical['coordinate_sha256'], 'coordinate fingerprints disagree')
    decision = json.loads(paths['decision_summary'].read_text(encoding='utf-8'))
    require(decision.get('dataset_version') == DATASET_VERSION and decision.get('n_events') == N_EVENTS, 'wrong MCA decision dataset')
    require(decision.get('qa_passed') == decision.get('qa_total') == len(pd.read_csv(args.mca_dir / 'tables/qa_checks.csv')), 'MCA decision QA counts disagree')
    event_hash = mca['input_sha256']
    require(event_hash == decision['input_sha256'] == macro['input_event_table_sha256'] == hierarchy['refit_provenance']['input_event_table_sha256'] == canonical['refit_provenance']['input_event_table_sha256'], 'event-input fingerprints disagree')
    require(hierarchy['refit_provenance']['macro_metadata_sha256'] == stage_files['macro']['sha256'], 'hierarchy uses a different macro run')
    require(canonical['export_geometry_roles']['primary_finer_macro'] == 'D4_K4', 'this comparison requires the D4/K4 canonical package')
    require(hierarchy['kmeans_reconstruction']['n_init'] == 10 and hierarchy['kmeans_reconstruction']['random_state'] == 42, 'hierarchy reconstruction settings differ')
    check(qa, 'current stage and source fingerprints', True)
    selected = pd.read_csv(paths['upstream_selected_k'])
    roles = [('full_sample', 'full_sample_parallel_retained_dimensions', 'full_sample_agnostic_reference', 'full_sample_reference_winner'), ('participant_robust', 'participant_robust_min_lopo_dimensions', 'participant_robust_agnostic_reference', 'participant_robust_reference_winner')]
    specs = {}
    for role, decision_key, model_key, projection_key in roles:
        d = int(decision[decision_key])
        row = selected[selected.dimensions.eq(d)]
        require(len(row) == 1 and str(row.iloc[0]['selected']).lower() == 'true', f'missing selected K for D{d}')
        k = int(row.iloc[0]['k'])
        model_id = f'D{d}_K{k}'
        require(model_id == macro[model_key] == hierarchy['refit_provenance'][model_key] == canonical['refit_provenance'][model_key], f'{role} reference selections disagree')
        require(model_id not in specs, 'references coincide; review duplicate export roles')
        specs[model_id] = {'dimensions': d, 'k': k, 'cluster_col': f'cluster_k{k}_d{d}', 'role': role, 'projection_winner': canonical['projection'][projection_key]}
    check(qa, 'reference models match upstream selections', True, '; '.join(specs), '; '.join(specs))
    provenance = {'stage_metadata': stage_files, 'input_event_table_sha256': event_hash, 'input_files': {key: str(path.resolve()) for key, path in paths.items()}, 'input_sha256': hashes, 'projection_record': canonical['projection']}
    return (paths, specs, provenance)

def write_report(path, model_id, spec, summary, associations, metrics, pair_df, flow_df):
    label = model_id.replace('_', '/')
    comparison = pair_df[(pair_df.model_a == model_id) | (pair_df.model_b == model_id)]
    flows = flow_df[(flow_df.model_a == 'D4_K4') & (flow_df.model_b == model_id)]
    counts = flows.pivot(index='cluster_a', columns='cluster_b', values='event_count').reset_index()
    counts = counts.rename(columns={'cluster_a': 'D4/K4 raw cluster', **{c: f'{label} raw C{c}' for c in counts.columns if c != 'cluster_a'}})
    lines = [f'# {label} reference macro characterization', '', f"Role: {spec['role'].replace('_', ' ')} geometry reference. D={spec['dimensions']} and K={spec['k']} were selected in the completed MCA analysis. D4/K4 remains the primary finer descriptive representation.", '', f"Clustering uses Dim0 through Dim{spec['dimensions'] - 1}. The recorded projection winner for this reference is {spec['projection_winner']}; projection selection was not rerun. The D4-specific plane is recorded separately. This export does not choose a basin projection.", '', 'Raw cluster numbers are specific to each model. Profile names have not been assigned; the residual cells below provide the descriptive evidence.', '', f'Silhouette {metrics.silhouette:.6f}; bootstrap ARI {metrics.bootstrap_stability_ari:.6f}; mean LOPO ARI {metrics.lopo_ari_mean:.6f}; mean held-out-family null-corrected R² {metrics.heldout_family_mean_null_corrected_R2:.6f}; globally significant held-out families {int(metrics.heldout_family_count_global_q05)}/6.', '', table(summary[['cluster', 'n_events', 'participants_represented', 'maximum_single_participant_fraction', 'n_construct_families_with_notable_cells']]), '']
    for r in summary.itertuples():
        lines += [f'Cluster {r.cluster}: {r.n_events} events; {r.participants_represented}/5 participants.', '', f'Enrichments: {r.top_positive_residual_cells}', '', f'Depletions: {r.top_negative_residual_cells}', '']
    lines += ['Cross-model partition comparisons:', '', table(comparison), '', 'Event cross-tabulation against D4/K4 (counts, not temporal transitions):', '', table(counts), '', 'These independently fitted partitions need not be strictly nested. The separate event-flow table provides both source- and destination-cluster fractions.', '', 'Eleven-field descriptive associations:', '', table(associations[['variable', 'construct_family', 'bias_corrected_cramers_v', 'adjusted_mutual_information', 'bh_q_global']]), '', "Residual thresholds describe the fitted partition; they are not independent tests of each cluster's existence. Whole-partition construct and LOPO metrics are copied from the completed macro grid.", '', f'Sources: macro-grid association, residual-cell, participant-support and model-summary tables for {model_id}; hierarchy membership column {model_id}; validated D4/K4 export.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    args = parse_args()
    qa = []
    paths, specs, provenance = validate_sources(args, qa)
    coords = normalize_keys(pd.read_csv(paths['full_coordinates']))
    archive = normalize_keys(pd.read_csv(paths['hierarchy_memberships']))
    d4 = normalize_keys(pd.read_csv(paths['d4_memberships']))
    for name, frame in (('coordinates', coords), ('hierarchy', archive), ('D4 export', d4)):
        require(len(frame) == N_EVENTS and (not frame.duplicated(KEYS).any()), f'{name} must have 256 unique event keys')
        require(frame.participant_id_norm.value_counts().to_dict() == EXPECTED_PARTICIPANT_COUNTS, f'{name} participant event counts differ')
    require(set(specs).issubset(archive.columns), 'selected reference model missing from hierarchy')
    check(qa, '256 unique event keys and participant counts in each input', True, N_EVENTS, N_EVENTS)
    max_dims = [f'Dim{i}' for i in range(max((s['dimensions'] for s in specs.values())))]
    require(set(max_dims).issubset(coords.columns), 'missing MCA prefix coordinates')
    merged = archive[KEYS + ['D4_K4'] + list(specs)].merge(coords[KEYS + max_dims], on=KEYS, how='left', validate='one_to_one')
    require(len(merged) == N_EVENTS and np.isfinite(merged[max_dims].to_numpy(float)).all(), 'coordinate join failed')
    check(qa, 'complete finite coordinate join', True, len(merged), N_EVENTS)
    d4_dims = [f'Dim{i}' for i in range(4)]
    joined = merged.merge(d4[KEYS + d4_dims + ['cluster_k4_d4']], on=KEYS, how='left', validate='one_to_one', suffixes=('', '_d4_export'))
    require(joined['cluster_k4_d4'].notna().all() and np.array_equal(joined.D4_K4.to_numpy(int), joined.cluster_k4_d4.to_numpy(int)), 'canonical D4/K4 labels disagree with hierarchy')
    require(np.allclose(joined[d4_dims], joined[[c + '_d4_export' for c in d4_dims]], rtol=0, atol=1e-12), 'D4 export coordinates disagree with full MCA prefix')
    check(qa, 'D4/K4 membership and coordinates match current hierarchy and MCA', True, len(joined), N_EVENTS)
    aligned_labels = {'D4_K4': merged.D4_K4.to_numpy(int)}
    associations_all = pd.read_csv(paths['feature_associations'])
    cells_all = pd.read_csv(paths['feature_cells'])
    support_all = pd.read_csv(paths['participant_support'])
    model_metrics = pd.read_csv(paths['macro_model_summary']).set_index('model_id', verify_integrity=True)
    pending = {}
    model_qas = {}
    for model_id, spec in specs.items():
        dims = [f'Dim{i}' for i in range(spec['dimensions'])]
        fitted = KMeans(n_clusters=spec['k'], n_init=10, random_state=42).fit_predict(merged[dims].to_numpy(float))
        archived = merged[model_id].to_numpy(int)
        aligned, mapping, overlap = align_labels(fitted, archived, spec['k'])
        ari = float(adjusted_rand_score(archived, aligned))
        ami = float(adjusted_mutual_info_score(archived, aligned))
        check(qa, f'{model_id} exact reconstructed labels and ARI/AMI=1', np.array_equal(aligned, archived) and abs(ari - 1) < 1e-12 and (abs(ami - 1) < 1e-12), int(np.sum(aligned == archived)), N_EVENTS)
        membership = merged[KEYS + dims].copy()
        membership[spec['cluster_col']] = aligned
        aligned_labels[model_id] = aligned
        associations = associations_all[associations_all.model_id == model_id].copy()
        cells = cells_all[cells_all.model_id == model_id].copy()
        support = support_all[support_all.model_id == model_id].copy()
        check(qa, f'{model_id} eleven distinct feature associations', len(associations) == 11 and set(associations.variable) == set(ACTIVE_VARIABLES), len(associations), 11)
        check(qa, f'{model_id} complete residual cells', len(cells) == 65 * spec['k'] and (not cells.duplicated(['cluster', 'variable', 'level']).any()) and (set(cells.variable) == set(ACTIVE_VARIABLES)) and (set(cells.cluster) == set(aligned)) and cells.groupby('cluster').size().eq(65).all(), len(cells), 65 * spec['k'])
        require(len(support) == spec['k'] and (not support.cluster.duplicated().any()), f'{model_id} support rows invalid')
        for row in support.itertuples():
            participants = membership.loc[membership[spec['cluster_col']].eq(row.cluster), 'participant_id_norm']
            counts = participants.value_counts().to_dict()
            require(counts == json.loads(row.participant_counts) and len(participants) == row.n_events and (len(counts) == row.participants_represented), f'{model_id} cluster {row.cluster} support mismatch')
            require(np.isclose(row.maximum_single_participant_fraction, max(counts.values()) / len(participants)), f'{model_id} participant fraction mismatch')
        require(model_id in model_metrics.index and int(model_metrics.loc[model_id, 'dimensions']) == spec['dimensions'] and (int(model_metrics.loc[model_id, 'k']) == spec['k']), f'{model_id} metric row mismatch')
        check(qa, f'{model_id} participant support and model metrics', int(support.n_events.sum()) == N_EVENTS, int(support.n_events.sum()), N_EVENTS)
        pending[model_id] = {'membership': membership, 'associations': associations, 'cells': cells, 'support': support, 'summary': profile_summary(model_id, cells, support), 'mapping': mapping, 'overlap': overlap}
        model_qas[model_id] = {'exact_label_matches': int(np.sum(aligned == archived)), 'ari': ari, 'ami': ami, 'hungarian_raw_to_hierarchy_map': mapping}
    pair_df, flow_df = pair_stats(aligned_labels)
    for (a, b), flow in flow_df.groupby(['model_a', 'model_b']):
        require(int(flow.event_count.sum()) == N_EVENTS, f'{a}/{b} cross-tab total mismatch')
        require(np.allclose(flow.groupby('cluster_a').fraction_of_cluster_a.sum(), 1) and np.allclose(flow.groupby('cluster_b').fraction_of_cluster_b.sum(), 1), 'cross-tab fractions do not sum to 1')
    check(qa, 'all three cross-model tables cover 256 events', len(pair_df) == 3, len(pair_df), 3)
    tables_dir = args.out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    for model_id, spec in specs.items():
        record = pending[model_id]
        prefix = model_id.lower()
        outputs = {'event_memberships': record['membership'], 'feature_associations': record['associations'], 'cluster_feature_cells': record['cells'], 'participant_support': record['support'], 'cluster_profile_summary': record['summary'], 'model_metrics': model_metrics.loc[[model_id]].reset_index(), 'hungarian_label_map': pd.DataFrame([{'raw_kmeans_label': x, 'aligned_archive_cluster': y} for x, y in sorted(record['mapping'].items())]), 'raw_archive_overlap': pd.DataFrame(record['overlap'])}
        for name, frame in outputs.items():
            frame.to_csv(tables_dir / f'{prefix}_{name}_REFIT.csv', index=False)
        write_report(args.out_dir / f'{model_id}_PROFILE_REPORT.md', model_id, spec, record['summary'], record['associations'], model_metrics.loc[model_id], pair_df, flow_df)
    pair_df.to_csv(tables_dir / 'macro_reference_pair_similarity_REFIT.csv', index=False)
    flow_df.to_csv(tables_dir / 'macro_reference_event_flow_REFIT.csv', index=False)
    reference_rows = [{'model_id': key, **spec} for key, spec in specs.items()]
    pd.DataFrame(reference_rows).to_csv(tables_dir / 'reference_model_registry_REFIT.csv', index=False)
    qa_frame = pd.DataFrame(qa)
    qa_frame.to_csv(tables_dir / 'qa_checks.csv', index=False)
    metadata = {'analysis': 'selected reference macro characterization', 'dataset_version': DATASET_VERSION, 'n_events': N_EVENTS, 'script_sha256': sha256(Path(__file__)), 'command': ' '.join(sys.argv), 'primary_finer_macro': 'D4_K4', 'reference_models': specs, 'provenance': provenance, 'kmeans': {'n_init': 10, 'random_state': 42}, 'reconstruction': model_qas, 'pair_similarity': pair_df.to_dict(orient='records'), 'qa_passed': len(qa), 'qa_total': len(qa), 'semantic_names_assigned': False, 'basin_projection_selected': False, 'cross_model_event_flow_is_temporal_transition': False, 'rerun_flags': {name: False for name in ['mca', 'parallel_analysis', 'k_selection', 'construct_permutations', 'bootstrap_stability', 'lopo', 'projection_selection', 'basin', 'transition', 'successor']}, 'software': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__}}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    (args.out_dir / 'REFERENCE_MACRO_REPORT.md').write_text('\n'.join(['# Reference macro characterization — REFIT 256', '', 'D4/K4 remains the primary finer descriptive representation. The models below characterize the full-sample and participant-robust geometric references using their previously selected K values.', '', table(pd.DataFrame(reference_rows)), '', table(pair_df), '', 'Reconstructed labels match the hierarchy exactly for all 256 events in each reference (ARI=AMI=1). Model-specific reports contain residual profiles and event cross-tabulations against D4/K4.', '', 'Cross-model event-flow counts describe membership overlap, not temporal transitions. Raw cluster labels are model-specific. No semantic label mapping is implied.', '', 'Projection winners are recorded from the validated canonical package. This export does not select the downstream basin plane.', '', f'QA: {len(qa)}/{len(qa)} passed. No upstream inference or downstream basin, transition or successor analysis was rerun.', '']), encoding='utf-8')
    print('[OK] Reference models:', ', '.join(specs))
    for model_id in specs:
        print(f'[OK] {model_id}: 256/256 exact labels; ARI=AMI=1')
        support = pending[model_id]['support'].sort_values('cluster')
        print(f'[OK] {model_id} cluster sizes: ' + ', '.join((str(int(x)) for x in support.n_events)))
    print(f'[OK] QA: {len(qa)}/{len(qa)} passed.')
    print('[OK] No upstream inference or downstream basin, transition, or SR analysis was rerun.')
    print('[OK] Output:', args.out_dir.resolve())
if __name__ == '__main__':
    main()
