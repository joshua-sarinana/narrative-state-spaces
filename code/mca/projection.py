"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.manifold import trustworthiness
ACTIVE_VARS = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
KEY_COLS = ['participant_id_norm', 'hippo_id_norm']
NULLISH = {'', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing'}
HISTORICAL_NULL_TOKEN = 'none'
EXPECTED_EVENTS = 256
EXPECTED_PARTICIPANT_COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
EXPECTED_Q = 11
EXPECTED_J = 65
EXPECTED_DMAX = 54
DATASET_VERSION = 'chronological_256_v2'
TRUST_K = (5, 10, 20)
TOP_TOL = 1e-12

@dataclass
class MCAFit:
    row_coordinates: np.ndarray
    eigenvalues: np.ndarray
    q: int
    j: int
    dmax: int

def stop(msg: str) -> None:
    raise RuntimeError(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ensure_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        stop(f'{label} missing required columns: {missing}')

def read_input(path: Path, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path, low_memory=False)
    if suffix in {'.xlsx', '.xlsm', '.xls'}:
        return pd.read_excel(path, sheet_name=sheet if sheet else 0)
    stop(f'Unsupported input extension: {suffix}')

def norm_participants(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

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

def fit_exact_mca(active: pd.DataFrame) -> MCAFit:
    n = len(active)
    q = len(ACTIVE_VARS)
    levels_by_var: list[list[str]] = []
    total_j = 0
    for col in ACTIVE_VARS:
        levels = sorted(pd.unique(active[col].astype(str)).tolist())
        levels_by_var.append(levels)
        total_j += len(levels)
    j = total_j
    dmax = min(n - 1, j - q)
    indicator = np.zeros((n, j), dtype=float)
    offset = 0
    rows = np.arange(n)
    for col, levels in zip(ACTIVE_VARS, levels_by_var):
        mapping = {level: i for i, level in enumerate(levels)}
        vals = active[col].astype(str).to_numpy()
        codes = np.fromiter((mapping[v] for v in vals), dtype=int, count=n)
        indicator[rows, offset + codes] = 1.0
        offset += len(levels)
    f = indicator.sum(axis=0)
    standardized = (indicator - f[None, :] / n) / np.sqrt(q * f[None, :])
    u, s, _ = np.linalg.svd(standardized, full_matrices=False)
    eigenvalues = np.maximum(s[:dmax] ** 2, 0.0)
    coords = math.sqrt(n) * u[:, :dmax] * s[None, :dmax]
    return MCAFit(row_coordinates=coords, eigenvalues=eigenvalues, q=q, j=j, dmax=dmax)

def safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if kind == 'pearson':
        return float(pearsonr(a, b).statistic)
    return float(spearmanr(a, b).statistic)

def nearest_neighbors(x: np.ndarray, k: int) -> np.ndarray:
    d = squareform(pdist(x))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind='mergesort')[:, :k]

def neighbor_overlap_metrics(reference: np.ndarray, projected: np.ndarray, k: int) -> tuple[float, float]:
    ref_nn = nearest_neighbors(reference, k)
    prj_nn = nearest_neighbors(projected, k)
    recalls = []
    jaccards = []
    for r, p in zip(ref_nn, prj_nn):
        rs = set(map(int, r))
        ps = set(map(int, p))
        inter = len(rs & ps)
        union = len(rs | ps)
        recalls.append(inter / k)
        jaccards.append(inter / union if union else np.nan)
    return (float(np.mean(recalls)), float(np.nanmean(jaccards)))

def evaluate_projection_pairs(coords: np.ndarray, reference_d: int, context_label: str, held_out_participant: str | None=None) -> pd.DataFrame:
    if reference_d < 2:
        stop('2D projection validation requires reference_d >= 2.')
    if reference_d > coords.shape[1]:
        stop(f'Requested reference D={reference_d} exceeds available coordinate dimensions={coords.shape[1]}.')
    ref = coords[:, :reference_d]
    ref_dist = pdist(ref)
    ref_var = float(np.var(ref, axis=0, ddof=0).sum())
    rows = []
    for a, b in combinations(range(reference_d), 2):
        proj = coords[:, [a, b]]
        proj_dist = pdist(proj)
        tw = {}
        recall = {}
        jac = {}
        for k in TRUST_K:
            tw[k] = float(trustworthiness(ref, proj, n_neighbors=k, metric='euclidean'))
            recall[k], jac[k] = neighbor_overlap_metrics(ref, proj, k)
        proj_name = f'Dim{a}xDim{b}'
        projected_var = float(np.var(proj, axis=0, ddof=0).sum())
        rows.append({'context': context_label, 'held_out_participant': held_out_participant or '', 'reference_dimensions': reference_d, 'projection': proj_name, 'axis_a': a, 'axis_b': b, 'historical_Dim0xDim3': bool(a == 0 and b == 3), 'mean_trustworthiness': float(np.mean([tw[k] for k in TRUST_K])), 'trustworthiness_k5': tw[5], 'trustworthiness_k10': tw[10], 'trustworthiness_k20': tw[20], 'distance_spearman': safe_corr(ref_dist, proj_dist, 'spearman'), 'distance_pearson': safe_corr(ref_dist, proj_dist, 'pearson'), 'retained_coordinate_variance': projected_var / ref_var if ref_var > 0 else np.nan, 'neighbor_recall_k5': recall[5], 'neighbor_recall_k10': recall[10], 'neighbor_recall_k20': recall[20], 'neighbor_jaccard_k5': jac[5], 'neighbor_jaccard_k10': jac[10], 'neighbor_jaccard_k20': jac[20]})
    return pd.DataFrame(rows)

def rank_projection_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        stop('Projection candidate table is empty.')
    out = df.copy()
    out = out.sort_values(['mean_trustworthiness', 'distance_spearman', 'distance_pearson', 'projection'], ascending=[False, False, False, True], kind='mergesort').reset_index(drop=True)
    out['descriptive_rank'] = np.arange(1, len(out) + 1)
    max_tw = float(out['mean_trustworthiness'].max())
    tie = out[(max_tw - out['mean_trustworthiness']).abs() <= TOP_TOL].copy()
    tie = tie.sort_values(['distance_spearman', 'distance_pearson', 'projection'], ascending=[False, False, True], kind='mergesort')
    winner_projection = str(tie.iloc[0]['projection'])
    out['selected'] = out['projection'].eq(winner_projection)
    winner = out.loc[out['selected']].iloc[0]
    return (out, winner)

def parse_lopo_retained(spec: str) -> dict[str, int]:
    out = {}
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '=' not in token:
            stop('--lopo-retained must look like p01=11,p02=10,p03=11,p04=12,p05=12')
        p, d = token.split('=', 1)
        p = p.strip().lower()
        out[p] = int(d.strip())
    return out

def winner_row(winner: pd.Series, analytical_role: str) -> dict:
    return {'analytical_role': analytical_role, 'context': winner['context'], 'held_out_participant': winner['held_out_participant'], 'reference_dimensions': int(winner['reference_dimensions']), 'projection': winner['projection'], 'axis_a': int(winner['axis_a']), 'axis_b': int(winner['axis_b']), 'mean_trustworthiness': float(winner['mean_trustworthiness']), 'trustworthiness_k5': float(winner['trustworthiness_k5']), 'trustworthiness_k10': float(winner['trustworthiness_k10']), 'trustworthiness_k20': float(winner['trustworthiness_k20']), 'distance_spearman': float(winner['distance_spearman']), 'distance_pearson': float(winner['distance_pearson']), 'retained_coordinate_variance': float(winner['retained_coordinate_variance']), 'historical_Dim0xDim3': bool(winner['historical_Dim0xDim3'])}

def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No rows._'

    def fmt(x):
        if pd.isna(x):
            return ''
        if isinstance(x, (float, np.floating)):
            return f'{float(x):.6f}'
        return str(x).replace('|', '\\|').replace('\n', ' ')
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join((fmt(row[c]) for c in cols)) + ' |')
    return '\n'.join(lines)

def load_upstream_reference(args) -> tuple[dict, dict]:
    paths = {'metadata': args.upstream_dir / 'RUN_METADATA.json', 'decision': args.upstream_dir / 'tables' / 'decision_summary.json', 'qa': args.upstream_dir / 'tables' / 'qa_checks.csv', 'parallel': args.upstream_dir / 'tables' / 'parallel_analysis_5000_summary.csv', 'lopo': args.upstream_dir / 'tables' / 'lopo_retention_summary_5000_each.csv'}
    for name, path in paths.items():
        if not path.is_file():
            stop(f'Required upstream {name} file not found: {path}')
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    decision = json.loads(paths['decision'].read_text(encoding='utf-8'))
    input_hash = sha256_file(args.input)
    for label, record in [('metadata', metadata), ('decision', decision)]:
        if record.get('input_sha256') != input_hash:
            stop(f'Input workbook does not match the upstream {label} input hash.')
        if record.get('dataset_version') != DATASET_VERSION:
            stop(f'Upstream {label} is not the {DATASET_VERSION} refit.')
    if metadata.get('input_sheet') != args.input_sheet:
        stop('--input-sheet must match RUN_METADATA.json from the upstream run.')
    expected = {'n_events': EXPECTED_EVENTS, 'active_variables_q': EXPECTED_Q, 'indicator_categories_j': EXPECTED_J, 'maximum_dimensions': EXPECTED_DMAX, 'full_parallel_analysis_permutations': 5000, 'lopo_parallel_permutations_per_fold': 5000, 'seed_full': args.seed}
    for key, value in expected.items():
        if decision.get(key) != value:
            stop(f'Upstream {key}={decision.get(key)!r}; expected {value!r}.')
    if metadata.get('active_variables') != ACTIVE_VARS:
        stop('Upstream active variables do not match this projection script.')
    qa = pd.read_csv(paths['qa'])
    if qa.empty or not qa['passed'].eq(True).all():
        stop('Upstream structural/numerical QA has not passed completely.')
    if decision.get('qa_passed') != len(qa) or decision.get('qa_total') != len(qa):
        stop('Upstream QA counts disagree with decision_summary.json.')
    full_d = int(decision['full_sample_parallel_retained_dimensions'])
    robust_d = int(decision['participant_robust_min_lopo_dimensions'])
    lopo = {str(p): int(d) for p, d in decision['lopo_retained_dimensions'].items()}
    if set(lopo) != set(EXPECTED_PARTICIPANT_COUNTS):
        stop('Upstream LOPO participant identifiers do not match this refit.')
    if not 2 <= robust_d <= full_d <= EXPECTED_DMAX:
        stop('Projection validation requires 2 <= robust D <= full D <= 54.')
    if any((d < 2 or d > EXPECTED_DMAX for d in lopo.values())):
        stop('Upstream LOPO dimensions are outside the supported range.')
    if robust_d != min(lopo.values()):
        stop('Upstream robust D does not equal the minimum LOPO retained D.')
    pa = pd.read_csv(paths['parallel'])
    ensure_columns(pa, ['axis', 'observed_eigenvalue', 'null_q95'], 'upstream parallel analysis')
    if pa['axis'].tolist() != list(range(1, EXPECTED_DMAX + 1)):
        stop('Upstream parallel-analysis axes are incomplete or out of order.')
    leading = int(np.cumprod(pa['observed_eigenvalue'].to_numpy() > pa['null_q95'].to_numpy()).sum())
    if leading != full_d:
        stop('Upstream full D disagrees with the saved parallel-analysis table.')
    folds = pd.read_csv(paths['lopo'])
    ensure_columns(folds, ['held_out_participant', 'parallel_retained_dimensions'], 'upstream LOPO')
    if len(folds) != 5 or folds['held_out_participant'].duplicated().any():
        stop('Upstream LOPO table must have one row per participant.')
    saved_lopo = dict(zip(folds['held_out_participant'], folds['parallel_retained_dimensions'].astype(int)))
    if saved_lopo != lopo:
        stop('Upstream LOPO table disagrees with decision_summary.json.')
    if args.full_d is not None and args.full_d != full_d:
        stop(f'--full-d {args.full_d} disagrees with the upstream choice {full_d}.')
    if args.robust_d is not None and args.robust_d != robust_d:
        stop(f'--robust-d {args.robust_d} disagrees with the upstream choice {robust_d}.')
    if args.lopo_retained is not None and parse_lopo_retained(args.lopo_retained) != lopo:
        stop('--lopo-retained disagrees with the upstream LOPO choices.')
    args.full_d = full_d
    args.robust_d = robust_d
    args.lopo_retained = ','.join((f'{p}={d}' for p, d in sorted(lopo.items())))
    provenance = {'directory': str(args.upstream_dir.resolve()), 'files': {name: {'path': str(path.resolve()), 'sha256': sha256_file(path)} for name, path in paths.items()}, 'input_sha256': input_hash, 'upstream_script_sha256': metadata.get('script_sha256')}
    print(f'[projection] Matched 256-event upstream input; full D={full_d}, robust D={robust_d}, LOPO={args.lopo_retained}', flush=True)
    return (decision, provenance)

def main() -> int:
    ap = argparse.ArgumentParser(description='Geometry-only MCA 2D projection validation for the 256-event refit.')
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--input-sheet', default='Sheet1')
    ap.add_argument('--upstream-dir', required=True, type=Path, help='Completed 256-event MCA upstream output directory.')
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--full-d', type=int, default=None, help='Optional override; must match upstream retention.')
    ap.add_argument('--robust-d', type=int, default=None, help='Optional override; must match upstream retention.')
    ap.add_argument('--lopo-retained', default=None, help='Optional override; must match upstream LOPO retention.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not args.input.exists():
        stop(f'Input not found: {args.input}')
    if args.seed != 42:
        stop('Canonical projection run requires --seed 42.')
    upstream_decision, upstream_provenance = load_upstream_reference(args)
    lopo_retained = parse_lopo_retained(args.lopo_retained)
    df = read_input(args.input, args.input_sheet).reset_index(drop=True)
    ensure_columns(df, KEY_COLS + ACTIVE_VARS, args.input.name)
    participants = norm_participants(df['participant_id_norm'])
    participant_counts = {str(k): int(v) for k, v in participants.value_counts().sort_index().items()}
    if len(df) != EXPECTED_EVENTS:
        stop(f'Events {len(df)} != {EXPECTED_EVENTS}.')
    if participant_counts != EXPECTED_PARTICIPANT_COUNTS:
        stop(f'Participant counts {participant_counts} != {EXPECTED_PARTICIPANT_COUNTS}.')
    if df.duplicated(KEY_COLS).any():
        stop('Duplicate participant/event keys detected.')
    active = normalize_active_table(df)
    full_fit = fit_exact_mca(active)
    if full_fit.q != EXPECTED_Q:
        stop(f'Q={full_fit.q} != {EXPECTED_Q}.')
    if full_fit.j != EXPECTED_J:
        stop(f'J={full_fit.j} != {EXPECTED_J}.')
    if full_fit.dmax != EXPECTED_DMAX:
        stop(f'dmax={full_fit.dmax} != {EXPECTED_DMAX}.')
    total_inertia_expected = (full_fit.j - full_fit.q) / full_fit.q
    if abs(full_fit.eigenvalues.sum() - total_inertia_expected) >= 1e-10:
        stop('Exact MCA total inertia check failed.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    print('[projection] Input structure and exact MCA inertia passed; evaluating axis pairs.', flush=True)
    full_candidate_tables = []
    full_winner_rows = []
    for d in range(2, args.full_d + 1):
        cand = evaluate_projection_pairs(full_fit.row_coordinates, reference_d=d, context_label='full_sample')
        ranked, winner = rank_projection_table(cand)
        full_candidate_tables.append(ranked)
        role = 'PRIMARY_FULL_SAMPLE' if d == args.full_d else 'PARTICIPANT_ROBUST_REFERENCE' if d == args.robust_d else 'DESCRIPTIVE_DIMENSIONAL_PREFIX'
        full_winner_rows.append(winner_row(winner, role))
        print(f"[projection] full sample D={d}: {winner['projection']} (mean TW={winner['mean_trustworthiness']:.6f}, Spearman={winner['distance_spearman']:.6f}, Pearson={winner['distance_pearson']:.6f})", flush=True)
    full_candidates = pd.concat(full_candidate_tables, ignore_index=True)
    full_winners = pd.DataFrame(full_winner_rows)
    primary_rank = full_candidates[full_candidates['reference_dimensions'] == args.full_d].copy()
    robust_rank = full_candidates[full_candidates['reference_dimensions'] == args.robust_d].copy()
    primary_winner = full_winners[full_winners['reference_dimensions'] == args.full_d].iloc[0]
    robust_winner = full_winners[full_winners['reference_dimensions'] == args.robust_d].iloc[0]
    common_tables = []
    common_winners = []
    fold_tables = []
    fold_winners = []
    p_arr = participants.to_numpy()
    for held_out in sorted(pd.unique(p_arr)):
        train = p_arr != held_out
        fold_df = df.loc[train].reset_index(drop=True)
        fold_active = normalize_active_table(fold_df)
        fold_fit = fit_exact_mca(fold_active)
        cand_common = evaluate_projection_pairs(fold_fit.row_coordinates, reference_d=args.robust_d, context_label='lopo_common_robust_D', held_out_participant=held_out)
        ranked_common, winner_common = rank_projection_table(cand_common)
        common_tables.append(ranked_common)
        common_winners.append(winner_row(winner_common, f'LOPO_COMMON_D{args.robust_d}_INFLUENCE'))
        fold_d = int(lopo_retained[held_out])
        cand_fold = evaluate_projection_pairs(fold_fit.row_coordinates, reference_d=fold_d, context_label='lopo_fold_retained_D', held_out_participant=held_out)
        ranked_fold, winner_fold = rank_projection_table(cand_fold)
        fold_tables.append(ranked_fold)
        fold_winners.append(winner_row(winner_fold, 'LOPO_FOLD_RETAINED_D_INFLUENCE'))
        print(f"[projection] LOPO exclude {held_out}: common D{args.robust_d} winner={winner_common['projection']}; fold-retained D{fold_d} winner={winner_fold['projection']}", flush=True)
    lopo_common_candidates = pd.concat(common_tables, ignore_index=True)
    lopo_common_winners = pd.DataFrame(common_winners)
    lopo_fold_candidates = pd.concat(fold_tables, ignore_index=True)
    lopo_fold_winners = pd.DataFrame(fold_winners)
    primary_name = str(primary_winner['projection'])
    robust_name = str(robust_winner['projection'])
    common_match_n = int(lopo_common_winners['projection'].eq(primary_name).sum())
    fold_match_n = int(lopo_fold_winners['projection'].eq(primary_name).sum())
    ref_summary = pd.DataFrame([{'reference': 'primary_full_sample', 'dimensions': args.full_d, 'selected_projection': primary_name, 'mean_trustworthiness': float(primary_winner['mean_trustworthiness']), 'distance_spearman': float(primary_winner['distance_spearman']), 'distance_pearson': float(primary_winner['distance_pearson']), 'retained_coordinate_variance': float(primary_winner['retained_coordinate_variance'])}, {'reference': 'participant_robust_full_sample', 'dimensions': args.robust_d, 'selected_projection': robust_name, 'mean_trustworthiness': float(robust_winner['mean_trustworthiness']), 'distance_spearman': float(robust_winner['distance_spearman']), 'distance_pearson': float(robust_winner['distance_pearson']), 'retained_coordinate_variance': float(robust_winner['retained_coordinate_variance'])}])
    expected_full_rows = math.comb(args.full_d + 1, 3)
    expected_primary_rows = math.comb(args.full_d, 2)
    expected_robust_rows = math.comb(args.robust_d, 2)
    expected_lopo_common_rows = 5 * expected_robust_rows
    expected_lopo_fold_rows = sum((math.comb(d, 2) for d in lopo_retained.values()))
    qa_rows = [('event_rows', len(df), EXPECTED_EVENTS), ('participants', participants.nunique(), 5), ('active_variables', full_fit.q, EXPECTED_Q), ('active_category_levels', full_fit.j, EXPECTED_J), ('nonzero_axes', full_fit.dmax, EXPECTED_DMAX), ('full_reference_D', args.full_d, upstream_decision['full_sample_parallel_retained_dimensions']), ('robust_reference_D', args.robust_d, upstream_decision['participant_robust_min_lopo_dimensions']), (f'full_D2_D{args.full_d}_candidate_rows', len(full_candidates), expected_full_rows), (f'primary_D{args.full_d}_candidate_rows', len(primary_rank), expected_primary_rows), (f'robust_D{args.robust_d}_candidate_rows', len(robust_rank), expected_robust_rows), (f'lopo_common_D{args.robust_d}_candidate_rows', len(lopo_common_candidates), expected_lopo_common_rows), ('lopo_fold_retained_candidate_rows', len(lopo_fold_candidates), expected_lopo_fold_rows), (f'full_winners_D2_D{args.full_d}', len(full_winners), args.full_d - 1), ('lopo_common_winners', len(lopo_common_winners), 5), ('lopo_fold_winners', len(lopo_fold_winners), 5), ('finite_full_mean_trustworthiness', int(np.isfinite(full_candidates['mean_trustworthiness']).sum()), len(full_candidates)), ('finite_lopo_common_mean_trustworthiness', int(np.isfinite(lopo_common_candidates['mean_trustworthiness']).sum()), len(lopo_common_candidates)), ('finite_lopo_fold_mean_trustworthiness', int(np.isfinite(lopo_fold_candidates['mean_trustworthiness']).sum()), len(lopo_fold_candidates))]
    qa = pd.DataFrame([{'check': name, 'observed': observed, 'expected': expected, 'passed': bool(observed == expected)} for name, observed, expected in qa_rows])
    if not qa['passed'].all():
        stop('Projection QA failed:\n' + qa.loc[~qa['passed']].to_string(index=False))
    full_candidates.to_csv(tables_dir / f'full_sample_projection_candidates_D2_D{args.full_d}.csv', index=False)
    full_winners.to_csv(tables_dir / 'full_sample_projection_winners_by_D.csv', index=False)
    primary_rank.to_csv(tables_dir / f'full_sample_primary_D{args.full_d}_ranking.csv', index=False)
    robust_rank.to_csv(tables_dir / f'full_sample_robust_D{args.robust_d}_ranking.csv', index=False)
    lopo_common_candidates.to_csv(tables_dir / f'lopo_common_D{args.robust_d}_projection_candidates.csv', index=False)
    lopo_common_winners.to_csv(tables_dir / f'lopo_common_D{args.robust_d}_projection_winners.csv', index=False)
    lopo_fold_candidates.to_csv(tables_dir / 'lopo_fold_retained_projection_candidates.csv', index=False)
    lopo_fold_winners.to_csv(tables_dir / 'lopo_fold_retained_projection_winners.csv', index=False)
    ref_summary.to_csv(tables_dir / 'projection_reference_summary.csv', index=False)
    qa.to_csv(tables_dir / 'qa_checks.csv', index=False)
    report = []
    report.append('# REFIT 256: MCA geometry-only projection validation')
    report.append('')
    report.append('Projection selection used only geometry-preservation criteria. No K-means, basin, transition, routing, successor, or recurrence result entered the selection.')
    report.append('')
    report.append('## Primary and participant-robust full-sample references')
    report.append('')
    report.append(md_table(ref_summary))
    report.append('')
    report.append(f'## Full-sample winners across D=2..{args.full_d}')
    report.append('')
    report.append(md_table(full_winners[['reference_dimensions', 'analytical_role', 'projection', 'mean_trustworthiness', 'distance_spearman', 'distance_pearson', 'retained_coordinate_variance']]))
    report.append('')
    report.append(f'## Proper MCA-refit LOPO: common D{args.robust_d}')
    report.append('')
    report.append(md_table(lopo_common_winners[['held_out_participant', 'reference_dimensions', 'projection', 'mean_trustworthiness', 'distance_spearman', 'distance_pearson']]))
    report.append('')
    report.append('## Proper MCA-refit LOPO: fold-retained dimensionality')
    report.append('')
    report.append(md_table(lopo_fold_winners[['held_out_participant', 'reference_dimensions', 'projection', 'mean_trustworthiness', 'distance_spearman', 'distance_pearson']]))
    report.append('')
    report.append('## Influence-diagnostic comparison')
    report.append('')
    report.append(f'- Primary full-sample D{args.full_d} projection: **{primary_name}**')
    report.append(f'- Full-sample D{args.robust_d} projection: **{robust_name}**')
    report.append(f'- Proper LOPO common-D{args.robust_d} folds selecting the same projection name as the full-sample D{args.full_d} winner: **{common_match_n}/5**.')
    report.append(f'- Proper LOPO fold-retained-D folds selecting the same projection name as the full-sample D{args.full_d} winner: **{fold_match_n}/5**.')
    report.append('')
    report.append('LOPO projection names are influence diagnostics because MCA axes can rotate after refitting. They do not vote on or override the full-sample geometry-only winner.')
    report.append('')
    (args.out_dir / 'PROJECTION_VALIDATION_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    metadata = {'dataset_version': DATASET_VERSION, 'script_path': str(Path(__file__).resolve()), 'script_sha256': sha256_file(Path(__file__).resolve()), 'input_sheet': args.input_sheet, 'n_events': len(df), 'participant_event_counts': participant_counts, 'upstream_reference': upstream_provenance, 'qa_passed': int(qa['passed'].sum()), 'qa_total': len(qa), 'run_utc': datetime.now(timezone.utc).isoformat(), 'command': ' '.join(sys.argv), 'input': str(args.input.resolve()), 'input_sha256': sha256_file(args.input), 'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn_version, 'active_variables': ACTIVE_VARS, 'full_reference_D': args.full_d, 'participant_robust_reference_D': args.robust_d, 'lopo_retained_D': lopo_retained, 'full_sample_tested_D': list(range(2, args.full_d + 1)), 'trustworthiness_neighbors': list(TRUST_K), 'primary_selection_metric': 'mean_trustworthiness_k5_k10_k20', 'top_tolerance': TOP_TOL, 'tie_break_order': ['distance_spearman_desc', 'distance_pearson_desc', 'projection_name_asc'], 'historical_Dim0xDim3_used_as_selection_prior': False, 'kmeans_used': False, 'basin_used': False, 'transition_used': False, 'successor_used': False, 'primary_projection': primary_name, 'robust_full_sample_projection': robust_name, f'lopo_common_D{args.robust_d}_same_name_as_primary_n': common_match_n, 'lopo_fold_retained_same_name_as_primary_n': fold_match_n}
    (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\n[OK] Geometry-only projection validation complete.')
    print(f'[OK] Events: {len(df)} / {EXPECTED_EVENTS}')
    print(f'[OK] Active variables: {full_fit.q} / {EXPECTED_Q}')
    print(f'[OK] Active category levels: {full_fit.j} / {EXPECTED_J}')
    print(f'[OK] Nonzero MCA axes: {full_fit.dmax} / {EXPECTED_DMAX}')
    print(f'[RESULT] Primary full-sample D{args.full_d} projection: {primary_name}')
    print(f'[RESULT] Participant-robust full-sample D{args.robust_d} projection: {robust_name}')
    print(f'[RESULT] LOPO common-D{args.robust_d} same projection name as primary: {common_match_n}/5')
    print(f'[RESULT] LOPO fold-retained-D same projection name as primary: {fold_match_n}/5')
    print('[OK] No K-means, basin, transition, routing, or SR analysis was run.')
    print(f"[OK] QA: {int(qa['passed'].sum())}/{len(qa)} passed.")
    print(f'[OK] Output: {args.out_dir.resolve()}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
