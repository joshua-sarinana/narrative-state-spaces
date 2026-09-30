"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
for _name in ('numpy', 'pandas', 'scipy'):
    if importlib.util.find_spec(_name) is None:
        raise SystemExit(f'Missing {_name}. Use the existing REFIT .venv interpreter. No packages were installed.')
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter
HERE = Path(__file__).resolve().parent
SOLUTIONS = ['D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39']
KEYS = ['participant_id_norm', 'hippo_id_norm']
COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
FLOW = {'bins': 25, 'pad_frac': 0.05, 'flow_sigma': 0.75, 'min_flow_count': 0.05, 'gaussian_boundary_mode': 'nearest'}
SOURCES = {'events': ('outputs/macro_transition/fixed_D4K4_memberships_with_D13_coordinates_REFIT.csv', 'computed at invocation'), 'edges': ('outputs/macro_transition/authoritative_event_transitions_D4K4_REFIT.csv', 'computed at invocation'), 'assignments': ('outputs/assignments/REFIT_Multiple_Event_Assignments.csv', 'computed at invocation'), 'independent_edges': ('outputs/assignments/inputs/event_edges.csv', 'computed at invocation'), 'independent_events': ('outputs/assignments/inputs/new_events.csv', 'computed at invocation')}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def load_inputs(root):
    tables = {}
    for role, (relative, expected) in SOURCES.items():
        path = root / relative
        require(path.is_file(), f'Missing input: {path}. Run the upstream stages first.')
        require(sha256(path) == expected, f'Input differs from the reviewed REFIT version: {path}. Stop and review the version; do not bypass this check.')
        tables[role] = pd.read_csv(path, keep_default_na=False)
    events = tables['events'].copy()
    require(len(events) == 256 and (not events.duplicated(KEYS).any()), 'Expected 256 unique event keys.')
    require(events.groupby(KEYS[0]).size().to_dict() == COUNTS, 'Unexpected participant counts.')
    require(events.groupby('cluster_k4_d4').size().to_dict() == {1: 133, 2: 59, 3: 17, 4: 47}, 'Unexpected macro-cluster counts.')
    key_index = pd.MultiIndex.from_frame(events[KEYS])
    for role in ('assignments', 'independent_events'):
        frame = tables[role]
        require(not frame.duplicated(KEYS).any(), f'Duplicate keys: {role}')
        require(set(map(tuple, frame[KEYS].to_numpy())) == set(key_index), f'Mismatched event keys: {role}')
        tables[role] = frame.set_index(KEYS).loc[key_index].reset_index()
    coords = [f'Dim{i}' for i in range(13)]
    require(np.isfinite(events[coords].to_numpy(float)).all(), 'Nonfinite coordinates.')
    require(np.allclose(events[coords[:4]], tables['independent_events'][coords[:4]], atol=1e-12, rtol=0), 'Independent coordinates disagree.')
    for solution in SOLUTIONS:
        labels = tables['assignments'][solution].to_numpy()
        k = int(solution.split('_')[1][1:])
        require(set(labels) == set(range(1, k + 1)), f'Invalid basin labels: {solution}')
        events[solution] = labels.astype(int)
    edge_keys = [KEYS[0], 'source_event', 'target_event']
    edges = tables['edges']
    require(len(edges) == 251 and (not edges.duplicated(edge_keys).any()), 'Expected 251 unique links.')
    require(set(map(tuple, edges[edge_keys].to_numpy())) == set(map(tuple, tables['independent_edges'][edge_keys].to_numpy())), 'Independent link sources disagree.')
    lookup = {key: i for i, key in enumerate(key_index)}
    chains, people = ([], sorted(COUNTS))
    for participant in people:
        group = edges.loc[edges[KEYS[0]] == participant]
        ids = set(events.loc[events[KEYS[0]] == participant, KEYS[1]])
        srcs, dsts = (list(group.source_event), list(group.target_event))
        require(len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts)), 'Chain has multiple successors or predecessors.')
        require(set(srcs) <= ids and set(dsts) <= ids and (len(group) == len(ids) - 1), 'Chain contains missing or cross-participant events.')
        starts = ids - set(dsts)
        require(len(starts) == 1, 'Chain does not have one starting event.')
        successors = dict(zip(srcs, dsts))
        cursor, order, visited = (next(iter(starts)), [], set())
        while cursor not in visited:
            visited.add(cursor)
            order.append(lookup[participant, cursor])
            if cursor not in successors:
                break
            cursor = successors[cursor]
        require(visited == ids and cursor not in successors, 'Chain is cyclic or disconnected.')
        chains.append(np.asarray(order, dtype=int))
    for row in edges.itertuples(index=False):
        require(events.iloc[lookup[row.participant_id_norm, row.source_event]].cluster_k4_d4 == row.source_cluster and events.iloc[lookup[row.participant_id_norm, row.target_event]].cluster_k4_d4 == row.target_cluster, 'Edge macro labels disagree.')
    return (events, chains, people)

def expected_deltas(xy, chains, source):
    expected = np.full_like(xy, np.nan, dtype=float)
    for chain in chains:
        inside = chain[1:-1]
        m = len(inside)
        if m == 0:
            expected[chain[0]] = xy[chain[-1]] - xy[chain[0]]
            continue
        total = xy[inside].sum(axis=0)
        expected[chain[0]] = total / m - xy[chain[0]]
        expected[inside] = (total - xy[inside] + xy[chain[-1]]) / m - xy[inside]
    require(np.isfinite(expected[source]).all(), 'Missing null expectation for a source.')
    return expected[source]

def flow_operator(xy, source):
    bins = FLOW['bins']
    lo, hi = (xy.min(axis=0), xy.max(axis=0))
    pad = FLOW['pad_frac'] * (hi - lo)
    xe = np.linspace(lo[0] - pad[0], hi[0] + pad[0], bins + 1)
    ye = np.linspace(lo[1] - pad[1], hi[1] + pad[1], bins + 1)
    sx, sy = xy[source].T
    ix = np.searchsorted(xe, sx, side='right') - 1
    iy = np.searchsorted(ye, sy, side='right') - 1
    require(np.all((ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)), 'Source outside flow grid.')
    impulses = np.zeros((bins, bins, len(source)))
    impulses[iy, ix, np.arange(len(source))] = 1.0
    kernel = gaussian_filter(impulses, (FLOW['flow_sigma'], FLOW['flow_sigma'], 0), mode=FLOW['gaussian_boundary_mode'])
    counts = kernel.sum(axis=2).ravel()
    operator = np.divide(kernel.reshape(-1, len(source)), counts[:, None], out=np.zeros((bins * bins, len(source))), where=counts[:, None] > 0)
    mask = counts >= FLOW['min_flow_count']
    weights = np.where(mask, counts, 0)
    weights /= weights.sum()
    xx, yy = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)
    grid = pd.DataFrame({'Dim0': xx.ravel(), 'Dim3': yy.ravel(), 'smoothed_source_count': counts, 'flow_visible': mask, 'statistic_weight': weights})
    return (operator, weights, grid, xe, ye)

def direct_field(xy, source, delta, xe, ye):

    def hist(weight=None):
        return np.histogram2d(xy[source, 0], xy[source, 1], bins=(xe, ye), weights=weight)[0].T
    counts = gaussian_filter(hist(), FLOW['flow_sigma'], mode='nearest')
    field = []
    for axis in range(2):
        sums = gaussian_filter(hist(delta[:, axis]), FLOW['flow_sigma'], mode='nearest')
        field.append(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).ravel())
    return np.column_stack(field)

def cmi_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    n_p = counts.sum(axis=(1, 2), keepdims=True)
    row = counts.sum(axis=2, keepdims=True)
    col = counts.sum(axis=1, keepdims=True)
    denominator = row * col
    ratio = np.divide(counts * n_p, denominator, out=np.ones_like(counts), where=denominator > 0)
    terms = np.zeros_like(counts)
    positive = counts > 0
    terms[positive] = counts[positive] * np.log(ratio[positive])
    return float(terms.sum() / counts.sum())

def routing_counts(participant, source_macro, destination_basin, k, n_people):
    flat = (participant * 4 + source_macro) * k + destination_basin
    return np.bincount(flat, minlength=n_people * 4 * k).reshape(n_people, 4, k)

def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind='stable')
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected

def successor_draw(rng, chains, source, n_events):
    lookup = np.full(n_events, -1, dtype=int)
    for chain in chains:
        shuffled = np.r_[chain[0], rng.permutation(chain[1:-1]), chain[-1]]
        lookup[shuffled[:-1]] = shuffled[1:]
    return lookup[source]

def incremental_prediction(events, source, target, participant, people, output):
    macro = events.cluster_k4_d4.to_numpy(int) - 1
    summaries, folds, support, predictions = ([], [], [], [])
    for solution in SOLUTIONS:
        k = int(solution.split('_')[1][1:])
        labels = events[solution].to_numpy(int) - 1
        c, b, dest = (macro[source], labels[source], labels[target])
        all_gains, all_supported, all_zeros = ([], [], [])
        for p, name in enumerate(people):
            train, test = (participant != p, participant == p)
            global_counts = np.bincount(dest[train], minlength=k)
            q = (global_counts + 1 / k) / (train.sum() + 1)
            macro_counts = np.bincount(c[train] * k + dest[train], minlength=4 * k).reshape(4, k)
            cb_counts = np.bincount((c[train] * k + b[train]) * k + dest[train], minlength=4 * k * k).reshape(4, k, k)
            n_c = macro_counts.sum(axis=1)
            n_cb = cb_counts.sum(axis=2)
            p0 = (macro_counts + 10 * q) / (n_c[:, None] + 10)
            p1 = (cb_counts + 10 * p0[:, None, :]) / (n_cb[:, :, None] + 10)
            require(np.allclose(p0.sum(axis=1), 1) and np.allclose(p1.sum(axis=2), 1), 'Invalid prediction probabilities.')
            people_cb = np.zeros((4, k), dtype=int)
            for training_p in range(len(people)):
                if training_p == p:
                    continue
                sel = participant == training_p
                people_cb += np.bincount(c[sel] * k + b[sel], minlength=4 * k).reshape(4, k) > 0
            ntrain = n_cb[c[test], b[test]]
            npart = people_cb[c[test], b[test]]
            supported = (ntrain >= 8) & (npart >= 3)
            log0 = np.log(p0[c[test], dest[test]])
            log1 = np.log(p1[c[test], b[test], dest[test]])
            gains = log1 - log0
            all_gains.extend(gains.tolist())
            all_supported.extend(supported.tolist())
            all_zeros.extend((ntrain == 0).tolist())
            folds.append(dict(solution=solution, held_out_participant=name, transitions=int(test.sum()), macro_only_log_score=float(log0.mean()), macro_plus_basin_log_score=float(log1.mean()), gain_nats_per_transition=float(gains.mean()), supported_transitions=int(supported.sum()), zero_training_support_transitions=int((ntrain == 0).sum())))
            for idx, s in enumerate(np.flatnonzero(test)):
                predictions.append(dict(solution=solution, participant_id_norm=name, source_event=events.iloc[source[s]][KEYS[1]], target_event=events.iloc[target[s]][KEYS[1]], source_macro=int(c[s] + 1), source_basin=int(b[s] + 1), target_basin=int(dest[s] + 1), macro_only_log_score=float(log0[idx]), macro_plus_basin_log_score=float(log1[idx]), gain_nats=float(gains[idx]), training_source_combination_links=int(ntrain[idx]), training_source_combination_participants=int(npart[idx]), heuristic_support=bool(supported[idx])))
            for ci in range(4):
                for bi in range(k):
                    support.append(dict(solution=solution, held_out_participant=name, source_macro=ci + 1, source_basin=bi + 1, training_links=int(n_cb[ci, bi]), training_participants=int(people_cb[ci, bi]), held_out_links=int(np.sum(test & (c == ci) & (b == bi))), heuristic_support=bool(n_cb[ci, bi] >= 8 and people_cb[ci, bi] >= 3)))
        solution_folds = [f for f in folds if f['solution'] == solution]
        summaries.append(dict(solution=solution, dimensions=solution.split('_')[0], basins=k, macro_solution='D4_K4', transitions=len(source), gain_nats_per_transition=float(np.mean(all_gains)), participants_with_positive_gain=sum((f['gain_nats_per_transition'] > 0 for f in solution_folds)), participants=len(people), supported_transitions=sum(all_supported), supported_fraction=float(np.mean(all_supported)), zero_training_support_transitions=sum(all_zeros), inference='descriptive; no p-value'))
    for name, records in [('incremental_prediction_summary', summaries), ('incremental_prediction_by_participant', folds), ('incremental_prediction_support', support), ('heldout_transition_predictions', predictions)]:
        pd.DataFrame(records).to_csv(output / f'{name}.csv', index=False)
    return pd.DataFrame(summaries)

def p_text(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'

def write_report(output, results, incremental, permutations, seed):
    field = results.iloc[0]
    lines = ['# REFIT executed results', '', f'256 events; 251 original links; five participants; {permutations:,} endpoint-fixed within-participant order permutations; seed {seed}.', '', '## Empirical field: Dim0 × Dim3', '', f'Global count-weighted squared departure from the exact null-expected field: T={field.observed_statistic:.6f}; null mean={field.null_mean:.6f}; raw p={p_text(field.p_raw)}; Holm p across six tests={p_text(field.p_holm_all_six)}.', '', '## Current macro-cluster → next basin', '', 'Macro-clusters are fixed at D4/K4 for every row. D2 local assignments use Dim0 × Dim3. Information measures are in nats.', '', '| Assignment | Observed CMI | Null mean | Excess CMI | Raw p | Holm p, five | Holm p, six | Significant, six |', '|---|---:|---:|---:|---:|---:|---:|:---:|']
    for r in results.iloc[1:].itertuples():
        lines.append(f"| {r.test} | {r.observed_statistic:.3f} | {r.null_mean:.3f} | {r.excess_over_null_mean:.3f} | {p_text(r.p_raw)} | {p_text(r.p_holm_macro_five)} | {p_text(r.p_holm_all_six)} | {('Yes' if r.significant_holm_six else 'No')} |")
    lines += ['', "These p-values compare the observed narratives with random interior order, conditional on each participant's event set, fixed endpoints, frozen geometry and selected assignments. They do not separate specific routing mechanisms from temporal persistence or establish population generalization. The six-test correction covers only this package's new tests.", '', '## Incremental basin information: descriptive held-out check', '', 'Positive gain favors adding current basin identity to current macro identity when predicting the next basin. Transition probabilities are estimated using the other four participants; geometry and labels are frozen from the full dataset. These comparisons have no significance test.', '', '| Assignment | Log-score gain, nats/link | Participants with gain | Links meeting training-support heuristic | Zero-support links |', '|---|---:|---:|---:|---:|']
    for r in incremental.itertuples():
        lines.append(f'| {r.solution} | {r.gain_nats_per_transition:.3f} | {r.participants_with_positive_gain}/5 | {r.supported_transitions}/251 ({100 * r.supported_fraction:.1f}%) | {r.zero_training_support_transitions} |')
    lines += ['', 'The support heuristic requires at least eight training links for a source macro–basin combination, from at least three training participants. It is not a guarantee of precision. Inspect participant-level scores and the complete support audit before making an incremental-prediction claim.', '', 'The field test applies to the pooled Dim0 × Dim3 field. It supplies no separate significance tests for context-specific S3 fields, individual arrows, basins, or streamlines. No inference is made about continuous trajectories or attractors.', '', 'Full-precision values, null distributions, field grids, routing counts, held-out scores, input hashes, settings and software versions are saved alongside this report. See METHODS.md for the protocol.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

def configure_paths(args):
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing path configuration: {config_path}. Keep input_paths.json beside the script.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Expected input_paths.json schema_version 1.')
    global SOURCES
    expected_keys = set(SOURCES)
    if set(config.get('inputs', {})) != expected_keys:
        raise ValueError(f'Configured input roles must be: {sorted(expected_keys)}')
    base = config_path.parent
    args.refit_root = (args.refit_root if args.refit_root is not None else base / config['refit_root_relative_to_config']).resolve()
    args.output_dir = (args.output_dir if args.output_dir is not None else base / config['output_dir_relative_to_config']).resolve()
    args.input_paths = {role: (args.refit_root / rel).resolve() for role, rel in config['inputs'].items()}
    args.config_path = config_path
    SOURCES = {role: (relative, sha256(args.refit_root / relative)) for role, relative in config['inputs'].items()}
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refit-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--permutations', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--check', action='store_true', help='Validate inputs only; no outputs written.')
    parser.add_argument('--config', type=Path, default=HERE / 'input_paths.json', help='JSON configuration with input and output paths.')
    args = configure_paths(parser.parse_args())
    require(args.permutations >= 99, 'Use at least 99 permutations (default 1,000,000).')
    start = time.monotonic()
    events, chains, people = load_inputs(args.refit_root.resolve())
    print('Validated all five input hashes, 256 event keys, 251 links, five complete chains, coordinates and labels.', flush=True)
    if args.check:
        return
    output = args.output_dir.resolve()
    require(output != args.refit_root.resolve(), 'Output directory must not be the REFIT root.')
    output.mkdir(parents=True, exist_ok=True)
    source = np.concatenate([chain[:-1] for chain in chains])
    target = np.concatenate([chain[1:] for chain in chains])
    participant = np.concatenate([np.full(len(chain) - 1, i) for i, chain in enumerate(chains)])
    xy = events[['Dim0', 'Dim3']].to_numpy(float)
    macro = events.cluster_k4_d4.to_numpy(int) - 1
    source_macro = macro[source]
    operator, weights, grid, xe, ye = flow_operator(xy, source)
    mean_delta = expected_deltas(xy, chains, source)
    observed_delta = xy[target] - xy[source]
    observed_field, null_field = (operator @ observed_delta, operator @ mean_delta)
    direct = direct_field(xy, source, observed_delta, xe, ye)
    require(np.allclose(observed_field, direct, rtol=0, atol=1e-12), 'Accelerated field does not match the Figure 2 histogram implementation.')
    reference_error = float(np.max(np.abs(observed_field - direct)))
    weighted_operator = operator * np.sqrt(weights[:, None])
    gram = weighted_operator.T @ weighted_operator
    centered = observed_delta - mean_delta
    observed_flow = float(np.einsum('ia,ij,ja->', centered, gram, centered, optimize=True))
    require(np.isclose(observed_flow, np.sum(weights[:, None] * (observed_field - null_field) ** 2), rtol=0, atol=1e-12), 'Quadratic flow statistic failed verification.')
    for axis, dim in enumerate((0, 3)):
        grid[f'observed_mean_delta_Dim{dim}'] = observed_field[:, axis]
        grid[f'null_expected_mean_delta_Dim{dim}'] = null_field[:, axis]
        grid[f'observed_minus_null_Dim{dim}'] = observed_field[:, axis] - null_field[:, axis]
    grid.to_csv(output / 'flow_grid_Dim0_Dim3.csv', index=False)
    labels = [events[s].to_numpy(int) - 1 for s in SOLUTIONS]
    ks = [int(s.split('_')[1][1:]) for s in SOLUTIONS]
    observed_mi, routing = ([], [])
    for solution, basin, k in zip(SOLUTIONS, labels, ks):
        counts = routing_counts(participant, source_macro, basin[target], k, len(people))
        observed_mi.append(cmi_from_counts(counts))
        for p, name in list(enumerate(people)) + [(len(people), 'pooled')]:
            matrix = counts[p] if p < len(people) else counts.sum(axis=0)
            for c in range(4):
                n = int(matrix[c].sum())
                for b in range(k):
                    routing.append(dict(solution=solution, participant=name, source_macro=c + 1, target_basin=b + 1, count=int(matrix[c, b]), source_row_links=n, row_probability=float(matrix[c, b] / n) if n else None))
    pd.DataFrame(routing).to_csv(output / 'macro_to_next_basin_counts.csv', index=False)
    observed = np.r_[observed_flow, observed_mi]
    null = np.empty((args.permutations, 6), dtype=float)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    batch_size = 100
    for first in range(0, args.permutations, batch_size):
        last = min(first + batch_size, args.permutations)
        draws = np.array([successor_draw(rng, chains, source, len(events)) for _ in range(last - first)])
        if first == 0:
            for draw in draws:
                for p in range(len(people)):
                    sel = participant == p
                    require(np.array_equal(np.sort(draw[sel]), np.sort(target[sel])), 'Permutation changed destination margins.')
        deviations = xy[draws] - xy[source][None, :, :] - mean_delta[None, :, :]
        null[first:last, 0] = np.einsum('bia,ij,bja->b', deviations, gram, deviations, optimize=True)
        for row, draw in enumerate(draws, start=first):
            for s, (basin, k) in enumerate(zip(labels, ks), start=1):
                null[row, s] = cmi_from_counts(routing_counts(participant, source_macro, basin[draw], k, len(people)))
        if last % 50000 == 0 or last == args.permutations:
            print(f'Completed {last:,}/{args.permutations:,} shared permutations ({time.monotonic() - start:.1f} s).', flush=True)
    require(np.isfinite(null).all(), 'Nonfinite null statistics.')
    exceed = (null >= observed[None, :] - 1e-12 * np.maximum(1, np.abs(observed))[None, :]).sum(axis=0)
    p_values = (exceed + 1) / (args.permutations + 1)
    adjusted = holm(p_values)
    names = ['flow_Dim0_Dim3'] + SOLUTIONS
    results = pd.DataFrame({'test': names, 'statistic': ['count_weighted_squared_field_departure'] + ['conditional_mutual_information_nats'] * 5, 'observed_statistic': observed, 'null_mean': null.mean(axis=0), 'excess_over_null_mean': observed - null.mean(axis=0), 'null_sd': null.std(axis=0, ddof=1), 'null_q025': np.quantile(null, 0.025, axis=0), 'null_q975': np.quantile(null, 0.975, axis=0), 'null_exceedances': exceed, 'permutations': args.permutations, 'p_raw': p_values, 'p_holm_macro_five': np.r_[np.nan, holm(p_values[1:])], 'p_holm_all_six': adjusted, 'significant_holm_six': adjusted < 0.05})
    results['p_raw_monte_carlo_se'] = np.sqrt(p_values * (1 - p_values) / (args.permutations + 1))
    results.to_csv(output / 'primary_test_results.csv', index=False)
    np.savez_compressed(output / 'permutation_null_statistics.npz', statistics=null, columns=np.asarray(names), seed=args.seed)
    incremental = incremental_prediction(events, source, target, participant, people, output)
    chronology = []
    for name, chain in zip(people, chains):
        for position, index in enumerate(chain):
            chronology.append(dict(participant_id_norm=name, position_zero_based=position, hippo_id_norm=events.iloc[index][KEYS[1]], fixed_endpoint=position in (0, len(chain) - 1)))
    pd.DataFrame(chronology).to_csv(output / 'validated_chronology.csv', index=False)
    write_report(output, results, incremental, args.permutations, args.seed)
    metadata = dict(protocol='METHODS.md; exploratory; fixed endpoints; frozen assignments', generated_utc=datetime.now(timezone.utc).isoformat(), events=len(events), links=len(source), participants=COUNTS, macro_solution='D4_K4', local_solutions=SOLUTIONS, D2_axes=['Dim0', 'Dim3'], flow=FLOW, flow_reference_max_abs_error=reference_error, permutations=args.permutations, seed=args.seed, rng='numpy.random.Generator(PCG64)', p_value='(upper-tail exceedances + 1)/(permutations + 1)', p_tie_tolerance='1e-12 * max(1, abs(observed))', primary_multiplicity='Holm across all six; also Holm across five macro-basin tests', prediction_global_prior_mass=1, prediction_conditional_prior_mass=10, prediction_inference='descriptive only; frozen full-sample representations', versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__), source_files={k: dict(path_from_0_REFIT=v[0], sha256=v[1]) for k, v in SOURCES.items()}, script_sha256=sha256(__file__), output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'RUN_METADATA.json'}, elapsed_seconds=time.monotonic() - start)
    if (HERE / 'METHODS.md').is_file():
        metadata['methods_sha256'] = sha256(HERE / 'METHODS.md')
    metadata['path_configuration_sha256'] = sha256(args.config_path)
    write_json(output / 'RUN_METADATA.json', metadata)
    print(results[['test', 'observed_statistic', 'p_raw', 'p_holm_all_six', 'significant_holm_six']].to_string(index=False), flush=True)
    print(f"Results: {output / 'RESULTS.md'}", flush=True)
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
