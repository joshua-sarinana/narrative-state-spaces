"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
import time
for _package in ('numpy', 'pandas', 'scipy'):
    if importlib.util.find_spec(_package) is None:
        raise SystemExit(f'Missing {_package}. See the dependency command in the notes at the top of this script.')
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta
HERE = Path(__file__).resolve().parent
DIMS = ['Dim0', 'Dim1', 'Dim2', 'Dim3']
KEYS = ['participant_id_norm', 'hippo_id_norm']
EDGE_KEYS = ['participant_id_norm', 'source_event', 'target_event']
PEOPLE = ['p01', 'p02', 'p03', 'p04', 'p05']
COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
BATCH_SIZE = 4096
SOURCES = {'events': ('outputs/macro_transition/fixed_D4K4_memberships_with_D13_coordinates_REFIT.csv', 'computed at invocation'), 'edges': ('outputs/macro_transition/authoritative_event_transitions_D4K4_REFIT.csv', 'computed at invocation'), 'independent_events': ('outputs/assignments/inputs/new_events.csv', 'computed at invocation'), 'independent_edges': ('outputs/assignments/inputs/event_edges.csv', 'computed at invocation')}
PROTOCOL = '# D4 event-distance test\n\nSpecified before the first result inspection for this package, September 26,\n2026. One exploratory primary test; all participant summaries descriptive.\n\nUse all 256 reviewed events and 251 authoritative directed links, reconstructing\nthe five participant chains by following successors. Compute Euclidean distance\nfrom raw Dim0-Dim3 MCA coordinates without standardizing or reweighting axes.\nRetain every observed link, including any zero-distance link. No event, cluster,\nbasin or dimension is selected on the test outcome.\n\nThe pooled statistic is mean step length across 251 links. Randomize interior\nevent order separately within participants, fixing each first and last event.\nThis preserves participant event sets and source/destination margins. It does\nnot preserve intervening interview-segment boundaries or other temporal\nstructure. The result concerns deviation from this explicitly specified random\norder, not a general test of the absence of every geometry-transition relation.\n\nFor a participant with start a, end z and m interior events I, the exact null\nexpected path-length sum is:\n\n  [sum_i d(a,i) + sum_i d(i,z) + sum_(i != j in I) d(i,j)] / m.\n\nEach endpoint connection and each ordered interior adjacency has probability\n1/m. Divide by m+1 links for the participant mean. For a two-event chain the\nsingle fixed distance is the expectation. The pooled expectation is the\nlink-count-weighted mean of these participant expectations.\n\nFor an observed mean d_obs and this exact expectation mu, use |d_obs - mu| as\nthe two-sided statistic. Compare with |d_shuffle - mu| for each randomized\ndataset. Include numerical ties with tolerance 1e-12 * max(1, |d_obs - mu|).\nThe Monte Carlo p-value is (exceedances + 1)/(B + 1). This absolute-departure\ndefinition is fixed in advance; do not select a tail after seeing the effect.\nDefault B=1,000,000, base seed=20260926, PCG64 child streams from SeedSequence,\nparticipant order p01, p02, p03, p04, p05. Fixed-size row-wise independent random\npermutations are evaluated by a precomputed distance matrix. Record software\nversions for reproducibility. No optional stopping is performed.\n\nReport observed and exact null-expected mean, signed difference, ratio, percent\nshorter (=100*(1-observed/null)), shuffled 2.5%-97.5% reference limits,\nexceedance count and p. The exact-binomial 95% interval for null-tail probability\nmeasures Monte Carlo precision only. Participant means and relative effects\nare descriptive; no participant p-values, family searches, or additional\ndimensionality tests are performed. No multiplicity correction is applied to\nthis single primary test, and it is not presented as corrected over the whole\nmanuscript. Equal weighting of links gives longer narratives more influence.\n\nInterpretation is conditional on the five narratives and the constructed event\nrepresentation. Event segmentation can constrain adjacent feature changes.\nA shorter mean supports local feature-space continuity relative to this null;\na longer mean supports larger changes. A nonsignificant mean does not establish\nindependence, equivalence or absence of structured routing. Sequence reversal\npreserves step lengths, so direction-specific claims are not supported.\n'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def distance_matrix(coords):
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))

def expected_total(d):
    n = len(d)
    require(n >= 2, 'A participant needs at least two events.')
    if n == 2:
        return float(d[0, 1])
    interior = np.arange(1, n - 1)
    return float((d[0, interior].sum() + d[interior, -1].sum() + d[np.ix_(interior, interior)].sum()) / len(interior))

def path_totals(d, inside):
    if inside.shape[1] == 0:
        return np.full(inside.shape[0], d[0, -1])
    totals = d[0, inside[:, 0]] + d[inside[:, -1], -1]
    if inside.shape[1] > 1:
        totals = totals + d[inside[:, :-1], inside[:, 1:]].sum(axis=1)
    return totals

def mathematical_checks():
    rng = np.random.Generator(np.random.PCG64(170326))
    expectation_errors = []
    distance_errors = []
    enumerated = 0
    for n in range(2, 9):
        coords = rng.normal(size=(n, 4))
        d = distance_matrix(coords)
        inside = np.asarray(list(itertools.permutations(range(1, n - 1))), dtype=int).reshape(-1, n - 2) if n > 2 else np.empty((1, 0), dtype=int)
        totals = path_totals(d, inside)
        brute = []
        for permutation in inside:
            order = np.r_[0, permutation, n - 1]
            brute.append(np.linalg.norm(np.diff(coords[order], axis=0), axis=1).sum())
        expectation_errors.append(abs(float(np.mean(brute)) - expected_total(d)))
        distance_errors.append(float(np.max(np.abs(totals - brute))))
        enumerated += len(inside)
    require(max(expectation_errors) < 1e-12 and max(distance_errors) < 1e-12, 'Exhaustive mathematical checks failed.')
    return {'toy_chain_sizes': list(range(2, 9)), 'enumerated_orders': enumerated, 'exact_expectation_max_abs_error': max(expectation_errors), 'vectorized_vs_direct_max_abs_error': max(distance_errors)}

def load_inputs(root):
    tables = {}
    provenance = {}
    for role, (relative, expected_hash) in SOURCES.items():
        path = root / relative
        require(path.is_file(), f'Missing input: {path}. See the input paths in the script notes.')
        actual = sha256(path)
        require(actual == expected_hash, f'Reviewed input version mismatch: {path}. Review the input; do not bypass this check.')
        provenance[role] = {'path_from_0_REFIT': relative, 'sha256': actual}
        tables[role] = pd.read_csv(path, keep_default_na=False)
    events = tables['events']
    edges = tables['edges']
    require(len(events) == 256 and (not events.duplicated(KEYS).any()), 'Expected 256 distinct events.')
    require(events.groupby(KEYS[0]).size().to_dict() == COUNTS, 'Participant event counts differ.')
    require(len(edges) == 251 and (not edges.duplicated(EDGE_KEYS).any()), 'Expected 251 unique links.')
    require(np.isfinite(events[DIMS].to_numpy(float)).all(), 'Nonfinite D4 coordinates.')
    independent = tables['independent_events']
    require(not independent.duplicated(KEYS).any(), 'Duplicate independent event keys.')
    a = events.set_index(KEYS).sort_index()
    b = independent.set_index(KEYS).sort_index()
    require(a.index.equals(b.index), 'Independent event inventories disagree.')
    require(np.allclose(a[DIMS], b[DIMS], rtol=0, atol=1e-12), 'Independent D4 coordinates disagree.')
    independent_edges = tables['independent_edges']
    require(len(independent_edges) == 251 and (not independent_edges.duplicated(EDGE_KEYS).any()), 'Invalid independent link inventory.')
    require(set(map(tuple, edges[EDGE_KEYS].to_numpy())) == set(map(tuple, independent_edges[EDGE_KEYS].to_numpy())), 'Independent link inventories disagree.')
    chains = []
    chronology = []
    step_rows = []
    for participant in PEOPLE:
        subset = events.loc[events[KEYS[0]].eq(participant)].set_index(KEYS[1])
        links = edges.loc[edges[KEYS[0]].eq(participant)]
        ids = set(subset.index)
        sources = list(links.source_event)
        targets = list(links.target_event)
        require(len(sources) == len(set(sources)) and len(targets) == len(set(targets)), 'Multiple successors/predecessors in a participant chain.')
        require(set(sources) <= ids and set(targets) <= ids and (len(links) == len(ids) - 1), 'Missing or cross-participant links.')
        starts = ids - set(targets)
        require(len(starts) == 1, 'Expected exactly one start per participant.')
        successors = dict(zip(sources, targets))
        current = next(iter(starts))
        visited = set()
        order = []
        while current not in visited:
            visited.add(current)
            order.append(current)
            if current not in successors:
                break
            current = successors[current]
        require(visited == ids and current not in successors, 'Cyclic or disconnected chain.')
        coords = subset.loc[order, DIMS].to_numpy(float)
        chains.append(coords)
        for t, (event, values) in enumerate(zip(order, coords)):
            chronology.append({'participant_id_norm': participant, 'chain_position': t, 'hippo_id_norm': event, **dict(zip(DIMS, values))})
        lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        for t, length in enumerate(lengths):
            step_rows.append({'participant_id_norm': participant, 'source_position': t, 'source_event': order[t], 'target_event': order[t + 1], 'distance_D4': float(length)})
    steps = pd.DataFrame(step_rows)
    require(set(map(tuple, steps[EDGE_KEYS].to_numpy())) == set(map(tuple, edges[EDGE_KEYS].to_numpy())), 'Reconstructed transitions differ from authoritative links.')
    return (chains, pd.DataFrame(chronology), steps, provenance)

def draw_null(chains, matrices, permutations, seed):
    child_seeds = np.random.SeedSequence(seed).spawn(len(chains))
    rngs = [np.random.Generator(np.random.PCG64(s)) for s in child_seeds]
    means = np.empty((permutations, len(chains)), dtype=float)
    for start in range(0, permutations, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, permutations)
        size = stop - start
        for column, (coords, d, rng) in enumerate(zip(chains, matrices, rngs)):
            template = np.broadcast_to(np.arange(1, len(coords) - 1), (size, len(coords) - 2))
            inside = rng.permuted(template, axis=1)
            if start == 0:
                require(np.array_equal(np.sort(inside, axis=1), template), 'Randomization lost or duplicated an event.')
                for j in range(min(8, size)):
                    order = np.r_[0, inside[j], len(coords) - 1]
                    direct = np.linalg.norm(np.diff(coords[order], axis=0), axis=1).sum()
                    require(np.isclose(path_totals(d, inside[j:j + 1])[0], direct, rtol=0, atol=1e-12), 'Permutation lookup differs from direct coordinates.')
            means[start:stop, column] = path_totals(d, inside) / (len(coords) - 1)
        if stop == permutations or start // 100000 != stop // 100000:
            print(f'Permutations: {stop:,}/{permutations:,}', flush=True)
    replay = np.empty((min(128, permutations), len(chains)))
    for column, (coords, d, s) in enumerate(zip(chains, matrices, np.random.SeedSequence(seed).spawn(len(chains)))):
        rng = np.random.Generator(np.random.PCG64(s))
        for j in range(len(replay)):
            inside = rng.permuted(np.arange(1, len(coords) - 1))
            order = np.r_[0, inside, len(coords) - 1]
            replay[j, column] = np.linalg.norm(np.diff(coords[order], axis=0), axis=1).mean()
    replay_error = float(np.max(np.abs(replay - means[:len(replay)])))
    require(replay_error < 1e-12, 'Independent deterministic permutation replay failed.')
    return (means, replay_error)

def p_text(p, exceedances):
    if exceedances == 0:
        return f'{p:.8f} (simulation floor; zero exceedances)'
    return f'{p:.6f}'

def results_text(pooled, participant, permutations, participant_labels):
    row = pooled.iloc[0]
    direction = 'shorter' if row.observed_minus_null < 0 else 'longer'
    lines = ['# D4 event-distance results', '', '256 events; 251 transitions; five participants. Frozen Dim0-Dim3 MCA coordinates.', f'{permutations:,} within-participant interior-order shuffles with endpoints fixed.', '', f'Observed pooled mean distance: {row.observed_mean_distance:.6f}', f'Exact shuffled-order expected mean: {row.null_expected_mean_exact:.6f}', f'Observed minus expected: {row.observed_minus_null:.6f}', f'Observed distances are {abs(row.percent_shorter_than_null):.2f}% {direction} than expected.', f'Two-sided randomization p: {p_text(row.p_two_sided, int(row.null_exceedances))}', f'Absolute-departure exceedances: {int(row.null_exceedances):,}/{permutations:,}', f'Shuffled 95% reference range: {row.null_q025:.6f} to {row.null_q975:.6f}. This is not an effect confidence interval.', f'95% Monte Carlo interval for the exact null-tail probability: {row.mc_tail_probability_ci_low:.8f} to {row.mc_tail_probability_ci_high:.8f}.', '', '| Participant | Links | Observed mean | Expected mean | Percent shorter |', '|---|---:|---:|---:|---:|']
    for r in participant.itertuples(index=False):
        lines.append(f'| {participant_labels[r.participant_id_norm]} | {r.links} | {r.observed_mean_distance:.3f} | {r.null_expected_mean_exact:.3f} | {r.percent_shorter_than_null:.1f}% |')
    lines += ['', 'Participant effects are descriptive; negative percent-shorter values mean longer observed steps. No participant-level significance tests were performed.', '', 'One exploratory primary test was specified. Its p-value is unadjusted and is not corrected across other manuscript analyses.', '', 'The observed mean differs from randomized order at the 0.05 level.' if row.p_two_sided < 0.05 else 'The test did not detect a difference in mean distance at the 0.05 level.', 'Inference is conditional on these narratives, event boundaries and frozen geometry. Whole-participant shuffling removes topic continuity and other interior-order structure. It does not preserve intervening segment/prompt boundaries. The result does not establish a causal mechanism, population generalization, full routing predictability or a specifically forward direction.', '']
    return '\n'.join(lines)

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
    args.participant_labels = config.get('participant_labels', {})
    require(isinstance(args.participant_labels, dict) and set(args.participant_labels) == set(PEOPLE) and (set(args.participant_labels.values()) == {f'Participant {i}' for i in range(1, 6)}), 'Configure five unique participant labels: Participant 1 through Participant 5.')
    SOURCES = {role: (relative, sha256(args.refit_root / relative)) for role, relative in config['inputs'].items()}
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refit-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--permutations', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--config', type=Path, default=HERE / 'input_paths.json', help='JSON configuration with input and output paths.')
    args = configure_paths(parser.parse_args())
    require(args.permutations >= 1000, 'Use at least 1000 permutations; the reporting default is 1000000.')
    require(args.seed >= 0, 'Seed must be nonnegative.')
    root = args.refit_root.resolve()
    output = args.output_dir.resolve()
    start_time = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=True)
        for name in ['VERIFICATION.json', 'RUN_METADATA.json']:
            (output / name).unlink(missing_ok=True)
    checks = mathematical_checks()
    chains, chronology, steps, provenance = load_inputs(root)
    matrices = [distance_matrix(x) for x in chains]
    counts = np.asarray([len(x) - 1 for x in chains])
    observed = np.asarray([np.linalg.norm(np.diff(x, axis=0), axis=1).mean() for x in chains])
    exact = np.asarray([expected_total(d) / count for d, count in zip(matrices, counts)])
    weights = counts / counts.sum()
    observed_pooled = float(observed @ weights)
    exact_pooled = float(exact @ weights)
    require(np.isclose(observed_pooled, steps.distance_D4.mean(), atol=1e-12, rtol=0), 'Pooled mean does not match the direct edge mean.')
    reverse_error = max((abs(np.linalg.norm(np.diff(x[::-1], axis=0), axis=1).mean() - o) for x, o in zip(chains, observed)))
    require(reverse_error < 1e-12, 'Sequence reversal unexpectedly changed mean distance.')
    checks.update({'input_hashes_match': True, 'independent_coordinates_match': True, 'independent_edges_match': True, 'events': len(chronology), 'links': len(steps), 'participants': len(chains), 'sequence_reversal_max_abs_error': reverse_error})
    print('Validated 256 events, 251 links, five participant chains and D4 coordinates.', flush=True)
    if args.check_only:
        print('CHECK ONLY PASSED: input and mathematical checks; no outputs written.', flush=True)
        return
    public_protocol = PROTOCOL.replace(', '.join(PEOPLE), ', '.join((args.participant_labels[p] for p in PEOPLE)))
    (output / 'PROTOCOL.md').write_text(public_protocol, encoding='utf-8')
    null_by_person, replay_error = draw_null(chains, matrices, args.permutations, args.seed)
    null_pooled = null_by_person @ weights
    deviation = abs(observed_pooled - exact_pooled)
    tolerance = 1e-12 * max(1, deviation)
    exceedances = int(np.count_nonzero(np.abs(null_pooled - exact_pooled) >= deviation - tolerance))
    p = (exceedances + 1) / (args.permutations + 1)
    lower_mc = float(beta.ppf(0.025, exceedances, args.permutations - exceedances + 1)) if exceedances else 0.0
    upper_mc = float(beta.ppf(0.975, exceedances + 1, args.permutations - exceedances)) if exceedances < args.permutations else 1.0
    pooled = pd.DataFrame([{'test': 'D4_event_distance', 'events': 256, 'links': 251, 'participants': 5, 'observed_mean_distance': observed_pooled, 'null_expected_mean_exact': exact_pooled, 'null_sample_mean': float(null_pooled.mean()), 'null_sd': float(null_pooled.std(ddof=1)), 'null_q025': float(np.quantile(null_pooled, 0.025)), 'null_q975': float(np.quantile(null_pooled, 0.975)), 'observed_minus_null': observed_pooled - exact_pooled, 'observed_over_null': observed_pooled / exact_pooled, 'percent_shorter_than_null': 100 * (1 - observed_pooled / exact_pooled), 'absolute_departure': deviation, 'null_exceedances': exceedances, 'permutations': args.permutations, 'p_two_sided': p, 'mc_tail_probability_ci_low': lower_mc, 'mc_tail_probability_ci_high': upper_mc, 'significant_unadjusted_0_05': p < 0.05}])
    participant = pd.DataFrame({'participant_id_norm': PEOPLE, 'events': counts + 1, 'links': counts, 'observed_mean_distance': observed, 'null_expected_mean_exact': exact, 'null_sample_mean': null_by_person.mean(axis=0), 'null_sd': null_by_person.std(axis=0, ddof=1), 'null_q025': np.quantile(null_by_person, 0.025, axis=0), 'null_q975': np.quantile(null_by_person, 0.975, axis=0), 'observed_minus_null': observed - exact, 'percent_shorter_than_null': 100 * (1 - observed / exact)})
    pooled.to_csv(output / 'pooled_event_distance_test.csv', index=False)
    participant.to_csv(output / 'participant_event_distances.csv', index=False)
    submission_participant = participant.copy()
    submission_participant.insert(0, 'participant', submission_participant.pop('participant_id_norm').map(args.participant_labels))
    submission_participant.to_csv(output / 'participant_event_distances_submission.csv', index=False)
    steps.to_csv(output / 'event_transition_distances.csv', index=False)
    chronology.to_csv(output / 'validated_chronology.csv', index=False)
    np.savez_compressed(output / 'event_distance_null_distributions.npz', pooled_mean=null_pooled, participant_means=null_by_person, participant_ids=np.asarray(PEOPLE), seed=np.asarray(args.seed), exact_expected_pooled=np.asarray(exact_pooled), exact_expected_participant=exact)
    report = results_text(pooled, participant, args.permutations, args.participant_labels)
    (output / 'RESULTS.md').write_text(report, encoding='utf-8')
    loaded_steps = pd.read_csv(output / 'event_transition_distances.csv')
    loaded_pooled = pd.read_csv(output / 'pooled_event_distance_test.csv').iloc[0]
    loaded_people = pd.read_csv(output / 'participant_event_distances.csv')
    loaded_submission = pd.read_csv(output / 'participant_event_distances_submission.csv')
    require(loaded_submission.participant.tolist() == [args.participant_labels[p] for p in PEOPLE], 'Submission participant labels differ.')
    require('participant_id_norm' not in loaded_submission.columns, 'Internal identifiers in submission table.')
    require(np.allclose(loaded_submission.drop(columns='participant').to_numpy(float), loaded_people.drop(columns='participant_id_norm').to_numpy(float), rtol=0, atol=1e-12), 'Submission table numerical values differ.')
    loaded_chronology = pd.read_csv(output / 'validated_chronology.csv')
    require(len(loaded_steps) == 251 and len(loaded_people) == 5 and (len(loaded_chronology) == 256), 'Output row-count verification failed.')
    require(np.isclose(loaded_steps.distance_D4.mean(), loaded_pooled.observed_mean_distance, atol=1e-12, rtol=0), 'Saved edge and pooled means differ.')
    with np.load(output / 'event_distance_null_distributions.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['pooled_mean'], null_pooled) and np.array_equal(saved['participant_means'], null_by_person), 'Saved null arrays differ.')
        saved_exceed = int(np.count_nonzero(np.abs(saved['pooled_mean'] - exact_pooled) >= deviation - tolerance))
    require(saved_exceed == exceedances and np.isclose((saved_exceed + 1) / (args.permutations + 1), loaded_pooled.p_two_sided, atol=1e-14, rtol=0), 'Saved p-value verification failed.')
    mc_mean_se = float(null_pooled.std(ddof=1) / np.sqrt(args.permutations))
    checks.update({'status': 'verified', 'permutation_replay_max_abs_error': replay_error, 'pooled_null_mean_minus_exact': float(null_pooled.mean() - exact_pooled), 'pooled_null_mean_monte_carlo_se': mc_mean_se, 'saved_output_checks_passed': True, 'null_array_rows': args.permutations, 'all_documented_outputs_written': True})
    write_json(output / 'VERIFICATION.json', checks)
    names = ['pooled_event_distance_test.csv', 'participant_event_distances.csv', 'participant_event_distances_submission.csv', 'event_transition_distances.csv', 'validated_chronology.csv', 'event_distance_null_distributions.npz', 'RESULTS.md', 'PROTOCOL.md', 'VERIFICATION.json']
    metadata = {'status': 'verified', 'started_utc': started_utc, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - start_time, 'refit_root': str(root), 'output_directory': str(output), 'script_sha256': sha256(__file__), 'dimensions': DIMS, 'distance': 'Euclidean; raw MCA coordinates; no axis rescaling', 'pooled_weighting': 'equal weight per transition', 'primary_tests': 1, 'multiplicity': 'one unadjusted exploratory primary test; not manuscript-wide corrected', 'participant_results': 'descriptive; no participant-level p-values', 'null': 'within-participant interior-order permutations; fixed endpoints', 'two_sided_statistic': 'absolute departure from exact null expectation', 'tie_tolerance': tolerance, 'seed': args.seed, 'rng': 'PCG64 with SeedSequence child streams', 'participant_order': PEOPLE, 'permutations': args.permutations, 'batch_size': BATCH_SIZE, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}, 'platform': platform.platform(), 'python_executable': sys.executable, 'sources': provenance, 'output_sha256': {name: sha256(output / name) for name in names}}
    metadata['path_configuration_sha256'] = sha256(args.config_path)
    metadata['participant_labels'] = args.participant_labels
    write_json(output / 'RUN_METADATA.json', metadata)
    print(report, flush=True)
    print(f'VERIFIED: all outputs written and checked in {output}', flush=True)
if __name__ == '__main__':
    main()
