"""Deterministic analysis of the frozen 256-event dataset."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path
import numpy as np
ASSIGNMENTS = ('D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39')
LABELS = ('4-D / 10 basins', '4-D / 9 basins', '4-D / 6 basins', '2-D / 9 basins', '2-D / 8 basins')
DIMENSIONS = (4, 10, 13)
PARTICIPANTS = ('p01', 'p02', 'p03', 'p04', 'p05')
FAMILIES = ('D4', 'D10', 'D13', 'all_three_spaces')
ALPHA = 0.05
GAMMA = 0.9
SEED = 42
N_PERM = 10000
EXACT_MAX_K = 8
TIE_TOLERANCE = 1e-12
CHECK_DRAWS = 128

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    require(bool(rows), 'Cannot write an empty result table')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def select_one(rows, **keys):
    found = [row for row in rows if all((row.get(k) == str(v) for k, v in keys.items()))]
    require(len(found) == 1, f'Expected one source row for {keys}; found {len(found)}')
    return found[0]

def partial_conjunction(pvalues):
    ordered = sorted(pvalues)
    require(all((math.isfinite(p) and 0 <= p <= 1 for p in ordered)), 'Invalid p-value')
    raw = [min(1.0, (len(ordered) - i) * p) for i, p in enumerate(ordered)]
    sequential = list(itertools.accumulate(raw, max))
    return (raw, sequential)

def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted

def exhaustive_checks(pvalues, raw, adjusted):
    m = len(pvalues)
    for u, actual in enumerate(raw, 1):
        size = m - u + 1
        expected = max((min(1.0, size * min((pvalues[i] for i in subset))) for subset in itertools.combinations(range(m), size)))
        require(abs(actual - expected) < 1e-14, 'Partial conjunction subset check failed')
    for i, actual in enumerate(adjusted):
        expected = max((min(1.0, len(subset) * min((pvalues[j] for j in subset))) for size in range(1, m + 1) for subset in itertools.combinations(range(m), size) if i in subset))
        require(abs(actual - expected) < 1e-14, 'Holm closed-testing check failed')

def distances(values):
    return np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2))

def correlation(x, y):
    xc, yc = (x - x.mean(), y - y.mean())
    require(np.linalg.norm(xc) > 0 and np.linalg.norm(yc) > 0, 'Constant distance vector')
    return float(xc @ yc / (np.linalg.norm(xc) * np.linalg.norm(yc)))

def matrix(path, k):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    labels = [f'B{i}' for i in range(1, k + 1)]
    require(rows[0][1:] == labels and [r[0] for r in rows[1:]] == labels, f'Unexpected matrix labels: {path}')
    result = np.asarray([r[1:] for r in rows[1:]], dtype=float)
    require(result.shape == (k, k) and np.isfinite(result).all(), f'Invalid matrix: {path}')
    return result

def exact_label_test(geometry, predictive, observed):
    k = len(predictive)
    upper = np.triu_indices(k, 1)
    x = geometry[upper]
    xc = x - x.mean()
    norm = np.linalg.norm(xc)
    extreme = extreme_wider = total = 0
    stream = itertools.permutations(range(k))
    while True:
        block = list(itertools.islice(stream, 2048))
        if not block:
            break
        index = np.asarray(block, dtype=np.int64)
        ys = predictive[index[:, upper[0]], index[:, upper[1]]]
        yc = ys - ys.mean(axis=1, keepdims=True)
        null = yc @ xc / (np.linalg.norm(yc, axis=1) * norm)
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed) - TIE_TOLERANCE))
        extreme_wider += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-10))
        total += len(block)
    require(total == math.factorial(k), 'Incomplete exact permutation group')
    require(extreme == extreme_wider, 'Exact result is sensitive to numerical tie tolerance')
    return (extreme / total, extreme, total)

def fp(value):
    return '<0.001' if value < 0.001 else f'{value:.3f}'

def run_partial_conjunction(input_root, output):
    require(not output.exists() or (output.is_dir() and (not any(output.iterdir()))), f'Output must be absent or empty: {output}')
    require(not input_root.is_relative_to(output), 'Output cannot contain the input tree')
    for protected in (input_root / 'local', input_root / 'robustness'):
        require(not output.is_relative_to(protected), 'Output cannot modify a source subtree')
    provenance, checks = ({}, [])

    def source(relative):
        path = input_root / relative
        require(path.is_file(), f'Missing required source: {path}')
        provenance[relative] = sha256(path)
        return path

    def check(name, condition):
        require(bool(condition), name)
        checks.append({'check': name, 'passed': True})
    geo = read_csv(source('robustness/comparisons/four_conditions_geometry.csv'))
    reference = read_csv(source('local/comparisons/successor_geometry_all_assignments.csv'))
    order_reference = read_csv(source('robustness/comparisons/event_order.csv'))
    rows, exact, matrices, sequences, canonical_order = ([], [], {}, {}, None)
    for aid, label in zip(ASSIGNMENTS, LABELS):
        k = int(aid.split('_')[1][1:])
        prefix = f'local/{aid}/geometry_090'
        successor = matrix(source(f'{prefix}/successor_matrix.csv'), k)
        check(f'{aid}: successor row sums', np.allclose(successor.sum(1), 10, rtol=0, atol=1e-09))
        predictive = distances(successor)
        geometries = {}
        for d in DIMENSIONS:
            row = select_one(geo, assignment_id=aid, participant_omitted='none', transition_unit='event_adjacency', geometry_D=d, geometry_condition='full_sample_geometry')
            ref = select_one(reference, assignment_id=aid, gamma='0.9', geometry_D=d, representation='full_profile')
            check(f'{aid}/D{d}: valid source status', row['status'] == 'ok')
            check(f'{aid}/D{d}: source settings', int(ref['n_permutations']) == N_PERM and int(ref['seed']) == SEED and (int(ref['n_pairs']) == k * (k - 1) // 2))
            r, p = (float(row['pearson_r']), float(row['permutation_p']))
            check(f'{aid}/D{d}: duplicate export agreement', abs(r - float(ref['pearson_r'])) < 1e-12 and abs(p - float(ref['two_sided_common_label_permutation_p'])) < 1e-12)
            check(f'{aid}/D{d}: finite r and p', math.isfinite(r) and -1 <= r <= 1 and math.isfinite(p) and (0 < p <= 1))
            geometry = matrix(source(f'{prefix}/geometry_distances_D{d}.csv'), k)
            check(f'{aid}/D{d}: symmetric geometry', np.allclose(geometry, geometry.T, rtol=0, atol=1e-12))
            upper = np.triu_indices(k, 1)
            observed = correlation(geometry[upper], predictive[upper])
            check(f'{aid}/D{d}: r recalculated from exported matrices', abs(observed - r) < 1e-10)
            exact_p = None
            if k <= EXACT_MAX_K:
                exact_p, count, total = exact_label_test(geometry, predictive, observed)
                exact.append({'assignment_id': aid, 'geometry_D': d, 'K': k, 'pearson_r': observed, 'source_monte_carlo_p': p, 'exact_p': exact_p, 'extreme_count': count, 'total_permutations': total, 'tie_tolerance': TIE_TOLERANCE})
            rows.append({'assignment_id': aid, 'solution': label, 'geometry_D': d, 'K': k, 'n_pairs': k * (k - 1) // 2, 'pearson_r': r, 'source_p': p, 'precision_checked_p': p if exact_p is None else exact_p, 'precision_method': 'exact' if exact_p is not None else 'source_Monte_Carlo', 'gamma': GAMMA, 'profile': 'complete_M', 'transition_unit': 'event_adjacency', 'tail': 'two_sided_absolute_r'})
            geometries[d] = geometry
        matrices[aid] = (k, geometries)
        manifest = read_csv(source(f'robustness/{aid}/event_order/full_sample/sequence_manifest_forward_and_reverse.csv'))
        manifest.sort(key=lambda x: (x['participant_id_norm'], int(x['sequence_position'])))
        identities = [(x['participant_id_norm'], int(x['sequence_position']), x['hippo_id_norm']) for x in manifest]
        check(f'{aid}: 256 unique event identities', len(identities) == 256 and len({(p, e) for p, _, e in identities}) == 256)
        if canonical_order is None:
            canonical_order = identities
        check(f'{aid}: same event identities and sequence positions', identities == canonical_order)
        seq = {}
        for pid in PARTICIPANTS:
            retained = [x for x in manifest if x['participant_id_norm'] == pid]
            check(f'{aid}/{pid}: contiguous sequence positions', [int(x['sequence_position']) for x in retained] == list(range(1, len(retained) + 1)))
            seq[pid] = np.asarray([int(x['state']) - 1 for x in retained], dtype=int)
            check(f'{aid}/{pid}: valid basin labels', len(seq[pid]) > 0 and bool(np.all((seq[pid] >= 0) & (seq[pid] < k))))
        check(f'{aid}: 251 within-participant edges', sum((len(v) - 1 for v in seq.values())) == 251)
        sequences[aid] = seq
    prefix_max_error = 0.0
    for aid in ASSIGNMENTS:
        k, geometries = matrices[aid]
        upper = np.triu_indices(k, 1)
        saved = read_csv(source(f'robustness/{aid}/event_order/full_sample/permutation_nulls.csv'))
        check(f'{aid}: complete saved order null', len(saved) == N_PERM and [int(x['permutation']) for x in saved] == list(range(1, N_PERM + 1)))
        rng = np.random.default_rng(SEED)
        max_error = 0.0
        for i in range(CHECK_DRAWS):
            counts = np.zeros((k, k))
            for pid in PARTICIPANTS:
                seq = sequences[aid][pid]
                shuffled = seq[rng.permutation(len(seq))]
                np.add.at(counts, (shuffled[:-1], shuffled[1:]), 1)
            totals = counts.sum(1, keepdims=True)
            pmat = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
            m = np.linalg.solve(np.eye(k) - GAMMA * pmat, np.eye(k))
            dy = distances(m)[upper]
            for d in DIMENSIONS:
                r = correlation(geometries[d][upper], dy)
                max_error = max(max_error, abs(r - float(saved[i][f'D{d}_order_shuffle_r'])))
        check(f'{aid}: first {CHECK_DRAWS} shared event-shuffle draws reproduced', max_error < 1e-10)
        prefix_max_error = max(prefix_max_error, max_error)
        for d in DIMENSIONS:
            ref = select_one(order_reference, assignment_id=aid, order='narrated', geometry_D=d)
            check(f'{aid}/D{d}: order-null settings', int(ref['n_permutations']) == N_PERM and int(ref['seed']) == SEED and (float(ref['gamma']) == GAMMA))
            r = float(ref['pearson_r'])
            p = (1 + sum((float(s[f'D{d}_order_shuffle_r']) >= r for s in saved))) / (N_PERM + 1)
            check(f'{aid}/D{d}: stored order p reproduced', abs(p - float(ref['order_shuffle_p_upper_tail'])) < 1e-12)
    pc_rows, solution_rows = ([], [])
    for basis in ('source_Monte_Carlo', 'precision_checked'):
        key = 'source_p' if basis == 'source_Monte_Carlo' else 'precision_checked_p'
        for family in FAMILIES:
            pvalues = []
            for aid in ASSIGNMENTS:
                relevant = [r for r in rows if r['assignment_id'] == aid and (family == 'all_three_spaces' or family == f"D{r['geometry_D']}")]
                pvalues.append(max((r[key] for r in relevant)))
            raw, sequential = partial_conjunction(pvalues)
            adjusted = holm(pvalues)
            exhaustive_checks(pvalues, raw, adjusted)
            check(f'{basis}/{family}: exhaustive subset and closed-testing verification', True)
            lower_bound = sum((p <= ALPHA for p in sequential))
            for u, (p, seq_p) in enumerate(zip(raw, sequential), 1):
                pc_rows.append({'basis': basis, 'family': family, 'minimum_assignments': u, 'n_assignments': len(ASSIGNMENTS), 'partial_conjunction_p': p, 'fixed_sequence_p': seq_p, 'reject_fixed_sequence_at_0_05': seq_p <= ALPHA, 'lower_bound_at_0_05': lower_bound})
            for aid, label, p, q in zip(ASSIGNMENTS, LABELS, pvalues, adjusted):
                solution_rows.append({'basis': basis, 'family': family, 'assignment_id': aid, 'solution': label, 'input_p': p, 'holm_p': q, 'reject_holm_at_0_05': q <= ALPHA})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'component_results.csv', rows)
    write_csv(output / 'partial_conjunction_results.csv', pc_rows)
    write_csv(output / 'assignment_holm_results.csv', solution_rows)
    write_csv(output / 'exact_permutation_check.csv', exact)
    feasibility = {'same_event_identities_and_positions': True, 'same_seed_and_participant_iteration_order': True, 'shared_draws_recomputed_per_assignment': CHECK_DRAWS, 'number_of_draw_correlations_checked': CHECK_DRAWS * len(ASSIGNMENTS) * len(DIMENSIONS), 'maximum_absolute_error': prefix_max_error, 'all_15_saved_order_pvalues_recalculated': True, 'joint_order_test_executed': False, 'scope': 'within-participant full narrated chains, including existing segment joins', 'limitation': 'Prefix reproduction and source-code audit establish feasibility; a future joint test should reproduce all synchronized draws or use verified saved draws. Its null concerns event order, not basin-label matching.'}
    (output / 'joint_test_feasibility.json').write_text(json.dumps(feasibility, indent=2) + '\n', encoding='utf-8')
    provenance_record = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(), 'script_sha256': sha256(Path(__file__).resolve()), 'input_sha256': provenance, 'analysis_type': 'exploratory, conditional on the five existing assignments', 'constants': {'alpha': ALPHA, 'gamma': GAMMA, 'dimensions': DIMENSIONS, 'exact_max_K': EXACT_MAX_K, 'tie_tolerance': TIE_TOLERANCE}, 'checks': checks, 'all_checks_passed': True, 'source_inputs_unchanged': all((sha256(input_root / p) == h for p, h in provenance.items())), 'refitting_executed': False, 'primary_nulls_regenerated': 'Exact enumeration for K6 and K8 only'}
    require(provenance_record['source_inputs_unchanged'], 'An input changed during execution')
    (output / 'verification.json').write_text(json.dumps(provenance_record, indent=2) + '\n', encoding='utf-8')
    report = make_report(rows, pc_rows, solution_rows, exact, feasibility, len(checks))
    (output / 'Robustness_results.md').write_text(report, encoding='utf-8')
    print(report)
    return output

def make_report(components, pc_rows, solutions, exact, feasibility, nchecks):

    def pc(basis, family, u):
        return next((x for x in pc_rows if x['basis'] == basis and x['family'] == family and (x['minimum_assignments'] == u)))
    text = ['# Robustness of geometry-successor correspondence', '', 'Exploratory analysis of five existing basin assignments. All geometry tests use complete successor profiles, gamma = 0.90, event adjacencies, and two-sided basin-label permutation tests. Fitting dimensionality and distance dimensionality are distinct.', '', '## Main results', '', '| Requirement | Original Monte Carlo p | Precision-checked p |', '| --- | ---: | ---: |']
    for family, description in (('D4', '4-D distances'), ('all_three_spaces', 'Every distance space: 4-D, 10-D, 13-D')):
        for u in (2, 3):
            a = pc('source_Monte_Carlo', family, u)['fixed_sequence_p']
            b = pc('precision_checked', family, u)['fixed_sequence_p']
            text.append(f'| At least {u} assignments; {description} | {fp(a)} | {fp(b)} |')
    text += ['', 'Precision-checked values substitute exact enumeration for every assignment with K <= 8; larger assignments retain their original 10,000-draw Monte Carlo p-values. This computational rule depends on basin count, not observed significance.', '', 'All 15 observed correlations are positive. Two-sided tests establish departure from the label-matching null; the direction is reported from the observed correlations and is not a separately tested directional conjunction.', '', '## Assignment-level results', '', "Holm adjustment controls familywise error across the five assignments within each displayed family. The all-space input is each assignment's largest p across the three distance spaces.", '', '| Assignment | 4-D r | Source 4-D p | Precision-checked 4-D Holm p | All-space Holm p |', '| --- | ---: | ---: | ---: | ---: |']
    for aid, label in zip(ASSIGNMENTS, LABELS):
        r = next((x for x in components if x['assignment_id'] == aid and x['geometry_D'] == 4))
        q = {x['family']: x['holm_p'] for x in solutions if x['basis'] == 'precision_checked' and x['assignment_id'] == aid}
        text.append(f"| {label} | {r['pearson_r']:.3f} | {fp(r['source_p'])} | {fp(q['D4'])} | {fp(q['all_three_spaces'])} |")
    text += ['', '## Calculation and interpretation', '', 'For m = 5 ordered p-values, the Bonferroni partial-conjunction p-value for at least u non-null assignments is min[1, (m-u+1) p_(u)]. It allows arbitrary dependence between valid component tests. A significant test for u = 2 rejects the hypothesis that at most one assignment departs from its label-matching null. It does not identify assignments by itself; the separate Holm results address identification.', '', 'Within each assignment, max(p_D4, p_D10, p_D13) tests the conjunction that all three distance-space nulls are false. Applying partial conjunction to these five maxima asks whether at least u of the same assignments retain support across every distance space. No independence between distance spaces is assumed.', '', "Fixed-sequence p-values are cumulative maxima of the raw partial-conjunction p-values as u increases. They prevent choosing a favorable u after inspection and yield a simultaneous lower bound within each family. D10 and D13 are descriptive sensitivity columns. The all-space claim implies the D4 claim; they are nested claims, not independent confirmations. The result does not adjust the entire manuscript's collection of endpoints.", '', 'The scope is the five supplied assignments, conditional on the existing geometry, memberships, annotations, and component null assumptions. These tests do not correct outcome-driven selection from a larger unreported specification search. They do not establish a population effect from five participants, equivalent effect sizes, a uniquely multistep successor mechanism, or robustness to every projection and assignment algorithm. Procrustes and feature routing are different endpoints and have not been pooled.', '', 'The common Dim0 x Dim3 projection is a visualization. These correlation tests measure distances in 4-D, 10-D, or 13-D, so they do not require the displayed projection to preserve every relationship.', '', '## Execution verification', '', f'{nchecks} checks passed. Recalculated all 15 correlations from exported matrices; checked source export agreement and settings; exactly enumerated 720 permutations for K6 and 40,320 for K8 in each distance space; independently checked partial conjunction using all relevant subsets and Holm using closed testing. Inputs were hashed and confirmed unchanged.', '', f"For future joint event-order testing, event identities and positions match across all five assignments. Reproduced {feasibility['number_of_draw_correlations_checked']} correlations from the first {CHECK_DRAWS} shared shuffles per assignment, maximum error {feasibility['maximum_absolute_error']:.3g}. Recalculated every stored full-sample order p-value. This partial-conjunction stage does not compute the joint order p-value; the subsequently executed assessment is in ../joint_event_order.", '', '## Methods references', '', 'Wang, J., & Owen, A. B. Admissibility in partial conjunction testing. https://arxiv.org/abs/1508.00934 (generalized partial conjunction under dependence).', 'Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. Nature Human Behaviour, 4, 1208-1214. https://doi.org/10.1038/s41562-020-0912-z', 'Winkler, A. M., et al. (2016). Non-parametric combination and related permutation tests for neuroimaging. Human Brain Mapping, 37, 1486-1511. https://doi.org/10.1002/hbm.23115', '']
    return '\n'.join(text)

def joint_event_order(input_root, output):
    require(not output.exists(), f'Joint output already exists: {output}')
    hashes = {}

    def source(relative):
        path = input_root / relative
        require(path.is_file(), f'Missing input: {path}')
        hashes[relative] = sha256(path)
        return path
    table = read_csv(source('robustness/comparisons/event_order.csv'))
    geometries, manifests, saved_nulls, observed = ({}, {}, {}, [])
    canonical = None
    checks = []
    for aid in ASSIGNMENTS:
        k = int(aid.split('_')[1][1:])
        manifest = read_csv(source(f'robustness/{aid}/event_order/full_sample/sequence_manifest_forward_and_reverse.csv'))
        manifest.sort(key=lambda x: (x['participant_id_norm'], int(x['sequence_position'])))
        ids = [(x['participant_id_norm'], int(x['sequence_position']), x['hippo_id_norm']) for x in manifest]
        if canonical is None:
            canonical = ids
        require(ids == canonical and len(ids) == 256, 'Event alignment differs across assignments')
        manifests[aid] = {pid: np.array([int(x['state']) - 1 for x in manifest if x['participant_id_norm'] == pid], dtype=int) for pid in PARTICIPANTS}
        require(sum((len(x) - 1 for x in manifests[aid].values())) == 251, 'Unexpected transition total')
        upper = np.triu_indices(k, 1)
        geometries[aid] = np.stack([matrix(source(f'local/{aid}/geometry_090/geometry_distances_D{d}.csv'), k)[upper] for d in DIMENSIONS])
        saved = read_csv(source(f'robustness/{aid}/event_order/full_sample/permutation_nulls.csv'))
        require(len(saved) == N_PERM and [int(x['permutation']) for x in saved] == list(range(1, N_PERM + 1)), 'Null draw IDs differ')
        saved_nulls[aid] = np.array([[float(row[f'D{d}_order_shuffle_r']) for d in DIMENSIONS] for row in saved])
        obs = []
        for d in DIMENSIONS:
            row = select_one(table, assignment_id=aid, order='narrated', geometry_D=d)
            require(int(row['seed']) == SEED and int(row['n_permutations']) == N_PERM and (float(row['gamma']) == GAMMA), 'Unexpected order-null settings')
            obs.append(float(row['pearson_r']))
        observed.append(obs)
    observed = np.array(observed)
    joint_null = np.empty((N_PERM, len(DIMENSIONS)))
    regenerated = np.empty((N_PERM, len(ASSIGNMENTS), len(DIMENSIONS)))
    rng = np.random.default_rng(SEED)
    for start in range(0, N_PERM, 500):
        stop = min(start + 500, N_PERM)
        indices = {pid: [] for pid in PARTICIPANTS}
        for draw in range(start, stop):
            for pid in PARTICIPANTS:
                indices[pid].append(rng.permutation(len(manifests[ASSIGNMENTS[0]][pid])))
        indices = {pid: np.array(values) for pid, values in indices.items()}
        for j, aid in enumerate(ASSIGNMENTS):
            k = int(aid.split('_')[1][1:])
            count = np.zeros((stop - start, k, k))
            for pid in PARTICIPANTS:
                seq = manifests[aid][pid][indices[pid]]
                np.add.at(count, (np.arange(stop - start)[:, None], seq[:, :-1], seq[:, 1:]), 1)
            totals = count.sum(2, keepdims=True)
            prob = np.divide(count, totals, out=np.zeros_like(count), where=totals > 0)
            eye = np.broadcast_to(np.eye(k), (stop - start, k, k))
            sr = np.linalg.solve(eye - GAMMA * prob, eye)
            iu = np.triu_indices(k, 1)
            delta = sr[:, iu[0], :] - sr[:, iu[1], :]
            y = np.sqrt(np.sum(delta * delta, axis=2))
            yc = y - y.mean(1, keepdims=True)
            x = geometries[aid]
            xc = x - x.mean(1, keepdims=True)
            rs = yc @ xc.T / (np.linalg.norm(yc, axis=1)[:, None] * np.linalg.norm(xc, axis=1)[None, :])
            require(np.isfinite(rs).all(), 'Nonfinite regenerated correlation')
            regenerated[start:stop, j, :] = rs
            error = float(np.max(np.abs(rs - saved_nulls[aid][start:stop])))
            require(error < 1e-10, f'Regenerated order null mismatch: {aid}, block {start}, error {error}')
            checks.append({'assignment_id': aid, 'draw_start': start + 1, 'draw_end': stop, 'maximum_absolute_error': error})
        joint_null[start:stop] = np.median(regenerated[start:stop], axis=1)
        print(f'Validated synchronized draws {stop}/{N_PERM}', flush=True)
    observed_median = np.median(observed, axis=0)
    rows = []
    for i, d in enumerate(DIMENSIONS):
        count = int(np.count_nonzero(joint_null[:, i] >= observed_median[i] - TIE_TOLERANCE))
        wider = int(np.count_nonzero(joint_null[:, i] >= observed_median[i] - 1e-10))
        require(count == wider, 'Joint p-value is sensitive to tie tolerance')
        saved_median = np.median(np.stack([saved_nulls[a][:, i] for a in ASSIGNMENTS], axis=1), axis=1)
        require(np.max(np.abs(saved_median - joint_null[:, i])) < 1e-10, 'Stored median differs from regenerated median')
        require(np.count_nonzero(saved_median >= observed_median[i] - TIE_TOLERANCE) == count, 'Stored and regenerated exceedance counts differ')
        rows.append({'geometry_D': d, 'observed_median_r': float(observed_median[i]), 'null_mean_median_r': float(joint_null[:, i].mean()), 'null_median_median_r': float(np.median(joint_null[:, i])), 'null_q025': float(np.quantile(joint_null[:, i], 0.025)), 'null_q975': float(np.quantile(joint_null[:, i], 0.975)), 'extreme_count': count, 'n_permutations': N_PERM, 'upper_tail_p': (1 + count) / (1 + N_PERM), 'n_assignments': 5, 'gamma': GAMMA, 'seed': SEED, 'statistic': 'median_Pearson_r_across_five_assignments'})
    conservative_p = max((row['upper_tail_p'] for row in rows))
    summary = {'primary_D4_p': rows[0]['upper_tail_p'], 'all_three_spaces_max_p': conservative_p, 'all_three_spaces_pass_0_05': conservative_p <= ALPHA, 'interpretation': 'The maximum is a conservative requirement that the same fixed aggregate test pass in each distance space. It does not establish an order effect in every individual assignment.', 'null': 'Uniform reordering of complete event records within each participant-global chain; participant lengths and basin occupancy fixed; P and M rebuilt.', 'preserved': 'Event identities, geometry, basin assignments, participant membership, and sequence lengths.', 'not_preserved': 'Original event adjacency, serial dependence, and segment-specific event composition. The existing chain convention includes 16 between-segment joins.', 'exploratory': True, 'statistic_fixed_before_joint_result': True, 'alternative_aggregate_statistics_searched': False}
    output.mkdir(parents=True)
    write_csv(output / 'joint_event_order_results.csv', rows)
    write_csv(output / 'joint_event_order_null.csv', [{'permutation': i + 1, **{f'D{d}_median_r': float(joint_null[i, j]) for j, d in enumerate(DIMENSIONS)}} for i in range(N_PERM)])
    (output / 'joint_event_order_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    verification = {'script_sha256': sha256(Path(__file__).resolve()), 'input_sha256': hashes, 'numpy': np.__version__, 'regenerated_null_draws': N_PERM, 'regenerated_component_correlations': int(regenerated.size), 'block_checks': checks, 'maximum_absolute_error': max((x['maximum_absolute_error'] for x in checks)), 'same_statistic_used_for_observed_and_null': True, 'source_inputs_unchanged': all((sha256(input_root / p) == h for p, h in hashes.items()))}
    require(verification['source_inputs_unchanged'], 'Input mutated during execution')
    (output / 'joint_event_order_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    report = ['# Joint event-order assessment', '', 'The fixed statistic is the median geometry–successor correlation across all five existing basin assignments. Each permutation reorders the same event records within participants and is applied to all assignments before rebuilding their transition and successor matrices. The primary distance geometry is 4-D; 10-D and 13-D are named sensitivity checks.', '', '| Distance space | Observed median r | Null median | Upper-tail p |', '| --- | ---: | ---: | ---: |']
    for row in rows:
        report.append(f"| {row['geometry_D']}-D | {row['observed_median_r']:.3f} | {row['null_median_median_r']:.3f} | {fp(row['upper_tail_p'])} |")
    report += ['', f'The conservative all-space criterion is the largest joint p-value: {fp(conservative_p)}.', '', 'This tests whether the observed cross-solution median exceeds that under the specified event-order null. It does not identify how many individual assignments have order-dependent correspondence; that question is distinct from the earlier partial-conjunction assessment of basin-label matching.', '', 'All 10,000 shared shuffles were regenerated, comprising 150,000 component correlations, and matched the original stored arrays within 1e-10. Saved-array and regenerated-array aggregate p-values agree. No alternative aggregate statistic or weighting was tried.', '', 'This analysis does not identify a unique multistep successor contribution or establish participant-level replication. Interpret it conditional on the existing event definitions, geometry, basin assignments, and within-participant exchangeability null.', '']
    (output / 'Joint_event_order_results.md').write_text('\n'.join(report))
    print('\n'.join(report))
    return output

def run_all(input_root, output):
    require(not output.exists(), f'Output must be a new directory: {output}')
    run_partial_conjunction(input_root, output / 'partial_conjunction')
    joint_event_order(input_root, output / 'joint_event_order')

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--refit-root', type=Path, help='Path to the original 0_REFIT folder')
    group.add_argument('--input-root', type=Path, help='Path with local/ and robustness/ source exports')
    parser.add_argument('--output', type=Path, help='New or empty result directory')
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.refit_root:
        root = args.refit_root.expanduser().resolve() / 'outputs/downstream'
    elif args.input_root:
        root = args.input_root.expanduser().resolve()
    elif (script_dir / 'inputs').is_dir():
        root = script_dir / 'inputs'
    elif (Path.cwd() / 'outputs/downstream').is_dir():
        root = Path.cwd() / 'outputs/downstream'
    else:
        parser.error('Use --refit-root /path/to/0_REFIT or place the script beside supplied inputs/')
    output = (args.output or Path.cwd() / 'cross_solution_results').expanduser().resolve()
    try:
        run_all(root.resolve(), output)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f'Input or validation error: {exc}\n')
if __name__ == '__main__':
    main()
