"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
GAMMA = 0.9
DIMENSIONS = (4, 10, 13)
SEED = 42
N_PERM = 10000

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write_csv(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)

def write_matrix(matrix, labels, path):
    write_csv(pd.DataFrame(matrix, index=labels, columns=labels), path, True)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def counts_from_sequences(sequences, k, collapse=False):
    counts = np.zeros((k, k), dtype=float)
    for values in sequences.values():
        values = np.asarray(values, dtype=np.int16)
        if collapse:
            values = collapse_runs(values)
        if len(values) > 1:
            np.add.at(counts, (values[:-1], values[1:]), 1.0)
    return counts

def collapse_runs(values):
    values = np.asarray(values)
    return values[np.r_[True, values[1:] != values[:-1]]] if len(values) else values.copy()

def row_normalize(counts):
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)

def successor(probabilities):
    identity = np.eye(len(probabilities))
    a = identity - GAMMA * probabilities
    matrix = np.linalg.solve(a, identity)
    require(np.isfinite(matrix).all() and (matrix >= -1e-12).all(), 'Invalid successor matrix.')
    require(np.linalg.norm(a @ matrix - identity, ord='fro') < 1e-09, 'Successor solve residual too large.')
    return matrix

def distances(values):
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def centroids(events, d, k):
    return events.groupby('state', sort=True)[[f'Dim{i}' for i in range(d)]].mean().reindex(range(1, k + 1)).to_numpy(float)

def bh_fdr(values):
    p = np.asarray(values, dtype=float)
    require(np.isfinite(p).all(), 'Nonfinite p-values in BH family.')
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    result = np.empty_like(p)
    result[order] = q
    return result

def upper_tail(null, observed):
    return (1 + np.count_nonzero(null >= np.asarray(observed)[None, ...], axis=0)) / (len(null) + 1)

def permutation_bank(k, n_perm=N_PERM):
    rng = np.random.default_rng(SEED)
    return np.array([rng.permutation(k) for _ in range(n_perm)], dtype=np.int16)

def correspondence(geometry, predictive, bank):
    k = len(geometry)
    upper = np.triu_indices(k, 1)
    x, y = (geometry[upper], predictive[upper])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return ({'pearson_r': np.nan, 'permutation_p': np.nan, 'status': 'not_estimable_missing_centroid'}, np.full(len(bank), np.nan))
    if np.std(x) == 0 or np.std(y) == 0:
        return ({'pearson_r': np.nan, 'permutation_p': np.nan, 'status': 'not_estimable_constant_distances'}, np.full(len(bank), np.nan))
    observed = float(np.corrcoef(x, y)[0, 1])
    unique, inverse = np.unique(bank, axis=0, return_inverse=True)
    null_unique = np.array([float(np.corrcoef(x, predictive[order][:, order][upper])[0, 1]) for order in unique])
    null = null_unique[inverse]
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)
    return ({'pearson_r': observed, 'permutation_p': float(p), 'null_mean_r': float(null.mean()), 'null_sd_r': float(null.std(ddof=1)), 'status': 'ok'}, null)

def recurrence_null(sequences, k, n_perm=N_PERM, progress=None):
    rng = np.random.default_rng(SEED)
    null = np.empty((n_perm, k))
    zero_rows = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        shuffled = {p: rng.permutation(v) for p, v in sequences.items()}
        counts = counts_from_sequences(shuffled, k)
        zero_rows[i] = np.count_nonzero(counts.sum(1) == 0)
        null[i] = successor(row_normalize(counts)).mean(0)
        if progress and (i + 1) % 2000 == 0:
            progress(i + 1)
    return (null, zero_rows)

def sequence_nulls(sequences, k, n_perm=N_PERM, progress=None):
    rng = np.random.default_rng(SEED)
    total_edges = sum((max(len(v) - 1, 0) for v in sequences.values()))
    result = {'self_counts': np.empty((n_perm, k)), 'self_probabilities': np.empty((n_perm, k)), 'visit_centrality': np.empty((n_perm, k)), 'visit_transitions': np.empty(n_perm, dtype=int), 'event_zero_rows': np.empty(n_perm, dtype=int), 'visit_zero_rows': np.empty(n_perm, dtype=int)}
    for i in range(n_perm):
        shuffled = {p: rng.permutation(v) for p, v in sequences.items()}
        counts = counts_from_sequences(shuffled, k)
        visits = counts_from_sequences(shuffled, k, collapse=True)
        expected = counts.copy()
        np.fill_diagonal(expected, 0)
        require(np.array_equal(visits, expected), 'Collapsed transitions differ from off-diagonal event counts.')
        require(counts.sum() == total_edges, 'Permutation changed event-adjacency count.')
        result['self_counts'][i] = np.diag(counts)
        result['self_probabilities'][i] = np.diag(row_normalize(counts))
        result['event_zero_rows'][i] = np.count_nonzero(counts.sum(1) == 0)
        result['visit_zero_rows'][i] = np.count_nonzero(visits.sum(1) == 0)
        result['visit_centrality'][i] = successor(row_normalize(visits)).mean(0)
        result['visit_transitions'][i] = int(visits.sum())
        if progress and (i + 1) % 2000 == 0:
            progress(i + 1)
    return result

def centrality_table(data, observed, null, event_counts, outgoing, omitted='none'):
    pvalues = upper_tail(null, observed)
    reference = data['recurrence'].set_index('basin').loc[data['labels']]
    return pd.DataFrame({'assignment_id': data['id'], 'participant_omitted': omitted, 'basin': data['labels'], 'n_events_retained': event_counts, 'outgoing_transitions': outgoing, 'successor_centrality': observed, 'null_mean': null.mean(0), 'null_sd': null.std(0, ddof=1), 'upper_tail_permutation_p': pvalues, 'bh_q_within_assignment_condition': bh_fdr(pvalues), 'full_event_centrality': reference.successor_centrality.to_numpy(), 'full_event_p': reference.successor_centrality_upper_tail_p.to_numpy(), 'full_event_bh_q': reference.successor_centrality_bh_q.to_numpy(), 'delta_centrality_from_full_event': observed - reference.successor_centrality.to_numpy(), 'n_permutations': len(null), 'seed': SEED, 'gamma': GAMMA})

def omission_control(data, out, omitted, n_perm=N_PERM):
    k, labels = (data['K'], data['labels'])
    sequences = {p: v for p, v in data['sequences'].items() if p != omitted}
    events = data['events'].loc[data['events'].participant_id_norm.ne(omitted)].copy()
    counts = counts_from_sequences(sequences, k)
    p = row_normalize(counts)
    m = successor(p)
    state_counts = np.bincount(events.state.to_numpy(int) - 1, minlength=k)
    support = counts.sum(1).astype(int)
    bank = permutation_bank(k, n_perm)
    geom_rows = []
    nulls = {'permutation': np.arange(1, n_perm + 1)}
    for d in DIMENSIONS:
        full_centroid = centroids(data['events'], d, k)
        retained_centroid = centroids(events, d, k)
        frozen = data['geometry'].loc[data['geometry'].geometry_D.eq(d) & data['geometry'].representation.eq('full_profile')].iloc[0]
        fixed, nfixed = correspondence(distances(full_centroid), distances(m), bank)
        specific, nspecific = correspondence(distances(retained_centroid), distances(m), bank)
        for name, stats in [('fixed_full_geometry', fixed), ('omission_specific_geometry', specific)]:
            adjustment = specific['pearson_r'] - fixed['pearson_r'] if name == 'omission_specific_geometry' else 0.0
            geom_rows.append({'assignment_id': data['id'], 'participant_omitted': omitted, 'n_participants_retained': len(sequences), 'n_events_retained': len(events), 'n_transitions_retained': int(counts.sum()), 'geometry_D': d, 'geometry_condition': name, 'gamma': GAMMA, 'K': k, 'n_pairs': k * (k - 1) // 2, **stats, 'full_sample_reference_r': frozen.pearson_r, 'full_sample_reference_p': frozen.two_sided_common_label_permutation_p, 'delta_r_from_full': stats['pearson_r'] - frozen.pearson_r, 'transition_effect_fixed_geometry_delta_r': fixed['pearson_r'] - frozen.pearson_r, 'geometry_adjustment_delta_r': adjustment, 'zero_outgoing_rows': int((support == 0).sum()), 'absent_basins': int((state_counts == 0).sum()), 'n_permutations': n_perm, 'seed': SEED})
        nulls[f'D{d}_fixed_full_geometry'] = nfixed
        nulls[f'D{d}_omission_specific_geometry'] = nspecific
        write_csv(pd.DataFrame(retained_centroid, index=labels, columns=[f'Dim{i}' for i in range(d)]), out / f'retained_centroids_D{d}.csv', True)
    progress = lambda n: print(f"[{data['id']}/omit_{omitted}] recurrence {n}/{n_perm}", flush=True)
    null, zero_rows = recurrence_null(sequences, k, n_perm, progress)
    rec = centrality_table(data, m.mean(0), null, state_counts, support, omitted)
    write_csv(rec, out / 'basin_recurrence.csv')
    for i, label in enumerate(labels):
        nulls[f'centrality_{label}'] = null[:, i]
    nulls['zero_outgoing_rows_in_recurrence_null'] = zero_rows
    write_csv(pd.DataFrame(nulls), out / 'permutation_nulls.csv')
    write_csv(pd.DataFrame(geom_rows), out / 'geometry_correspondence.csv')
    write_matrix(counts, labels, out / 'transition_counts.csv')
    write_matrix(p, labels, out / 'transition_probabilities.csv')
    write_matrix(m, labels, out / 'successor_matrix.csv')
    write_csv(events[['participant_id_norm', 'hippo_id_norm', 'state']], out / 'retained_memberships.csv')
    write_csv(data['edges'].loc[data['edges'].participant_id_norm.ne(omitted)], out / 'retained_event_edges.csv')
    write_csv(pd.DataFrame({'basin': labels, 'retained_events': state_counts, 'outgoing': support, 'incoming': counts.sum(0).astype(int), 'self_transitions': np.diag(counts).astype(int)}), out / 'basin_support.csv')
    write_json(out / 'CONTROL_METADATA.json', {'assignment_id': data['id'], 'participant_omitted': omitted, 'events_retained': len(events), 'transitions_retained': int(counts.sum()), 'zero_outgoing_basins': [labels[i] for i in np.flatnonzero(support == 0)], 'absent_basins': [labels[i] for i in np.flatnonzero(state_counts == 0)], 'recurrence_nulls_with_zero_rows': int((zero_rows > 0).sum()), 'MCA_refitted': False, 'basin_assignments_refitted': False, 'smoothing': False, 'geometry_delta_is_descriptive': True})

def persistence_and_visit_controls(data, out, n_perm=N_PERM):
    k, labels, sequences = (data['K'], data['labels'], data['sequences'])
    c, p = (data['C'], data['P'])
    visit_counts = counts_from_sequences(sequences, k, collapse=True)
    no_self = c.copy()
    np.fill_diagonal(no_self, 0)
    require(np.array_equal(visit_counts, no_self), 'Visit collapse failed the off-diagonal identity.')
    vp = row_normalize(visit_counts)
    vm = successor(vp)
    progress = lambda n: print(f"[{data['id']}/sequence_controls] {n}/{n_perm}", flush=True)
    null = sequence_nulls(sequences, k, n_perm, progress)
    self_counts, self_probs = (np.diag(c).copy(), np.diag(p).copy())
    observed_total = int(self_counts.sum())
    null_total = null['self_counts'].sum(1)
    global_p = float(upper_tail(null_total, observed_total))
    visit_p = float((1 + np.count_nonzero(null['visit_transitions'] <= visit_counts.sum())) / (n_perm + 1))
    require(global_p == visit_p, 'Self-transition and visit-count complement tests disagree.')
    persistence = {'assignment_id': data['id'], 'basin': labels, 'n_source_transitions': c.sum(1).astype(int)}
    for name, obs, draws in [('self_transition_count', self_counts, null['self_counts']), ('self_transition_probability', self_probs, null['self_probabilities'])]:
        pvalues = upper_tail(draws, obs)
        persistence[name] = obs
        persistence[name + '_null_mean'] = draws.mean(0)
        persistence[name + '_null_sd'] = draws.std(0, ddof=1)
        persistence[name + '_upper_tail_p'] = pvalues
        persistence[name + '_bh_q'] = bh_fdr(pvalues)
    write_csv(pd.DataFrame(persistence), out / 'basin_persistence.csv')
    write_csv(pd.DataFrame([{'assignment_id': data['id'], 'event_transitions': int(c.sum()), 'self_transitions': observed_total, 'visit_transitions': int(visit_counts.sum()), 'null_self_mean': float(null_total.mean()), 'null_self_sd': float(null_total.std(ddof=1)), 'self_transition_upper_tail_p': global_p, 'visit_transition_lower_tail_p': visit_p, 'n_permutations': n_perm, 'seed': SEED}]), out / 'overall_persistence.csv')
    event_counts = np.bincount(data['events'].state.to_numpy(int) - 1, minlength=k)
    write_csv(centrality_table(data, vm.mean(0), null['visit_centrality'], event_counts, visit_counts.sum(1).astype(int)), out / 'visit_basin_recurrence.csv')
    bank = permutation_bank(k, n_perm)
    null_columns = {'permutation': np.arange(1, n_perm + 1), 'total_self_transitions': null_total, 'visit_transitions': null['visit_transitions'], 'event_zero_rows': null['event_zero_rows'], 'visit_zero_rows': null['visit_zero_rows']}
    geometry_rows = []
    for d in DIMENSIONS:
        geom = distances(centroids(data['events'], d, k))
        stats, draws = correspondence(geom, distances(vm), bank)
        ref = data['geometry'].loc[data['geometry'].geometry_D.eq(d) & data['geometry'].representation.eq('full_profile')].iloc[0]
        geometry_rows.append({'assignment_id': data['id'], 'geometry_D': d, 'gamma': GAMMA, 'K': k, 'n_pairs': k * (k - 1) // 2, 'transition_unit': 'basin_visit', **stats, 'event_adjacency_r': ref.pearson_r, 'event_adjacency_p': ref.two_sided_common_label_permutation_p, 'delta_r_visit_minus_event': stats['pearson_r'] - ref.pearson_r, 'event_transitions': int(c.sum()), 'self_transitions_removed': observed_total, 'visit_transitions': int(visit_counts.sum()), 'zero_outgoing_rows': int((visit_counts.sum(1) == 0).sum()), 'n_permutations': n_perm, 'seed': SEED})
        null_columns[f'D{d}_visit_geometry_r'] = draws
    for i, label in enumerate(labels):
        for name, key in [('self_count', 'self_counts'), ('self_probability', 'self_probabilities'), ('visit_centrality', 'visit_centrality')]:
            null_columns[f'{name}_{label}'] = null[key][:, i]
    write_csv(pd.DataFrame(null_columns), out / 'permutation_nulls.csv')
    write_csv(pd.DataFrame(geometry_rows), out / 'visit_geometry_correspondence.csv')
    write_matrix(visit_counts, labels, out / 'visit_transition_counts.csv')
    write_matrix(vp, labels, out / 'visit_transition_probabilities.csv')
    write_matrix(vm, labels, out / 'visit_successor_matrix.csv')
    manifest, visit_rows, visit_edges = ([], [], [])
    for participant, frame in data['sequence_manifest'].groupby('participant_id_norm', sort=True):
        frame = frame.sort_values('sequence_position')
        states = frame.state.to_numpy(int)
        ids = frame.hippo_id_norm.tolist()
        starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
        stops = np.r_[starts[1:], len(states)]
        for i, (start, stop) in enumerate(zip(starts, stops)):
            visit_rows.append({'participant_id_norm': participant, 'visit_index': i + 1, 'state': int(states[start]), 'first_event': ids[start], 'last_event': ids[stop - 1], 'n_events_in_visit': int(stop - start)})
            if i + 1 < len(starts):
                nxt = starts[i + 1]
                visit_edges.append({'participant_id_norm': participant, 'source_visit': i + 1, 'target_visit': i + 2, 'source_state': int(states[start]), 'target_state': int(states[nxt]), 'source_boundary_event': ids[stop - 1], 'target_boundary_event': ids[nxt]})
        manifest.append({'participant_id_norm': participant, 'events': len(frame), 'visits': len(starts), 'visit_transitions': max(len(starts) - 1, 0), 'self_transitions_removed': len(frame) - len(starts)})
    require(len(visit_edges) == int(visit_counts.sum()), 'Visit edge export does not match counts.')
    write_csv(pd.DataFrame(manifest), out / 'visit_sequence_summary.csv')
    write_csv(pd.DataFrame(visit_rows), out / 'basin_visits.csv')
    write_csv(pd.DataFrame(visit_edges), out / 'visit_transition_records.csv')
    write_json(out / 'CONTROL_METADATA.json', {'assignment_id': data['id'], 'K': k, 'gamma': GAMMA, 'collapse_within_each_participant': True, 'collapse_after_each_null_shuffle': True, 'event_occupancy_preserved_in_null': True, 'visit_counts_can_vary_under_null': True, 'visit_counts_equal_event_counts_with_diagonal_zeroed': True, 'persistence_nulls_with_zero_rows': int((null['event_zero_rows'] > 0).sum()), 'visit_nulls_with_zero_rows': int((null['visit_zero_rows'] > 0).sum()), 'zero_outgoing_visit_basins': [labels[i] for i in np.flatnonzero(visit_counts.sum(1) == 0)], 'basin_assignments_refitted': False, 'smoothing': False})
