"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import numerics as n
import routing_methods as routing

def retained_data(data, omitted):
    out = dict(data)
    out['events'] = data['events'].loc[data['events'].participant_id_norm.ne(omitted)].copy()
    out['edges'] = data['edges'].loc[data['edges'].participant_id_norm.ne(omitted)].copy()
    out['sequence_manifest'] = data['sequence_manifest'].loc[data['sequence_manifest'].participant_id_norm.ne(omitted)].copy()
    out['sequences'] = {p: v for p, v in data['sequences'].items() if p != omitted}
    out['C'] = n.counts_from_sequences(out['sequences'], data['K'])
    out['P'] = n.row_normalize(out['C'])
    return out

def export_visits(data, folder):
    rows, edges = ([], [])
    for participant, frame in data['sequence_manifest'].groupby('participant_id_norm', sort=True):
        frame = frame.sort_values('sequence_position')
        states = frame.state.to_numpy(int)
        ids = frame.hippo_id_norm.tolist()
        starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
        stops = np.r_[starts[1:], len(states)]
        for i, (a, b) in enumerate(zip(starts, stops)):
            rows.append({'participant_id_norm': participant, 'visit_index': i + 1, 'state': int(states[a]), 'first_event': ids[a], 'last_event': ids[b - 1], 'n_events': int(b - a)})
            if i + 1 < len(starts):
                edges.append({'participant_id_norm': participant, 'source_visit': i + 1, 'target_visit': i + 2, 'source_event': ids[b - 1], 'target_event': ids[starts[i + 1]], 'source_state': int(states[a]), 'target_state': int(states[starts[i + 1]])})
    n.write_csv(pd.DataFrame(rows), folder / 'basin_visits.csv')
    n.write_csv(pd.DataFrame(edges), folder / 'visit_edges.csv')
    return (len(rows), len(edges))

def crossed_visits(data, controls, folder, omitted, n_perm=n.N_PERM):
    retained = retained_data(data, omitted)
    k = data['K']
    labels = data['labels']
    counts = n.counts_from_sequences(retained['sequences'], k, collapse=True)
    expected = retained['C'].copy()
    np.fill_diagonal(expected, 0)
    n.require(np.array_equal(counts, expected), 'Omitted visit counts do not preserve between-basin boundaries.')
    p = n.row_normalize(counts)
    m = n.successor(p)
    bank = n.permutation_bank(k, n_perm)
    previous = pd.read_csv(controls / 'comparisons/participant_exclusion_geometry.csv')
    previous = previous.loc[previous.assignment_id.eq(data['id']) & previous.participant_omitted.eq(omitted)]
    full = pd.read_csv(controls / 'comparisons/visit_geometry.csv')
    full = full.loc[full.assignment_id.eq(data['id'])]
    nulls = {'permutation': np.arange(1, n_perm + 1)}
    rows = []
    for d in n.DIMENSIONS:
        reference = full.loc[full.geometry_D.eq(d)].iloc[0]
        for condition, events in [('fixed_full_geometry', data['events']), ('omission_specific_geometry', retained['events'])]:
            points = n.centroids(events, d, k)
            stats, draws = n.correspondence(n.distances(points), n.distances(m), bank)
            event_ref = previous.loc[previous.geometry_D.eq(d) & previous.geometry_condition.eq(condition)].iloc[0]
            rows.append({'assignment_id': data['id'], 'participant_omitted': omitted, 'K': k, 'geometry_D': d, 'geometry_condition': condition, 'transition_unit': 'basin_visit', 'gamma': n.GAMMA, **stats, 'full_visit_r': reference.pearson_r, 'full_visit_p': reference.permutation_p, 'omitted_event_r': event_ref.pearson_r, 'omitted_event_p': event_ref.permutation_p, 'delta_r_from_full_visit': stats['pearson_r'] - reference.pearson_r, 'delta_r_from_omitted_event': stats['pearson_r'] - event_ref.pearson_r, 'n_events': len(retained['events']), 'n_event_edges': int(retained['C'].sum()), 'n_visit_edges': int(counts.sum()), 'n_permutations': n_perm, 'seed': n.SEED})
            nulls[f'D{d}_{condition}'] = draws
            n.write_csv(pd.DataFrame(points, index=labels, columns=[f'Dim{i}' for i in range(d)]), folder / f'centroids_D{d}_{condition}.csv', True)
    rng = np.random.default_rng(n.SEED)
    draws = np.empty((n_perm, k))
    zero = np.empty(n_perm, int)
    for i in range(n_perm):
        shuffled = {pid: rng.permutation(v) for pid, v in retained['sequences'].items()}
        c = n.counts_from_sequences(shuffled, k, collapse=True)
        zero[i] = np.count_nonzero(c.sum(1) == 0)
        draws[i] = n.successor(n.row_normalize(c)).mean(0)
        if (i + 1) % 2000 == 0:
            print(f"[{data['id']}/visit_omit_{omitted}] {i + 1}/{n_perm}", flush=True)
    pvalues = n.upper_tail(draws, m.mean(0))
    fullrec = pd.read_csv(controls / 'comparisons/visit_recurrence.csv')
    fullrec = fullrec.loc[fullrec.assignment_id.eq(data['id'])].set_index('basin').loc[labels]
    eventrec = pd.read_csv(controls / 'comparisons/participant_exclusion_recurrence.csv')
    eventrec = eventrec.loc[eventrec.assignment_id.eq(data['id']) & eventrec.participant_omitted.eq(omitted)].set_index('basin').loc[labels]
    support = np.bincount(retained['events'].state.to_numpy(int) - 1, minlength=k)
    rec = pd.DataFrame({'assignment_id': data['id'], 'participant_omitted': omitted, 'basin': labels, 'n_events_retained': support, 'outgoing_visit_edges': counts.sum(1).astype(int), 'successor_centrality': m.mean(0), 'null_mean': draws.mean(0), 'null_sd': draws.std(0, ddof=1), 'upper_tail_p': pvalues, 'bh_q_within_assignment_omission': n.bh_fdr(pvalues), 'full_visit_centrality': fullrec.successor_centrality.to_numpy(), 'full_visit_p': fullrec.upper_tail_permutation_p.to_numpy(), 'full_visit_bh_q': fullrec.bh_q_within_assignment_condition.to_numpy(), 'omitted_event_centrality': eventrec.successor_centrality.to_numpy(), 'omitted_event_bh_q': eventrec.bh_q_within_assignment_condition.to_numpy(), 'gamma': n.GAMMA, 'n_permutations': n_perm, 'seed': n.SEED})
    for i, b in enumerate(labels):
        nulls[f'visit_centrality_{b}'] = draws[:, i]
    nulls['zero_outgoing_rows_in_recurrence_null'] = zero
    n.write_csv(pd.DataFrame(nulls), folder / 'permutation_nulls.csv')
    n.write_csv(pd.DataFrame(rows), folder / 'geometry_correspondence.csv')
    n.write_csv(rec, folder / 'basin_recurrence.csv')
    for name, a in [('transition_counts', counts), ('transition_probabilities', p), ('successor_matrix', m)]:
        n.write_matrix(a, labels, folder / f'{name}.csv')
    n.write_csv(retained['events'][['participant_id_norm', 'hippo_id_norm', 'state']], folder / 'retained_memberships.csv')
    n.write_csv(retained['edges'], folder / 'retained_event_edges.csv')
    visits, edges = export_visits(retained, folder)
    n.require(edges == counts.sum(), 'Visit edge export mismatch.')
    n.write_json(folder / 'DIAGNOSTICS.json', {'absent_basins': [labels[i] for i in np.flatnonzero(support == 0)], 'zero_outgoing_basins': [labels[i] for i in np.flatnonzero(counts.sum(1) == 0)], 'nulls_with_zero_outgoing_rows': int((zero > 0).sum()), 'events_retained': len(retained['events']), 'visits_retained': visits, 'visit_edges': edges, 'collapse_after_null_shuffle': True})

def order_control(data, folder, n_perm=n.N_PERM):
    k = data['K']
    labels = data['labels']
    bank = n.permutation_bank(k, n_perm)
    upper = np.triu_indices(k, 1)
    geometry = {d: n.distances(n.centroids(data['events'], d, k)) for d in n.DIMENSIONS}
    counts = {'narrated': data['C'], 'reversed': n.counts_from_sequences({p: v[::-1] for p, v in data['sequences'].items()}, k)}
    n.require(np.array_equal(counts['reversed'], data['C'].T), 'Reversed counts must equal the forward transpose.')
    rng = np.random.default_rng(n.SEED)
    shuffle = np.empty((n_perm, len(n.DIMENSIONS)))
    zero = np.zeros(n_perm, int)
    for i in range(n_perm):
        sequences = {p: v[rng.permutation(len(v))] for p, v in data['sequences'].items()}
        c = n.counts_from_sequences(sequences, k)
        zero[i] = np.count_nonzero(c.sum(1) == 0)
        distance = n.distances(n.successor(n.row_normalize(c)))
        for j, d in enumerate(n.DIMENSIONS):
            shuffle[i, j] = np.corrcoef(geometry[d][upper], distance[upper])[0, 1]
        if (i + 1) % 2000 == 0:
            print(f"[{data['id']}/event_order] {i + 1}/{n_perm}", flush=True)
    rows = []
    nulls = {'permutation': np.arange(1, n_perm + 1), 'shuffle_zero_outgoing_rows': zero}
    for j, d in enumerate(n.DIMENSIONS):
        nulls[f'D{d}_order_shuffle_r'] = shuffle[:, j]
    for order, c in counts.items():
        p = n.row_normalize(c)
        m = n.successor(p)
        for name, a in [('counts', c), ('probabilities', p), ('successor', m)]:
            n.write_matrix(a, labels, folder / f'{order}_{name}.csv')
        for j, d in enumerate(n.DIMENSIONS):
            stats, draws = n.correspondence(geometry[d], n.distances(m), bank)
            null = shuffle[:, j]
            rows.append({'assignment_id': data['id'], 'order': order, 'geometry_D': d, 'K': k, 'gamma': n.GAMMA, 'pearson_r': stats['pearson_r'], 'label_permutation_p_two_sided': stats['permutation_p'], 'status': stats['status'], 'order_shuffle_p_upper_tail': float(n.upper_tail(null, stats['pearson_r'])), 'shuffle_null_mean_r': float(null.mean()), 'shuffle_null_sd_r': float(null.std(ddof=1)), 'shuffle_null_q025': float(np.quantile(null, 0.025)), 'shuffle_null_q975': float(np.quantile(null, 0.975)), 'n_permutations': n_perm, 'seed': n.SEED})
            nulls[f'{order}_D{d}_label_r'] = draws
            if order == 'narrated':
                ref = data['geometry'].loc[data['geometry'].geometry_D.eq(d) & data['geometry'].representation.eq('full_profile')].iloc[0]
                np.testing.assert_allclose(stats['pearson_r'], ref.pearson_r, rtol=0, atol=1e-11)
    n.write_csv(pd.DataFrame(rows), folder / 'order_correspondence.csv')
    n.write_csv(pd.DataFrame(nulls), folder / 'permutation_nulls.csv')
    manifest = data['sequence_manifest'].copy()
    manifest['reversed_sequence_position'] = manifest.groupby('participant_id_norm').sequence_position.transform('max') + 1 - manifest.sequence_position
    n.write_csv(manifest, folder / 'sequence_manifest_forward_and_reverse.csv')
    n.write_json(folder / 'DIAGNOSTICS.json', {'reversed_count_transpose_verified': True, 'event_occupancy_preserved': True, 'same_shuffle_null_for_forward_and_reverse': True, 'nulls_with_zero_outgoing_rows': int((zero > 0).sum()), 'formal_forward_reverse_difference_test': False})

def profile_controls(data, folder, n_perm=n.N_PERM):
    p = data['P']
    m = n.successor(p)
    eye = np.eye(data['K'])
    bank = n.permutation_bank(data['K'], n_perm)
    profiles = {'P': p, 'I_plus_gammaP': eye + n.GAMMA * p, 'M': m, 'M_minus_I': m - eye}
    rows = []
    nulls = {'permutation': np.arange(1, n_perm + 1)}
    for name, values in profiles.items():
        n.write_matrix(values, data['labels'], folder / f'profile_{name}.csv')
        for d in n.DIMENSIONS:
            stats, draws = n.correspondence(n.distances(n.centroids(data['events'], d, data['K'])), n.distances(values), bank)
            rows.append({'assignment_id': data['id'], 'profile': name, 'geometry_D': d, 'K': data['K'], 'gamma': n.GAMMA, **stats, 'n_permutations': n_perm, 'seed': n.SEED})
            nulls[f'D{d}_{name}'] = draws
    n.write_csv(pd.DataFrame(rows), folder / 'profile_correspondence.csv')
    n.write_csv(pd.DataFrame(nulls), folder / 'permutation_nulls.csv')
    n.write_json(folder / 'DIAGNOSTICS.json', {'identity_alone_pairwise_distance': float(np.sqrt(2)), 'identity_alone_correlation': 'undefined_constant_distances', 'profile_differences_are_descriptive': True, 'transition_unit': 'event_adjacency'})

def routing_omission(data, baseline, folder, omitted, n_perm=n.N_PERM):
    retained = retained_data(data, omitted)
    k = data['K']
    labels = data['labels']
    retained['transitions'] = retained['edges'].copy()
    retained['transitions'][['source_state', 'target_state']] -= 1
    n.write_csv(retained['events'], folder / 'retained_events.csv')
    n.write_csv(retained['edges'], folder / 'retained_event_edges.csv')
    for name, a in [('event_counts', retained['C']), ('event_probabilities', retained['P'])]:
        n.write_matrix(a, labels, folder / f'{name}.csv')
    feature_rows = []
    for mode in ['primary3', 'all11']:
        dest = folder / f'feature_{mode}'
        dest.mkdir()
        routing.feature_routing(retained, dest, n_perm, mode)
        row = pd.read_csv(dest / 'feature_routing_stats.csv').iloc[0].to_dict()
        ref = pd.read_csv(baseline / data['id'] / f'feature_{mode}/feature_routing_stats.csv').iloc[0]
        row.update({'assignment_id': data['id'], 'participant_omitted': omitted, 'n_events_retained': len(retained['events']), 'n_edges_retained': len(retained['edges']), 'full_sample_r': ref.pearson_r, 'full_sample_p': ref.pearson_permutation_p_two_sided, 'delta_r_from_full': row['pearson_r'] - ref.pearson_r, 'feature_counts_recomputed_from_retained_events': True})
        feature_rows.append(row)
    all_rows = []
    support_rows = []
    for family in routing.FAMILIES:
        dest = folder / f'conditioned_{family}'
        dest.mkdir()
        transitions, payloads = routing.conditioned_payloads(retained['transitions'], family)
        candidates = sorted({v for payload in payloads for v in payload})
        totals = {v: sum((v in payload for payload in payloads)) for v in candidates}
        full_conditions = pd.read_csv(baseline / data['id'] / f'conditioned_{family}/condition_summary.csv').condition.tolist()
        for condition in sorted(set(candidates) | set(full_conditions)):
            support_rows.append({'assignment_id': data['id'], 'participant_omitted': omitted, 'family': family, 'condition': condition, 'retained_source_transitions': totals.get(condition, 0), 'included_after_omission': totals.get(condition, 0) >= 10, 'included_in_full_sample': condition in full_conditions})
        if sum((v >= 10 for v in totals.values())) < 2:
            rows = pd.DataFrame([{'field_key': family, 'test_family': test, 'observed_stat': np.nan, 'null_mean': np.nan, 'lower_tail_p_value': np.nan, 'n_valid_nulls': 0, 'n_perm': n_perm, 'seed': n.SEED, 'n_conditions': sum((v >= 10 for v in totals.values())), 'n_basins': k, 'min_condition_transitions': 10, 'min_source_outgoing': 3, 'status': 'not_estimable_fewer_than_two_conditions'} for test in ['full_matrix', 'supported_source']])
            n.write_csv(rows, dest / 'conditioned_routing_results.csv')
        else:
            routing.conditioned_routing(retained, dest, n_perm, family)
            rows = pd.read_csv(dest / 'conditioned_routing_results.csv')
        all_rows.append(rows)
    conditioned = pd.concat(all_rows, ignore_index=True)
    conditioned['assignment_id'] = data['id']
    conditioned['participant_omitted'] = omitted
    conditioned['n_events_retained'] = len(retained['events'])
    conditioned['n_edges_retained'] = len(retained['edges'])
    conditioned['bh_q_three_families'] = np.nan
    for test, ix in conditioned.groupby('test_family').groups.items():
        pvalues = conditioned.loc[ix, 'lower_tail_p_value'].to_numpy(float)
        valid = np.isfinite(pvalues)
        q = n.bh_fdr(np.where(valid, pvalues, 1.0))
        q[~valid] = np.nan
        conditioned.loc[ix, 'bh_q_three_families'] = q
    reference = pd.read_csv(baseline / 'comparisons/conditioned_routing_all_assignments.csv')
    reference = reference.loc[reference.assignment_id.eq(data['id'])]
    refcols = ['field_key', 'test_family', 'observed_stat', 'null_mean', 'lower_tail_p_value', 'bh_q_within_assignment_and_test_family']
    ref = reference[refcols].rename(columns={x: 'full_sample_' + x for x in refcols if x not in ['field_key', 'test_family']})
    conditioned = conditioned.merge(ref, on=['field_key', 'test_family'], how='left', validate='one_to_one')
    conditioned['null_minus_observed'] = conditioned.null_mean - conditioned.observed_stat
    conditioned['full_sample_null_minus_observed'] = conditioned.full_sample_null_mean - conditioned.full_sample_observed_stat
    n.write_csv(pd.DataFrame(feature_rows), folder / 'feature_routing.csv')
    n.write_csv(conditioned, folder / 'conditioned_routing.csv')
    n.write_csv(pd.DataFrame(support_rows), folder / 'condition_support_changes.csv')
    n.write_json(folder / 'DIAGNOSTICS.json', {'participant_omitted': omitted, 'routing_transition_unit': 'event_adjacency', 'source_features_attached_to_original_event_edges': True, 'condition_threshold': 10, 'supported_source_threshold': 3, 'null': 'global condition-payload shuffle over retained source transitions; mechanism pairs move jointly', 'feature_profiles': 'recomputed from retained event features', 'BH_family_size': 3, 'zero_outgoing_basins': [labels[i] for i in np.flatnonzero(retained['C'].sum(1) == 0)]})
