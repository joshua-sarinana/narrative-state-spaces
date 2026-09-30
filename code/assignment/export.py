"""Deterministic analysis of the frozen 256-event dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import atomic_json, run_root
ROOT = run_root()
OUT = ROOT / 'results'
verified = json.loads((OUT / 'Independent_Verification.json').read_text())
assert verified['passed']
short = pd.read_csv(OUT / 'Final_Shortlist.csv')
memberships = pd.read_csv(OUT / 'Final_Event_Assignments.csv')
events = pd.read_csv(ROOT / 'inputs/new_events.csv', keep_default_na=False)
validation = pd.read_csv(OUT / 'validation/Summary.csv')
keys = ['participant_id_norm', 'hippo_id_norm']
assert memberships[keys].equals(events[keys])
exports, states, table = ([], [], [])

def summary(cid, scheme='omit5pct', mode='refitted'):
    return validation[validation.id.eq(cid) & validation.scheme.eq(scheme) & validation['mode'].eq(mode)].iloc[0]
for config in short.itertuples():
    cid = config.id
    full = json.loads((OUT / 'full' / cid / 'Fit.json').read_text())
    a = full['summary']
    b = summary(cid)
    c = summary(cid, 'actual_added_events')
    fixed = summary(cid, mode='fixed')
    actual_fixed = summary(cid, 'actual_added_events', 'fixed')
    exports.append({'assignment': cid, 'dimension': config.dim, 'K': config.K, 'minimum_bound': config.lower, 'maximum_bound': config.upper, 'actual_minimum_events': a['n_min'], 'actual_maximum_events': a['n_max'], 'minimum_outgoing': a['min_outgoing'], 'states_with_fewer_than_10_events': sum((v['n_events'] < 10 for v in full['states'])), 'states_with_fewer_than_10_outgoing': a['states_outgoing_lt10'], 'D4_explained_fraction': a['D4_explained_fraction'], 'fit_space_silhouette': a['fit_silhouette'], 'minimum_state_fit_silhouette': a['min_state_fit_silhouette'], 'nonnearest_fraction': a['nonnearest_fraction'], 'refit_13event_mean_ARI': b.ARI_mean, 'refit_13event_p10_ARI': b.ARI_p10, 'refit_13event_availability': b.availability, 'refit_13event_support_fraction': b.support_fraction, 'refit_13event_mean_min_state_Jaccard': b.min_state_Jaccard_mean, 'fixed_13event_mean_ARI': fixed.ARI_mean, 'refit_15_to_30_start_mean_ARI': b.half_to_full_ARI_mean, 'actual_refit_changed': c.changed_mean, 'actual_refit_ARI': c.ARI_mean, 'actual_fixed_changed': actual_fixed.changed_mean, 'actual_fixed_ARI': actual_fixed.ARI_mean})
    states.extend([{'assignment': cid, **s} for s in full['states']])
    folder = ROOT / 'selected' / cid
    folder.mkdir(parents=True, exist_ok=True)
    memberships[keys + [cid]].rename(columns={cid: 'state'}).to_csv(folder / 'Event_Memberships.csv', index=False)
    pd.DataFrame(full['states']).to_csv(folder / 'State_Diagnostics.csv', index=False)
    dimensions = [f'Dim{i}' for i in range(4)]
    centers = events[dimensions].groupby(memberships[cid]).mean()
    centers.index.name = 'state'
    centers.to_csv(folder / 'Centroids_D4.csv')
    atomic_json(folder / 'Assignment.json', {'assignment_id': cid, 'method': 'event-weighted profile-indivisible constrained kmeans', 'fit_axes': [0, 1, 2, 3] if config.dim == 4 else [0, 3], 'K': int(config.K), 'lower': int(config.lower), 'upper': int(config.upper), 'partition_hash': a['partition_hash'], 'full_sample_events': 256, 'full_sample_edges': 251, 'status': 'Exploratory retained basin solution', 'source': 'results/Frozen_Shortlist.json', 'downstream_results_computed': False})
    table.append(f"| {cid} | {a['n_min']}–{a['n_max']} | {a['min_outgoing']} | {b.ARI_mean:.6f} | {b.support_fraction:.1%} | {c.changed_mean:.0f}/254 |")
pd.DataFrame(exports, columns=list(exports[0]) if exports else ['assignment']).to_csv(ROOT / 'REFIT_Multiple_Assignment_Summary.csv', index=False)
memberships.to_csv(ROOT / 'REFIT_Multiple_Event_Assignments.csv', index=False)
pd.DataFrame(states, columns=list(states[0]) if states else ['assignment', 'state']).to_csv(OUT / 'Shortlist_State_Diagnostics.csv', index=False)
print('Exported selected memberships and state diagnostics.')
