"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
from runtime import run_root, atomic_json
import json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
ROOT = run_root()
OUT = ROOT / 'results'
INP = ROOT / 'inputs'
events = pd.read_csv(INP / 'new_events.csv', keep_default_na=False)
edges = pd.read_csv(INP / 'event_edges.csv')
keys = ['participant_id_norm', 'hippo_id_norm']
lookup = {tuple(r): i for i, r in enumerate(events[keys].to_numpy())}
src = np.array([lookup[r.participant_id_norm, r.source_event] for r in edges.itertuples()])
dst = np.array([lookup[r.participant_id_norm, r.target_event] for r in edges.itertuples()])
features = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
f = events[features].copy()
for c in features:
    x = f[c].fillna('none').astype(str).str.strip()
    f[c] = x.mask(x.str.lower().isin(['', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing']), 'none')
_, atom = np.unique(f.to_numpy(str), axis=0, return_inverse=True)
X = events[[f'Dim{i}' for i in range(4)]].to_numpy()
assert len(events) == 256 and len(edges) == 251

def countcheck(lab, idx, k, lower, upper, row):
    assert np.array_equal(np.unique(lab), np.arange(k))
    count = np.bincount(lab, minlength=k)
    assert count.min() >= lower and count.max() <= upper
    all_labels = np.full(256, -1)
    all_labels[idx] = lab
    keep = (all_labels[src] >= 0) & (all_labels[dst] >= 0)
    out = np.bincount(all_labels[src[keep]], minlength=k)
    assert len(idx) == row['n_events'] and keep.sum() == row['n_edges']
    assert count.min() == row['n_min'] and count.max() == row['n_max'] and (out.min() == row['min_outgoing'])
    assert bool(out.min() >= 8) == row['eligible']
    for j in np.unique(atom[idx]):
        assert len(np.unique(lab[atom[idx] == j])) == 1
s = pd.read_csv(OUT / 'Screen.csv')
a = pd.read_csv(OUT / 'Screen_Assignments.csv')
assert len(s) == 2496 and (not s.duplicated(['dim', 'K', 'lower', 'upper']).any())
for r in s[s.available].to_dict('records'):
    lab = a[r['id']].to_numpy() - 1
    countcheck(lab, np.arange(256), r['K'], r['lower'], r['upper'], r)
    z = X[:, [0, 1, 2, 3] if r['dim'] == 4 else [0, 3]]
    centers = np.array([z[lab == j].mean(0) for j in range(r['K'])])
    sse = ((z - centers[lab]) ** 2).sum()
    assert np.isclose(sse, r['fit_SSE'], atol=1e-08, rtol=1e-10)
samples = {(v['scheme'], v['repeat']): v['retained_indices'] for v in json.loads((INP / 'PERTURBATIONS.json').read_text())}
samples['actual_added_events', 0] = np.flatnonzero(~events.hippo_id_norm.isin(['p02-s3-h012', 'p03-s5-h007'])).tolist()
configs = pd.read_csv(OUT / 'Pilot_Configurations.csv').set_index('id')
full_count = 0
for p in (OUT / 'full').glob('*/Fit.json'):
    v = json.loads(p.read_text())
    if not v['available']:
        continue
    c = configs.loc[v['id']]
    lab = np.array(v['labels'])
    countcheck(lab, np.arange(256), int(c.K), int(c.lower), int(c.upper), v['summary'])
    z = X[:, [0, 1, 2, 3] if c.dim == 4 else [0, 3]]
    centers = np.array([z[lab == j].mean(0) for j in range(int(c.K))])
    assert np.isclose(((z - centers[lab]) ** 2).sum(), v['summary']['fit_SSE'], atol=1e-08, rtol=1e-10)
    full_count += 1
maximum_error = 0.0
trialcounts = {}
selected = json.loads((OUT / 'Frozen_Shortlist.json').read_text())
for stage in ['pilot', 'validation']:
    t = pd.read_csv(OUT / stage / 'Trials.csv')
    seen = set()
    n = 0
    stage_configs = pd.read_csv(OUT / ('Pilot_Configurations.csv' if stage == 'pilot' else 'Validation_Configurations.csv'))
    expected_ids = [cid for cid in stage_configs.id if json.loads((OUT / 'full' / cid / 'Fit.json').read_text())['available']]
    if stage == 'pilot':
        stage_samples = [key for key in samples if key[0] != 'actual_added_events' and key[1] < 10]
    else:
        stage_samples = [key for key in samples if key[0] == 'actual_added_events' or key[1] >= 10]
    expected_keys = {(cid, scheme, repeat, mode) for cid in expected_ids for scheme, repeat in stage_samples for mode in (['refitted'] if stage == 'pilot' else ['fixed', 'refitted'])}
    for p in (OUT / stage / 'checkpoints').glob('*.json'):
        v = json.loads(p.read_text())
        r = v['summary']
        key = (r['id'], r['scheme'], r['repeat'], r['mode'])
        assert key not in seen
        seen.add(key)
        if not r['available']:
            continue
        idx = np.array(v['retained_indices'])
        assert idx.tolist() == samples[r['scheme'], r['repeat']]
        if stage == 'pilot':
            assert r['repeat'] < 10 and r['scheme'] != 'actual_added_events'
        elif r['scheme'] != 'actual_added_events':
            assert r['repeat'] >= 10
        lab = np.array(v['labels'])
        full = json.loads((OUT / 'full' / r['id'] / 'Fit.json').read_text())
        ref = np.array(full['labels'])[idx]
        countcheck(lab, idx, r['K'], r['lower'], r['upper'], r)
        ari = adjusted_rand_score(ref, lab)
        maximum_error = max(maximum_error, abs(ari - r['ARI']))
        tab = np.zeros((r['K'], r['K']), int)
        np.add.at(tab, (ref, lab), 1)
        rr, cc = linear_sum_assignment(-tab)
        mapping = np.zeros(r['K'], int)
        mapping[cc] = rr
        b = mapping[lab]
        assert np.sum(ref != b) == r['changed']
        jac = np.array([np.sum((ref == j) & (b == j)) / np.sum((ref == j) | (b == j)) for j in range(r['K'])])
        assert np.allclose(jac, v['reference_state_Jaccard'], atol=1e-12)
        assert np.isclose(jac.min(), r['min_state_Jaccard'], atol=1e-12)
        if r['scheme'] == 'actual_added_events':
            assert r['n_events'] == 254 and r['n_edges'] == 247
        n += 1
    assert len(t) == len(seen)
    assert seen == expected_keys, 'Missing or unexpected omission checkpoint(s)'
    assert set(map(tuple, t[['id', 'scheme', 'repeat', 'mode']].to_numpy())) == expected_keys
    trialcounts[stage] = {'attempts': len(t), 'available': n}
    expected_per_config = 20 if stage == 'pilot' else 262
    assert t.groupby('id').size().eq(expected_per_config).all()
assert maximum_error < 1e-12
final = pd.read_csv(OUT / 'Final_Event_Assignments.csv')
short = pd.read_csv(OUT / 'Final_Shortlist.csv')
assert final[keys].equals(events[keys]) and len(short) <= 5
assert hashlib.sha256((OUT / 'Final_Event_Assignments.csv').read_bytes()).hexdigest() == selected['assignments_hash']
for r in short.to_dict('records'):
    full = json.loads((OUT / 'full' / r['id'] / 'Fit.json').read_text())
    lab = final[r['id']].to_numpy() - 1
    assert np.array_equal(lab, full['labels'])
    countcheck(lab, np.arange(256), r['K'], r['lower'], r['upper'], full['summary'])
    assert r['completion_min'] >= 0.95 and r['support_min'] >= 0.8
    for other in short[short.dim.eq(r['dim']) & short.id.ne(r['id'])].id:
        assert adjusted_rand_score(lab, final[other].to_numpy()) < 0.95
result = {'passed': True, 'screen_cells': len(s), 'screen_available': int(s.available.sum()), 'screen_eligible': int(s.eligible.fillna(False).sum()), 'refined_full_fits_verified': full_count, 'trial_counts': trialcounts, 'max_independent_ARI_error': maximum_error, 'identical_profiles_intact': True, 'pilot_and_validation_sample_ids_disjoint': True, 'all_full_events': 256, 'all_full_edges': 251, 'actual_omission_events': 254, 'actual_omission_original_edges': 247, 'frozen_shortlist_hash_verified': True, 'shortlist_size': len(short)}
atomic_json(OUT / 'Independent_Verification.json', result)
print(json.dumps(result, indent=2))
