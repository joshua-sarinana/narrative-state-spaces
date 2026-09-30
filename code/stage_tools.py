"""Input checks and deterministic exports joining analysis stages."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
KEYS = ['participant_id_norm', 'hippo_id_norm']
FEATURES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
COUNTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_events():
    e = pd.read_csv(ROOT / 'data/event_level_analysis_input.csv', keep_default_na=False)
    require(len(e) == 256 and (not e.duplicated(KEYS).any()), 'Expected 256 unique event keys.')
    require(e.groupby(KEYS[0]).size().to_dict() == COUNTS, 'Participant counts differ.')
    require(e.uid.nunique() == 21 and e.n_excerpts_event.sum() == 442, 'Segment/excerpt totals differ.')
    return e

def edges_from_events(e):
    lookup = e.set_index(KEYS)
    rows = []
    for p, g in e.groupby(KEYS[0], sort=True):
        links = {r.hippo_id_norm: (r.hippo_prev_global, r.hippo_next_global) for r in g.itertuples()}
        starts = [k for k, v in links.items() if not v[0]]
        require(len(starts) == 1, 'Expected one chain start.')
        seen = set()
        cur = starts[0]
        while cur:
            require(cur in links and cur not in seen, 'Broken/cyclic sequence link.')
            seen.add(cur)
            prev, nxt = links[cur]
            if prev:
                require(links[prev][1] == cur, 'Nonreciprocal previous link.')
            if nxt:
                require(nxt in links and links[nxt][0] == cur, 'Nonreciprocal next link.')
                rows.append({KEYS[0]: p, 'source_event': cur, 'target_event': nxt})
            cur = nxt
        require(len(seen) == len(g), 'Disconnected chain.')
    edges = pd.DataFrame(rows)
    require(len(edges) == 251, 'Expected 251 links.')
    within = sum((lookup.loc[(r.participant_id_norm, r.source_event), 'uid'] == lookup.loc[(r.participant_id_norm, r.target_event), 'uid'] for r in edges.itertuples()))
    require(within == 235, 'Expected 235 within-segment boundaries.')
    return edges

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def orient():
    p = ROOT / 'outputs/mca/tables/mca_full_54d_row_coordinates.csv'
    e = pd.read_csv(p)
    rows = e.set_index(KEYS[1])
    for rule in json.loads((ROOT / 'provenance/MCA_AXIS_ORIENTATION.json').read_text()):
        x = float(rows.loc[rule['anchor_event'], rule['axis']])
        require(abs(x) > 1e-12, 'Invalid orientation anchor.')
        e[rule['axis']] *= rule['expected_sign'] * np.sign(x)
    e.to_csv(p, index=False)

def assignment_inputs():
    e = read_events()
    edges = edges_from_events(e)
    c = pd.read_csv(ROOT / 'outputs/mca/tables/mca_full_54d_row_coordinates.csv')
    cols = KEYS + [f'Dim{i}' for i in range(4)]
    e = e[KEYS + FEATURES].merge(c[cols], on=KEYS, validate='one_to_one', sort=False)
    require(len(e) == 256, 'Coordinate join lost events.')
    out = ROOT / 'outputs/assignments/inputs'
    out.mkdir(parents=True, exist_ok=True)
    e.to_csv(out / 'new_events.csv', index=False)
    edges.to_csv(out / 'event_edges.csv', index=False)
    shutil.copyfile(ROOT / 'provenance/PERTURBATIONS.json', out / 'PERTURBATIONS.json')
    provenance = {'events':256, 'edges':251, 'row_order':'frozen public event-table order', 'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/event_level_analysis_input.csv', ROOT/'outputs/mca/tables/mca_full_54d_row_coordinates.csv']}}
    (out/'INPUT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')

def distance_summary():
    out=ROOT/'outputs/event_distance'
    people=pd.read_csv(out/'participant_event_distances_submission.csv')
    pooled=pd.read_csv(out/'pooled_event_distance_test.csv')
    cols=['participant','links','observed_mean_distance','null_expected_mean_exact','percent_shorter_than_null']
    people=people[cols].assign(summary_level='participant')
    whole=pooled.assign(participant='Pooled',summary_level='pooled')[cols+['summary_level']]
    pd.concat([people,whole],ignore_index=True).to_csv(out/'participant_and_pooled_effects.csv',index=False)

def macro_transition():
    e = read_events()
    edges_from_events(e)
    c = pd.read_csv(ROOT / 'outputs/mca/tables/mca_full_54d_row_coordinates.csv')
    membership = pd.read_csv(ROOT / 'outputs/macro_canonical/tables/d4_k4_event_memberships_REFIT.csv')
    events = e.merge(c[KEYS + [f'Dim{i}' for i in range(13)]], on=KEYS, validate='one_to_one').merge(membership[KEYS + ['cluster_k4_d4']], on=KEYS, validate='one_to_one')
    m = load_module('macro_transition_methods', ROOT / 'code/mca/macro_transition.py')
    transitions = m.build_transitions(events, e[KEYS + ['hippo_next_global']])
    C, P = m.matrices(transitions)
    out = ROOT / 'outputs/macro_transition'
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / 'fixed_D4K4_memberships_with_D13_coordinates_REFIT.csv', index=False)
    transitions.to_csv(out / 'authoritative_event_transitions_D4K4_REFIT.csv', index=False)
    for name, mat in [('counts', C), ('probabilities', P)]:
        pd.DataFrame(mat, index=[f'C{i}' for i in range(1,5)], columns=[f'C{i}' for i in range(1,5)]).to_csv(out / f'cluster_transition_{name}_D4K4_REFIT.csv')
    summaries = []
    pairs = []
    perms = []
    for d in [4, 10, 13]:
        D, cent = m.centroid_dist(events, d)
        stats, perm, mask = m.exact_test(D, C, P)
        summaries.append({'mca_dimensions': d, 'geometry_role': m.ROLES[d], **stats})
        perm.insert(0, 'mca_dimensions', d)
        perm.insert(1, 'geometry_role', m.ROLES[d])
        perms.append(perm)
        cent.to_csv(out / f'cluster_centroids_D{d}_fixed_D4K4_REFIT.csv', index=False)
        for i, j in zip(*np.where(mask)):
            pairs.append({'mca_dimensions': d, 'source_cluster': i + 1, 'target_cluster': j + 1, 'transition_count': int(C[i, j]), 'transition_probability': float(P[i, j]), 'centroid_distance': float(D[i, j])})
    pd.DataFrame(summaries).to_csv(out / 'macro_cluster_transition_geometry_summary_REFIT.csv', index=False)
    pairs=pd.DataFrame(pairs)
    wide=pairs[pairs.mca_dimensions==4].drop(columns=['mca_dimensions']).rename(columns={'centroid_distance':'centroid_distance_D4'})
    for d in [10,13]:
        wide=wide.merge(pairs[pairs.mca_dimensions==d][['source_cluster','target_cluster','centroid_distance']].rename(columns={'centroid_distance':f'centroid_distance_D{d}'}),on=['source_cluster','target_cluster'],validate='one_to_one')
    wide.to_csv(out / 'observed_nonzero_directed_pairs_REFIT.csv', index=False)
    pd.concat(perms).to_csv(out / 'exact_24_label_permutations_REFIT.csv', index=False)
if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('stage', choices=['validate', 'orient', 'assignment_inputs', 'macro_transition','distance_summary'])
    v = a.parse_args()
    if v.stage == 'validate':
        edges_from_events(read_events())
        print('256 events; 251 reciprocal sequence links; 235 within-segment boundaries.')
    else:
        globals()[v.stage]()
