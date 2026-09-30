"""Deterministic analysis of the frozen 256-event dataset."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, hashlib, warnings
from pathlib import Path
from concurrent.futures import as_completed
from runtime import worker_pool, atomic_json, run_root
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits
from capacity_engine import Engine, load_inputs, canonical
from mca_refit import normalize, fit_mca
ROOT = run_root()
OUT = ROOT / 'results'
INP = ROOT / 'inputs'
EVENTS, X, SRC, DST, Y, W, ATOM = load_inputs(INP)
KEYS = ['participant_id_norm', 'hippo_id_norm']
AXES = {4: [0, 1, 2, 3], 2: [0, 3]}
SAMPLES = json.loads((INP / 'PERTURBATIONS.json').read_text())
PILOT = [s for s in SAMPLES if s['repeat'] < 10]
VALIDATION = [s for s in SAMPLES if s['repeat'] >= 10]
ACTUAL = {'scheme': 'actual_added_events', 'repeat': 0, 'retained_indices': np.flatnonzero(~EVENTS.hippo_id_norm.isin(['p02-s3-h012', 'p03-s5-h007'])).tolist()}
PROTOCOL = {'status': 'Exploratory selection of operational basin solutions.', 'events': 256, 'original_edges': 251, 'spaces': {'primary': 4, 'comparison': [0, 3]}, 'geometry': 'Frozen raw MCA scaling, no whitening; refit the same eleven active categorical variables under omissions.', 'K': list(range(5, 21)), 'event_lower': [8, 9, 10], 'event_upper': list(range(25, 51)), 'count_policy': 'Event bounds are design heuristics. Post-fit outgoing eligibility is >=8 in every state for all lower-bound settings. Counts at 8/9 and at >=10 are reported separately; neither proves adequate precision.', 'profiles': 'Identical normalized eleven-feature profiles remain indivisible and event-weighted; no clipping or source changes.', 'fitting': 'Same binary-MILP fixed-centroid assignments and weighted-centroid Lloyd iterations; fixed-center gap1e-9,30s cap,200 iterations. Global clustering optimum is not claimed.', 'seed_rule': '42+1000*K+start_index, independent weighted KMeans initializations; common seeds across dimensionalities and bounds.', 'screen': '2496 requested cells,10 independent starts per feasible cell, plus previous-cap geometry/eligible warm starts. Retain lowest SSE outgoing-eligible solution; otherwise geometry best flagged unsupported. Arithmetic-infeasible and equal-size-only cells recorded.', 'screen_selection': 'For each (dimension,K), choose one representative from eligible cells using nondominated solutions in mean fit-space silhouette, minimum state fit-space silhouette, minimum outgoing support and negative nonnearest fraction. Break frontier ties by equal mean percentile rank on those four metrics within that (dimension,K), then SSE,lower,upper. Add best representative for each (dimension,lower) if its partition is not already present.', 'screen_deduplication': 'Exact event-partition duplicates removed; all source configurations retained in screen table.', 'full_refinement': '100 independent starts per pilot configuration, retaining its screen partition as an extra non-independent candidate. All perturbations compare with this refined full-sample assignment.', 'pilot': 'First10 specified two-event and first10 specified13-event samples,refitted MCA only,20 independent starts per fit. Actual pair excluded from selection.', 'pilot_ranking': 'Equal mean percentile rank within dimension on mean ARI,p10 ARI,mean minimum matched-state Jaccard and fraction outgoing-eligible; nondominated candidates prioritized. Exact full-sample duplicates removed; skip partitions with ARI>=0.95 against already selected candidates in that dimension.', 'validation_candidates': 'Up to5 D4 and3 D2 pilot representatives,chosen before held-out omission results. Require eligible full sample and >=90% completed pilot fits. No absolute stability cutoff is claimed.', 'validation': 'Remaining40 specified two-event plus90 specified13-event samples; fixed and refitted geometry;30 independent starts each. Actual two-event omission separately uses100 starts in both modes. Full reference labels never initialize omission fits.', 'optimization_check': 'Record selected first-half-budget vs full-budget ARI for every perturbation,with failed starts and geometric best separately. Numerical uncertainty remains visible.', 'final_shortlist': 'From validation candidates with >=95% refitted completion and >=80% outgoing-eligible fraction separately in both omission schemes, retain up to3 D4 and2 D2 assignments,using the pilot ranking rule on held-out13-event refits and ARI<0.95 diversity. Do not fill failed slots with weaker candidates. Actual omission is a separately reported stress test and does not determine ranking.', 'thresholds_are': 'Declared exploratory screening rules,not statistical validation standards. No change after results to fill the shortlist.', 'sequence': 'All256 events/all251 original edges in full sample. Omission uses only original edges with both endpoints retained; no bridging.', 'downstream': 'Freeze final labels before loading/running new successor or routing outcomes. Basin-specific correspondence cannot be assumed across K. Any subsequent cross-K smoothing uses a declared fixed total prior mass.', 'limits': 'Finite start budgets; constrained groups are operational states; pilot/validation omissions reuse the same five participants and are not independent replication. The screen cannot prove the globally best configuration.'}

def write_json(path, obj):
    atomic_json(path, obj)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def config_id(d, k, l, u):
    return f'D{d}_K{k}_L{l}_U{u}'

def ckey(labels):
    return hashlib.sha256(canonical(np.asarray(labels)).astype('<i2').tobytes()).hexdigest()

def check_protocol():
    assert json.loads((ROOT / 'Protocol.json').read_text()) == PROTOCOL

def describe(coords, lab, idx, dim):
    lab = canonical(lab)
    k = len(np.unique(lab))
    sub = EVENTS.iloc[idx].reset_index(drop=True)
    present = np.zeros(256, bool)
    present[idx] = True
    valid = present[SRC] & present[DST]
    lookup = np.full(256, -1, int)
    lookup[idx] = np.arange(len(idx))
    src = lookup[SRC[valid]]
    dst = lookup[DST[valid]]
    count = np.bincount(lab, minlength=k)
    out = np.bincount(lab[src], minlength=k)
    z = coords[:, AXES[dim]]
    cen = np.array([z[lab == j].mean(0) for j in range(k)])
    ds = cdist(z, cen, 'sqeuclidean')
    assigned = ds[np.arange(len(z)), lab]
    sf = silhouette_samples(z, lab)
    s4 = silhouette_samples(coords, lab)
    cen4 = np.array([coords[lab == j].mean(0) for j in range(k)])
    ss4 = ((coords - cen4[lab]) ** 2).sum(1)
    total4 = ((coords - coords.mean(0)) ** 2).sum()
    nonnearest = assigned > ds.min(1) + 1e-10
    states = []
    for j in range(k):
        m = lab == j
        parts = sub.loc[m, 'participant_id_norm'].value_counts()
        states.append({'state': j + 1, 'n_events': int(count[j]), 'outgoing': int(out[j]), 'participants': len(parts), 'max_participant_share': float(parts.max() / m.sum()), 'fit_silhouette': float(sf[m].mean()), 'D4_silhouette': float(s4[m].mean()), 'D4_SS_per_event': float(ss4[m].mean()), 'nonnearest_events': int(nonnearest[m].sum())})
    row = {'n_events': len(idx), 'n_edges': int(valid.sum()), 'n_min': int(count.min()), 'n_max': int(count.max()), 'min_outgoing': int(out.min()), 'states_outgoing_lt10': int((out < 10).sum()), 'eligible': bool(out.min() >= 8), 'event_floor_realized': int(count.min()), 'fit_silhouette': float(sf.mean()), 'min_state_fit_silhouette': min((v['fit_silhouette'] for v in states)), 'D4_silhouette': float(s4.mean()), 'D4_explained_fraction': float(1 - ss4.sum() / total4), 'fit_SSE': float(assigned.sum()), 'nonnearest_fraction': float(nonnearest.mean()), 'min_participants': min((v['participants'] for v in states)), 'max_participant_share': max((v['max_participant_share'] for v in states)), 'partition_hash': ckey(lab)}
    return (row, states)

def atomize(coords, idx, dim):
    sub = EVENTS.iloc[idx]
    _, atom, w = np.unique(normalize(sub).to_numpy(dtype=str), axis=0, return_inverse=True, return_counts=True)
    z = coords[:, AXES[dim]]
    y = np.array([z[atom == j].mean(0) for j in range(len(w))])
    assert np.max(abs(z - y[atom])) < 1e-10
    present = np.zeros(256, bool)
    present[idx] = True
    valid = present[SRC] & present[DST]
    ow = np.bincount(SRC[valid], minlength=256)[idx]
    return (y, w, atom, ow)

def fit_many(coords, idx, d, k, l, u, starts, retained=None):
    y, w, atom, ow = atomize(coords, idx, d)
    engine = Engine(y, w, k, l)
    best = None
    eligible = None
    half = None
    trials = []
    failures = []

    def add(fit, source):
        nonlocal best, eligible
        lab = canonical(fit['labels'][atom])
        oc = np.bincount(lab, weights=ow, minlength=k)
        r = {'labels': lab, 'sse': fit['sse'], 'min_outgoing': int(oc.min()), 'source': source}
        trials.append({'source': source, 'sse': fit['sse'], 'min_outgoing': r['min_outgoing'], 'solver_calls': fit['solver_calls'], 'iterations': fit['iterations'], 'mip_gap': fit['max_fixed_center_mip_gap'], 'partition_hash': ckey(lab)})
        if best is None or r['sse'] < best['sse'] - 1e-10:
            best = r
        if r['min_outgoing'] >= 8 and (eligible is None or r['sse'] < eligible['sse'] - 1e-10):
            eligible = r
    for s in range(starts):
        km = KMeans(n_clusters=k, n_init=1, max_iter=300, tol=1e-08, random_state=42 + 1000 * k + s).fit(y, sample_weight=w)
        try:
            add(engine.fit(km.cluster_centers_, u), f'start_{s}')
        except RuntimeError as e:
            failures.append({'start': s, 'reason': str(e)})
        if s + 1 == starts // 2 and best is not None:
            half = (eligible or best)['labels'].copy()
    if retained is not None:
        al = np.array([retained[np.flatnonzero(atom == j)[0]] for j in range(len(w))])
        assert np.array_equal(al[atom], retained)
        try:
            add(engine.fit(engine.centers(al), u), 'retained_screen')
        except RuntimeError as e:
            failures.append({'start': 'retained_screen', 'reason': str(e)})
    if best is None:
        return (None, {'failures': failures, 'trials': trials})
    chosen = eligible or best
    return (chosen, {'failures': failures, 'trials': trials, 'best_geometry_SSE': best['sse'], 'best_geometry_labels': best['labels'].tolist(), 'eligible_found': eligible is not None, 'half_to_full_ARI': float(adjusted_rand_score(half, chosen['labels'])) if half is not None else None, 'solver_calls': engine.calls, 'unique_partitions': len({t['partition_hash'] for t in trials})})

def screen_job(d, k):
    check_protocol()
    folder = OUT / 'screen' / f'D{d}_K{k}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        marker = json.loads((folder / 'complete.json').read_text())
        assert marker['protocol_hash'] == sha(ROOT / 'Protocol.json')
        assert all((sha(folder / name) == digest for name, digest in marker['output_hashes'].items()))
        return f'D{d}_K{k}'
    rows = []
    states = []
    trials = []
    assign = {}
    idx = np.arange(256)
    start = time.monotonic()
    with threadpool_limits(limits=1):
        y, w, atom, ow = atomize(X, idx, d)
        centers = [KMeans(n_clusters=k, n_init=1, max_iter=300, tol=1e-08, random_state=42 + 1000 * k + s).fit(y, sample_weight=w).cluster_centers_ for s in range(10)]
        for l in [8, 9, 10]:
            engine = Engine(y, w, k, l)
            prevbest = prevvalid = None
            for u in range(25, 51):
                cid = config_id(d, k, l, u)
                base = {'id': cid, 'dim': d, 'K': k, 'lower': l, 'upper': u}
                if k * u <= 256:
                    rows.append({**base, 'available': False, 'reason': 'Arithmetic infeasible' if k * u < 256 else 'Equal-size-only design excluded'})
                    continue
                pool = []
                independent = []
                starts = [(f'start_{s}', c) for s, c in enumerate(centers)]
                for label, r in [('previous_geometry', prevbest), ('previous_eligible', prevvalid)]:
                    if r is not None:
                        starts.append((label, engine.centers(r['labels'])))
                for source, c in starts:
                    try:
                        r = engine.fit(c, u)
                        lab = canonical(r['labels'][atom])
                        out = np.bincount(lab, weights=ow, minlength=k)
                        r['out'] = int(out.min())
                        r['source'] = source
                        pool.append(r)
                        if source.startswith('start_'):
                            independent.append(lab)
                        trials.append({**base, 'source': source, 'available': True, 'sse': r['sse'], 'min_outgoing': r['out'], 'mip_gap': r['max_fixed_center_mip_gap'], 'solver_calls': r['solver_calls']})
                    except RuntimeError as e:
                        trials.append({**base, 'source': source, 'available': False, 'reason': str(e)})
                for r in [prevbest, prevvalid]:
                    if r is not None:
                        pool.append(r)
                if not pool:
                    rows.append({**base, 'available': False, 'reason': 'All starts failed'})
                    continue
                best = min(pool, key=lambda r: r['sse'])
                valid = [r for r in pool if r['out'] >= 8]
                chosen = min(valid, key=lambda r: r['sse']) if valid else best
                lab = canonical(chosen['labels'][atom])
                a, ss = describe(X, lab, idx, d)
                a.update({'geometry_best_SSE': best['sse'], 'eligibility_extra_SSE': chosen['sse'] - best['sse'], 'independent_starts_ok': len(independent), 'unique_independent_partitions': len({ckey(v) for v in independent}), 'mean_start_ARI': float(np.mean([adjusted_rand_score(lab, v) for v in independent])) if independent else None})
                rows.append({**base, 'available': True, 'reason': '', **a})
                states.extend([{**base, **v} for v in ss])
                assign[cid] = lab + 1
                prevbest = best
                prevvalid = chosen if valid else None
            pd.DataFrame(rows).to_csv(folder / 'Screen.csv', index=False)
    pd.DataFrame(states).to_csv(folder / 'States.csv', index=False)
    pd.DataFrame(trials).to_csv(folder / 'Starts.csv', index=False)
    pd.DataFrame(assign, index=np.arange(256)).to_csv(folder / 'Assignments.csv', index=False)
    write_json(folder / 'complete.json', {'seconds': time.monotonic() - start, 'protocol_hash': sha(ROOT / 'Protocol.json'), 'output_hashes': {name: sha(folder / name) for name in ['Screen.csv', 'States.csv', 'Starts.csv', 'Assignments.csv']}})
    return f'D{d}_K{k}'

def run_screen():
    check_protocol()
    start = time.monotonic()
    with worker_pool() as pool:
        futures = [pool.submit(screen_job, d, k) for d in [4, 2] for k in range(5, 21)]
        for i, f in enumerate(as_completed(futures), 1):
            print(f'Screen {i}/32 {f.result()}, elapsed {time.monotonic() - start:.1f}s', flush=True)
    folders = [OUT / 'screen' / f'D{d}_K{k}' for d in [4, 2] for k in range(5, 21)]
    for fn in ['Screen', 'States', 'Starts']:
        frames = []
        for folder in folders:
            try:
                frames.append(pd.read_csv(folder / (fn + '.csv')))
            except pd.errors.EmptyDataError:
                pass
        pd.concat(frames, ignore_index=True).to_csv(OUT / (fn + '.csv'), index=False)
    frames = [EVENTS[KEYS]]
    for folder in folders:
        try:
            frames.append(pd.read_csv(folder / 'Assignments.csv'))
        except pd.errors.EmptyDataError:
            pass
    pd.concat(frames, axis=1).to_csv(OUT / 'Screen_Assignments.csv', index=False)
    select_pilot()

def pareto_rank(df, cols):
    df = df.copy()
    v = df[cols].to_numpy(float)
    front = np.ones(len(df), bool)
    for i in range(len(df)):
        front[i] = not np.any(np.all(v >= v[i] - 1e-12, axis=1) & np.any(v > v[i] + 1e-12, axis=1))
    df['frontier'] = front
    df['rank_score'] = df[cols].rank(pct=True).mean(axis=1)
    ties = ['fit_SSE', 'lower', 'upper', 'id'] if 'fit_SSE' in df.columns else ['id']
    return df.sort_values(['frontier', 'rank_score'] + ties, ascending=[False, False] + [True] * len(ties))

def select_pilot():
    s = pd.read_csv(OUT / 'Screen.csv')
    s = s[s.available & s.eligible.fillna(False)].copy()
    s['negative_nonnearest'] = -s.nonnearest_fraction
    cols = ['fit_silhouette', 'min_state_fit_silhouette', 'min_outgoing', 'negative_nonnearest']
    chosen = []
    for key, g in s.groupby(['dim', 'K']):
        ranked = pareto_rank(g, cols)
        r = ranked.iloc[0]
        chosen.append(r.to_dict())
    for key, g in s.groupby(['dim', 'lower']):
        r = pareto_rank(g, cols).iloc[0]
        if r.partition_hash not in [v['partition_hash'] for v in chosen]:
            chosen.append(r.to_dict())
    c = pd.DataFrame(chosen).drop_duplicates('partition_hash')
    c.to_csv(OUT / 'Pilot_Configurations.csv', index=False)
    print(f'Frozen {len(c)} pilot configurations; downstream outcomes unopened', flush=True)

def refine_job(config):
    cid = config['id']
    folder = OUT / 'full' / cid
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'Fit.json').exists():
        return cid
    a = pd.read_csv(OUT / 'Screen_Assignments.csv', usecols=[cid])[cid].to_numpy() - 1
    with threadpool_limits(limits=1):
        fit, detail = fit_many(X, np.arange(256), int(config['dim']), int(config['K']), int(config['lower']), int(config['upper']), 100, a)
        if fit is None:
            write_json(folder / 'Fit.json', {'id': cid, 'available': False, 'detail': detail})
            return cid
        row, states = describe(X, fit['labels'], np.arange(256), int(config['dim']))
    write_json(folder / 'Fit.json', {'id': cid, 'available': True, 'summary': row, 'labels': fit['labels'].tolist(), 'detail': detail, 'states': states})
    return cid

def agreement(a, b, k):
    tab = np.zeros((k, k), int)
    np.add.at(tab, (a, b), 1)
    rr, cc = linear_sum_assignment(-tab)
    mapping = np.empty(k, int)
    mapping[cc] = rr
    c = mapping[b]
    jac = np.array([np.sum((a == j) & (c == j)) / np.sum((a == j) | (c == j)) if np.any((a == j) | (c == j)) else 1.0 for j in range(k)])
    return ({'ARI': float(adjusted_rand_score(a, b)), 'changed': int(np.sum(a != c)), 'min_state_Jaccard': float(jac.min()), 'mean_state_Jaccard': float(jac.mean())}, jac)

def perturb_job(stage, config, sample, mode):
    cid = config['id']
    folder = OUT / stage / 'checkpoints'
    folder.mkdir(parents=True, exist_ok=True)
    tag = f"{cid}_{sample['scheme']}_{sample['repeat']:03d}_{mode}"
    target = folder / (tag + '.json')
    if target.exists():
        return tag
    idx = np.array(sample['retained_indices'])
    d = int(config['dim'])
    k = int(config['K'])
    l = int(config['lower'])
    u = int(config['upper'])
    full = json.loads((OUT / 'full' / cid / 'Fit.json').read_text())
    ref = np.array(full['labels'])[idx]
    starts = 20 if stage == 'pilot' else 100 if sample['scheme'] == 'actual_added_events' else 30
    start = time.monotonic()
    with threadpool_limits(limits=1):
        coords = X[idx] if mode == 'fixed' else fit_mca(EVENTS.iloc[idx].reset_index(drop=True), X[idx])[0]
        fit, detail = fit_many(coords, idx, d, k, l, u, starts)
        summary = {'id': cid, 'dim': d, 'K': k, 'lower': l, 'upper': u, 'scheme': sample['scheme'], 'repeat': sample['repeat'], 'mode': mode, 'available': fit is not None, 'starts_requested': starts, 'seconds': time.monotonic() - start}
        if fit is None:
            write_json(target, {'summary': summary, 'detail': detail})
            return tag
        a, states = describe(coords, fit['labels'], idx, d)
        metrics, jac = agreement(ref, fit['labels'], k)
        originalcounts = np.bincount(ref, minlength=k)
        present = np.zeros(256, bool)
        present[idx] = True
        eo = present[SRC] & present[DST]
        originalfull = np.array(full['labels'])
        origout = np.bincount(originalfull[SRC[eo]], minlength=k)
        summary.update({**a, **metrics, 'half_to_full_ARI': detail['half_to_full_ARI'], 'failed_starts': len(detail['failures']), 'reference_floor_crossed': bool(originalcounts.min() < l), 'reference_outgoing8_crossed': bool(origout.min() < 8), 'pairwise_distance_r': float(np.corrcoef(pdist(X[idx][:, AXES[d]]), pdist(coords[:, AXES[d]]))[0, 1])})
    write_json(target, {'summary': summary, 'retained_indices': idx.tolist(), 'labels': fit['labels'].tolist(), 'reference_state_Jaccard': jac.tolist(), 'states': states, 'detail': detail})
    return tag

def collate(stage):
    rows = []
    states = []
    for p in sorted((OUT / stage / 'checkpoints').glob('*.json')):
        r = json.loads(p.read_text())
        rows.append(r['summary'])
        if r['summary']['available']:
            for i, j in enumerate(r['reference_state_Jaccard']):
                states.append({**{k: r['summary'][k] for k in ['id', 'scheme', 'repeat', 'mode']}, 'reference_state': i + 1, 'Jaccard': j})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / stage / 'Trials.csv', index=False)
    pd.DataFrame(states).to_csv(OUT / stage / 'State_Stability.csv', index=False)
    t['support_success'] = t.available & t.eligible.fillna(False).astype(bool)
    summary = t.groupby(['id', 'dim', 'K', 'lower', 'upper', 'scheme', 'mode']).agg(trials=('available', 'size'), availability=('available', 'mean'), support_fraction=('support_success', 'mean'), ARI_mean=('ARI', 'mean'), ARI_p10=('ARI', lambda v: v.quantile(0.1)), ARI_median=('ARI', 'median'), changed_mean=('changed', 'mean'), min_state_Jaccard_mean=('min_state_Jaccard', 'mean'), half_to_full_ARI_mean=('half_to_full_ARI', 'mean'), floor_cross_fraction=('reference_floor_crossed', 'mean'), outgoing_cross_fraction=('reference_outgoing8_crossed', 'mean'), min_outgoing_min=('min_outgoing', 'min')).reset_index()
    summary.to_csv(OUT / stage / 'Summary.csv', index=False)
    return (t, summary)

def run_stage(stage):
    check_protocol()
    file = 'Pilot_Configurations.csv' if stage == 'pilot' else 'Validation_Configurations.csv'
    configs = pd.read_csv(OUT / file).to_dict('records')
    start = time.monotonic()
    if stage == 'pilot':
        with worker_pool() as pool:
            ff = [pool.submit(refine_job, c) for c in configs]
            for i, f in enumerate(as_completed(ff), 1):
                print(f'Refined {i}/{len(ff)}: {f.result()}', flush=True)
        configs = [c for c in configs if json.loads((OUT / 'full' / c['id'] / 'Fit.json').read_text())['available']]
    samples = PILOT if stage == 'pilot' else VALIDATION + [ACTUAL]
    modes = ['refitted'] if stage == 'pilot' else ['fixed', 'refitted']
    jobs = [(stage, c, s, mode) for s in samples for c in configs for mode in modes]
    with worker_pool() as pool:
        ff = [pool.submit(perturb_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(ff), 1):
            tag = f.result()
            if i % 20 == 0 or i == len(ff):
                print(f'{stage} {i}/{len(ff)}; {time.monotonic() - start:.1f}s; {tag}', flush=True)
    t, summary = collate(stage)
    if stage == 'pilot':
        select_validation(t)

def select_by_stability(t, configs, quotas, stage):
    lookup = {c['id']: c for c in configs}
    rows = []
    for cid, g in t[t['mode'].eq('refitted') & t.scheme.eq('omit5pct')].groupby('id'):
        c = lookup[cid]
        full = json.loads((OUT / 'full' / cid / 'Fit.json').read_text())
        if not full['available'] or not full['summary']['eligible']:
            continue
        allg = t[t.id.eq(cid) & t['mode'].eq('refitted') & t.scheme.ne('actual_added_events')]
        completion = allg.groupby('scheme').available.mean()
        support = allg.assign(ok=allg.available & allg.eligible.fillna(False)).groupby('scheme').ok.mean()
        if completion.min() < (0.9 if stage == 'pilot' else 0.95):
            continue
        if stage != 'pilot' and support.min() < 0.8:
            continue
        rows.append({**c, **full['summary'], 'ARI_mean': g.ARI.mean(), 'ARI_p10': g.ARI.quantile(0.1), 'min_state_Jaccard_mean': g.min_state_Jaccard.mean(), 'support_fraction': float((g.available & g.eligible.fillna(False)).mean()), 'completion_min': completion.min(), 'support_min': support.min()})
    selected = []
    rankedrows = []
    for d, quota in quotas.items():
        subset = pd.DataFrame([v for v in rows if v['dim'] == d])
        if not len(subset):
            continue
        ranked = pareto_rank(subset, ['ARI_mean', 'ARI_p10', 'min_state_Jaccard_mean', 'support_fraction'])
        rankedrows.append(ranked)
        labs = []
        for _, r in ranked.iterrows():
            lab = np.array(json.loads((OUT / 'full' / r.id / 'Fit.json').read_text())['labels'])
            if any((adjusted_rand_score(lab, a) >= 0.95 for a in labs)):
                continue
            selected.append(r.to_dict())
            labs.append(lab)
            if len(labs) == quota:
                break
    if rankedrows:
        pd.concat(rankedrows).to_csv(OUT / (stage + '_Ranking.csv'), index=False)
    return pd.DataFrame(selected)

def select_validation(t):
    configs = pd.read_csv(OUT / 'Pilot_Configurations.csv').to_dict('records')
    chosen = select_by_stability(t, configs, {4: 5, 2: 3}, 'pilot')
    if chosen.empty:
        raise RuntimeError('No candidates passed the pilot rules; do not relax thresholds to fill slots.')
    chosen.to_csv(OUT / 'Validation_Configurations.csv', index=False)
    print('Validation configurations frozen:', chosen.id.tolist(), flush=True)

def select_final():
    t = pd.read_csv(OUT / 'validation/Trials.csv')
    configs = pd.read_csv(OUT / 'Validation_Configurations.csv').to_dict('records')
    chosen = select_by_stability(t, configs, {4: 3, 2: 2}, 'validation')
    if chosen.empty:
        chosen = pd.DataFrame(columns=['id', 'dim', 'K', 'lower', 'upper', 'completion_min', 'support_min'])
    chosen.to_csv(OUT / 'Final_Shortlist.csv', index=False)
    assign = EVENTS[KEYS].copy()
    details = []
    for r in chosen.itertuples():
        a = json.loads((OUT / 'full' / r.id / 'Fit.json').read_text())
        assign[r.id] = np.array(a['labels']) + 1
        details.append({'id': r.id, 'dim': int(r.dim), 'K': int(r.K), 'lower': int(r.lower), 'upper': int(r.upper), 'label_hash': ckey(a['labels']), 'full_summary': a['summary']})
    assign.to_csv(OUT / 'Final_Event_Assignments.csv', index=False)
    write_json(OUT / 'Frozen_Shortlist.json', {'protocol_hash': sha(ROOT / 'Protocol.json'), 'assignments_hash': sha(OUT / 'Final_Event_Assignments.csv'), 'selected_before_new_downstream_outcomes': True, 'candidates': details, 'interpretation': 'Relative stability shortlist,not proof of uniquely valid basins.'})
    print('Final shortlist frozen:', chosen.id.tolist(), flush=True)

def init():
    if (ROOT / 'Protocol.json').exists():
        check_protocol()
    else:
        write_json(ROOT / 'Protocol.json', PROTOCOL)
    assert len(PILOT) == 20 and len(VALIDATION) == 130 and (len(ACTUAL['retained_indices']) == 254)
    assert np.allclose(fit_mca(EVENTS, X)[0], X, atol=1e-10)
    write_json(ROOT / 'Input_Checks.json', {'full_MCA_reproduced': True, 'unique_profiles': len(W), 'largest_profile': int(W.max()), 'input_hashes': {p.name: sha(p) for p in INP.iterdir() if p.is_file()}, 'pilot_samples': [(s['scheme'], s['repeat']) for s in PILOT], 'validation_samples': [(s['scheme'], s['repeat']) for s in VALIDATION]})
    print('Protocol frozen; inputs and full MCA verified', flush=True)
if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'init':
        init()
    elif command == 'screen':
        run_screen()
    elif command == 'pilot':
        run_stage('pilot')
    elif command == 'validate':
        run_stage('validation')
    elif command == 'select':
        select_final()
    else:
        raise ValueError(command)
