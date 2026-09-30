"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_methods import FEATURES, GAMMAS, require, csv, square, row_normalize, count_matrix, bh_fdr, run_component
PACKAGE = Path(__file__).resolve().parent
KEYS = ['participant_id_norm', 'hippo_id_norm']
COMPONENTS = ['feature_primary3', 'feature_all11', 'conditioned_context', 'conditioned_mechanism', 'conditioned_meta_theme', 'geometry_070', 'geometry_080', 'geometry_090', 'geometry_095', 'recurrence']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)

def utc():
    return datetime.now(timezone.utc).isoformat()

def file_manifest(folder, exclude=()):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in exclude and ('__pycache__' not in p.parts)}

def environment():
    return {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}

def partition_hash(values):
    mapping = {}
    canonical = [mapping.setdefault(int(v), len(mapping)) for v in values]
    return hashlib.sha256(np.asarray(canonical, dtype='<i2').tobytes()).hexdigest()

def normalize_keys(frame, cols=KEYS):
    frame = frame.copy()
    for col in cols:
        frame[col] = frame[col].astype(str).str.strip().str.lower()
        require(frame[col].ne('').all(), f'Empty key in {col}.')
    return frame

def reconstruct_sequences(events, edges):
    sequences, manifest = ({}, [])
    for participant, group in events.groupby(KEYS[0], sort=True):
        event_ids = set(group[KEYS[1]])
        sub = edges.loc[edges.participant_id_norm.eq(participant)]
        require(len(sub) == len(group) - 1, f'Broken global sequence for {participant}.')
        require(not sub.source_event.duplicated().any() and (not sub.target_event.duplicated().any()), 'Branched sequence.')
        require(set(sub.source_event).issubset(event_ids) and set(sub.target_event).issubset(event_ids), 'Unknown/cross-participant edge.')
        heads = event_ids - set(sub.target_event)
        require(len(heads) == 1, f'Expected one sequence head for {participant}.')
        nxt = dict(zip(sub.source_event, sub.target_event))
        state = dict(zip(group[KEYS[1]], group.state.astype(int) - 1))
        current, ordered, seen = (next(iter(heads)), [], set())
        while current is not None:
            require(current not in seen, 'Cycle in frozen sequence.')
            seen.add(current)
            ordered.append(current)
            current = nxt.get(current)
        require(seen == event_ids, f'Sequence traversal omitted events for {participant}.')
        sequences[participant] = np.array([state[e] for e in ordered], dtype=np.int16)
        manifest.extend(({'participant_id_norm': participant, 'sequence_position': i + 1, 'hippo_id_norm': e, 'state': state[e] + 1} for i, e in enumerate(ordered)))
    require(len(sequences) == 5 and sum(map(len, sequences.values())) == 256, 'Unexpected participant/event count.')
    return (sequences, pd.DataFrame(manifest))

def assemble_data(events, membership, edges, coordinates, assignment_id, k):
    events, membership, coordinates = [normalize_keys(x) for x in [events, membership, coordinates]]
    edges = normalize_keys(edges, [KEYS[0], 'source_event', 'target_event'])
    for name, frame in [('features', events), ('memberships', membership), ('coordinates', coordinates)]:
        require(len(frame) == 256 and (not frame.duplicated(KEYS).any()), f'{name}: expected 256 unique events.')
        require(set(map(tuple, frame[KEYS].to_numpy())) == set(map(tuple, events[KEYS].to_numpy())), f'{name}: mismatched keys.')
    require(len(edges) == 251 and (not edges.duplicated().any()), 'Expected 251 unique frozen edges.')
    require(set(edges[KEYS[0]]) == set(events[KEYS[0]]), 'Participant mismatch in edges.')
    numeric = pd.to_numeric(membership.state, errors='raise').to_numpy(float)
    require(np.isfinite(numeric).all() and np.equal(numeric, np.floor(numeric)).all(), 'Noninteger membership.')
    require(set(numeric) == set(range(1, k + 1)), 'Noncontiguous/missing basin labels.')
    membership['state'] = numeric.astype(int)
    dims = [f'Dim{i}' for i in range(13)]
    merged = events[KEYS + FEATURES].merge(membership[KEYS + ['state']], on=KEYS, validate='one_to_one', sort=False)
    merged = merged.merge(coordinates[KEYS + dims], on=KEYS, validate='one_to_one', sort=False)
    require(np.isfinite(merged[dims].to_numpy(float)).all(), 'Nonfinite MCA coordinate.')
    feature_lookup = merged.rename(columns={KEYS[1]: 'source_event', 'state': 'source_state'})
    transitions = edges.merge(feature_lookup[[KEYS[0], 'source_event', 'source_state'] + FEATURES], on=[KEYS[0], 'source_event'], validate='one_to_one', sort=False)
    targets = membership.rename(columns={KEYS[1]: 'target_event', 'state': 'target_state'})
    transitions = transitions.merge(targets, on=[KEYS[0], 'target_event'], validate='one_to_one', sort=False)
    require(len(transitions) == 251, 'An edge failed membership lookup.')
    transitions[['source_state', 'target_state']] -= 1
    counts = count_matrix(transitions.source_state.to_numpy(int), transitions.target_state.to_numpy(int), k)
    sequences, sequence_manifest = reconstruct_sequences(merged, edges)
    by_sequence = sum((count_matrix(v[:-1], v[1:], k) for v in sequences.values()), np.zeros((k, k)))
    require(np.array_equal(counts, by_sequence), 'Sequence/count disagreement.')
    return {'id': assignment_id, 'K': k, 'labels': [f'B{i}' for i in range(1, k + 1)], 'events': merged, 'transitions': transitions, 'sequences': sequences, 'sequence_manifest': sequence_manifest, 'C': counts, 'P': row_normalize(counts)}

def load_inputs(args):
    run = args.assignment_run
    names = json.loads((PACKAGE / 'required_inputs.json').read_text())
    input_paths = {name: run / name for name in names}
    input_paths['full_mca_coordinates'] = args.coordinates
    for path in input_paths.values():
        require(path.is_file(), f'Missing completed analysis input: {path}')
    verification = json.loads((run / 'results/Independent_Verification.json').read_text())
    require(verification['passed'], 'Independent basin checks have not passed.')
    shortlist = json.loads((run / 'results/Frozen_Shortlist.json').read_text())
    require(shortlist['selected_before_new_downstream_outcomes'], 'Shortlist was not frozen before downstream outcomes.')
    require(sha(run / 'results/Final_Event_Assignments.csv') == shortlist['assignments_hash'], 'Frozen final assignments changed.')
    events = pd.read_csv(run / 'inputs/new_events.csv', keep_default_na=False)
    edges = pd.read_csv(run / 'inputs/event_edges.csv', keep_default_na=False)
    coordinates = pd.read_csv(args.coordinates, keep_default_na=False)
    dimcheck = events[KEYS + [f'Dim{i}' for i in range(4)]].merge(coordinates, on=KEYS, validate='one_to_one', suffixes=('_frozen', ''))
    require(len(dimcheck) == 256 and all((np.allclose(dimcheck[f'Dim{i}_frozen'], dimcheck[f'Dim{i}'], atol=1e-12, rtol=0) for i in range(4))), 'D4 geometry changed.')
    data = []
    for candidate in shortlist['candidates']:
        aid, k = (candidate['id'], int(candidate['K']))
        membership = pd.read_csv(run / f'selected/{aid}/Event_Memberships.csv', keep_default_na=False)
        spec = json.loads((run / f'selected/{aid}/Assignment.json').read_text())
        aligned = events[KEYS].merge(membership, on=KEYS, validate='one_to_one', sort=False)
        require(partition_hash(aligned.state.to_numpy(int)) == candidate['label_hash'] == spec['partition_hash'], 'Frozen partition changed.')
        item = assemble_data(events, membership, edges, coordinates, aid, k)
        item['assignment'] = spec
        data.append(item)
    require(len(data) == 5, 'Expected all five assignments.')
    fingerprints = {name: {'path': str(path.resolve()), 'sha256': sha(path)} for name, path in input_paths.items()}
    return (data, fingerprints)

def signature(args, fingerprints, mode):
    return {'mode': mode, 'n_permutations': 199 if mode == 'smoke' else 10000, 'seed': 42, 'inputs': {name: value['sha256'] for name, value in fingerprints.items()}, 'code': {p.name: sha(p) for p in sorted(PACKAGE.glob('*.py'))}, 'protocol': sha(PACKAGE / 'METHODS.md'), 'expected_inputs': sha(PACKAGE / 'required_inputs.json'), 'environment': environment(), 'components': COMPONENTS}

def verified_stage(folder):
    marker = folder / 'COMPLETE.json'
    require(marker.is_file(), f'Stage is incomplete: {folder}')
    meta = json.loads(marker.read_text())
    require(file_manifest(folder, exclude=('COMPLETE.json',)) == meta['outputs'], f'Stage output changed: {folder}')
    return meta

def write_assignment_inputs(data, out):
    csv(data['events'], out / 'event_memberships_features_coordinates.csv')
    exported = data['transitions'].copy()
    exported[['source_state', 'target_state']] += 1
    csv(exported, out / 'event_transition_records.csv')
    csv(data['sequence_manifest'], out / 'participant_sequence_manifest.csv')
    square(data['C'], data['labels'], out / 'transition_counts.csv')
    square(data['P'], data['labels'], out / 'transition_probabilities_empirical.csv')
    diagnostics = pd.DataFrame({'basin': data['labels'], 'n_events': np.bincount(data['events'].state.to_numpy(int) - 1, minlength=data['K']), 'outgoing': data['C'].sum(1).astype(int), 'incoming': data['C'].sum(0).astype(int), 'self_transitions': np.diag(data['C']).astype(int)})
    csv(diagnostics, out / 'basin_support.csv')
    json_write(out / 'Assignment.json', data['assignment'])
    json_write(out / 'COMPLETE.json', {'outputs': file_manifest(out), 'completed_utc': utc()})

def prepare_assignment(data, output):
    target = output / data['id'] / 'inputs'
    if target.exists():
        verified_stage(target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    work = output / '.work'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{data['id']}__inputs__", dir=work) as temp:
        folder = Path(temp)
        write_assignment_inputs(data, folder)
        shutil.move(str(folder), str(target))

def worker(data, output, component, n_perm):
    start = time.monotonic()
    output = Path(output)
    target = output / data['id'] / component
    print(f"[{data['id']}] {component}: started", flush=True)
    work = output / '.work'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{data['id']}__{component}__", dir=work) as temp:
        folder = Path(temp)
        with threadpool_limits(limits=1):
            run_component(data, folder, component, n_perm)
        elapsed = time.monotonic() - start
        json_write(folder / 'COMPLETE.json', {'assignment_id': data['id'], 'component': component, 'n_permutations': n_perm, 'elapsed_seconds': elapsed, 'completed_utc': utc(), 'outputs': file_manifest(folder)})
        require(not target.exists(), f'Refusing to replace existing stage: {target}')
        shutil.move(str(folder), str(target))
    print(f"[{data['id']}] {component}: complete ({elapsed:.1f}s)", flush=True)
    return elapsed

def build_comparisons(data, output, mode):
    out = output / 'comparisons'
    out.mkdir(exist_ok=True)
    feature_rows, condition_rows, geometry_rows, recurrence_rows, overview = ([], [], [], [], [])
    for item in data:
        aid, k = (item['id'], item['K'])
        base = output / aid
        for component in COMPONENTS:
            verified_stage(base / component)
        for profile in ['primary3', 'all11']:
            frame = pd.read_csv(base / f'feature_{profile}/feature_routing_stats.csv')
            frame.insert(0, 'assignment_id', aid)
            feature_rows.append(frame)
        conditioned = pd.concat([pd.read_csv(base / f'conditioned_{f}/conditioned_routing_results.csv') for f in ['context', 'mechanism', 'meta_theme']], ignore_index=True)
        conditioned.insert(0, 'assignment_id', aid)
        conditioned['bh_q_within_assignment_and_test_family'] = np.nan
        for family in ['full_matrix', 'supported_source']:
            mask = conditioned.test_family.eq(family)
            conditioned.loc[mask, 'bh_q_within_assignment_and_test_family'] = bh_fdr(conditioned.loc[mask, 'lower_tail_p_value'].to_numpy(float))
        condition_rows.append(conditioned)
        geometry = pd.concat([pd.read_csv(base / f'geometry_{int(round(g * 100)):03d}/distance_correspondence.csv') for g in GAMMAS], ignore_index=True)
        geometry.insert(0, 'assignment_id', aid)
        geometry_rows.append(geometry)
        recur = pd.read_csv(base / 'recurrence/basin_recurrence_significance.csv')
        recur.insert(0, 'assignment_id', aid)
        recurrence_rows.append(recur)
        primary = geometry.loc[geometry.is_primary].iloc[0]
        support = pd.read_csv(base / 'inputs/basin_support.csv')
        overview.append({'assignment_id': aid, 'fit_dimensions': len(item['assignment']['fit_axes']), 'K': k, 'events': len(item['events']), 'edges': len(item['transitions']), 'min_events': int(support.n_events.min()), 'max_events': int(support.n_events.max()), 'self_transitions': int(np.trace(item['C'])), 'primary_D4_gamma090_pearson_r': primary.pearson_r, 'primary_D4_gamma090_permutation_p': primary.two_sided_common_label_permutation_p, 'recurrence_basins_q_lt_005': int((recur.successor_centrality_bh_q < 0.05).sum()), 'mode': mode})
    for name, frames in [('feature_routing', feature_rows), ('conditioned_routing', condition_rows), ('successor_geometry', geometry_rows), ('recurrence', recurrence_rows)]:
        csv(pd.concat(frames, ignore_index=True), out / f'{name}_all_assignments.csv')
    csv(pd.DataFrame(overview), out / 'assignment_overview.csv')
    (out / 'READ_RESULTS.md').write_text(f'# REFIT downstream results ({mode})\n\n' + ('SMOKE ONLY: 199 permutations. These p-values are pipeline diagnostics, not analysis results.\n\n' if mode == 'smoke' else 'Full run: 10,000 permutations per test, seed 42.\n\n') + 'Read assignment_overview.csv alongside feature_routing_all_assignments.csv, conditioned_routing_all_assignments.csv, successor_geometry_all_assignments.csv and recurrence_all_assignments.csv.\n\nAll five assignments were frozen upstream. D4/full successor profiles at gamma 0.90 is the common primary geometry comparison. D10/D13, other gammas, PCA truncations, and Dim0 x Dim3 Procrustes are sensitivities. Basin numbers are local to each assignment; B1 does not identify the same state across assignments.\n\nThe five analyses reuse the same events and are dependent sensitivity analyses, not five replications. No assignment is selected or ranked by downstream significance. Individual p-values and the historical within-assignment BH families are retained; no study-wide or cross-assignment multiplicity control is implied. Do not pool p-values or claim an effect merely because any assignment crosses a significance threshold. Recurrence significance is not evidence of a physical attractor.\n\nParticipant-exclusion, visit-collapse, directionality, finite-horizon, predecessor-conditioned, and publication-figure stages are not executed by this base package.\n', encoding='utf-8')

def verify_run(data, output, sig):
    metadata = json.loads((output / 'RUN.json').read_text())
    require(metadata['signature'] == sig, 'Run signature changed (inputs/code/settings/environment); use the original environment or a new output folder.')
    for item in data:
        verified_stage(output / item['id'] / 'inputs')
        for component in COMPONENTS:
            meta = verified_stage(output / item['id'] / component)
            require(meta['n_permutations'] == sig['n_permutations'], 'Permutation budget mismatch.')
    require(metadata.get('completed'), 'Run has not completed its comparison stage.')
    require(file_manifest(output / 'comparisons') == metadata['comparison_outputs'], 'Comparison outputs changed.')
    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['check', 'smoke', 'run', 'verify', 'bundle'])
    parser.add_argument('--project-root', type=Path, default=PACKAGE.parent.parent)
    parser.add_argument('--assignment-run', type=Path)
    parser.add_argument('--coordinates', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.assignment_run = (args.assignment_run or args.project_root / 'outputs/assignments').resolve()
    args.coordinates = (args.coordinates or args.project_root / 'outputs/mca/tables/mca_full_54d_row_coordinates.csv').resolve()
    args.output = (args.output or args.project_root / 'outputs/downstream' / ('smoke' if args.command == 'smoke' else 'local')).resolve()
    require(args.workers >= 1, 'workers must be positive.')
    for source in [PACKAGE, args.assignment_run, args.coordinates.parent]:
        require(args.output != source and source not in args.output.parents and (args.output not in source.parents), f'Output overlaps code/inputs: {args.output}')
    return args

def main():
    args = parse_args()
    data, fingerprints = load_inputs(args)
    print('[check] Five frozen assignments verified; 256 events, 251 edges, 5 intact global sequences in each.', flush=True)
    if args.command == 'check':
        print(json.dumps({'assignments': [x['id'] for x in data], 'assignment_run': str(args.assignment_run), 'coordinates': str(args.coordinates), 'output': str(args.output), 'environment': environment()}, indent=2))
        return
    mode = 'smoke' if args.command == 'smoke' else 'full'
    if args.command in {'verify', 'bundle'}:
        require((args.output / 'RUN.json').is_file(), f'No run at {args.output}')
        mode = json.loads((args.output / 'RUN.json').read_text())['signature']['mode']
    sig = signature(args, fingerprints, mode)
    if args.command in {'verify', 'bundle'}:
        verify_run(data, args.output, sig)
        print(f'[verified] {args.output}', flush=True)
        if args.command == 'bundle':
            destination = args.output.parent.parent / f"REFIT_Downstream_{('Smoke' if mode == 'smoke' else 'Results')}.zip"
            with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(args.output.rglob('*')):
                    if path.is_file() and '.work' not in path.parts and (path.name != '.run.lock'):
                        archive.write(path, Path('REFIT_Downstream_Results') / path.relative_to(args.output))
            print(f'[bundle] {destination}', flush=True)
        return
    if args.output.exists() and (not (args.output / 'RUN.json').exists()):
        require(not any(args.output.iterdir()), 'Output folder is nonempty and does not belong to this runner.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another downstream run is using this output folder.') from error
        path = args.output / 'RUN.json'
        if path.exists():
            metadata = json.loads(path.read_text())
            require(metadata['signature'] == sig, 'Cannot resume with changed inputs/code/settings/environment. Use a new --output folder.')
        else:
            metadata = {'created_utc': utc(), 'signature': sig, 'input_paths': fingerprints, 'completed': False, 'scope': 'Frozen-assignment routing, successor geometry and recurrence; no basin fitting/selection.'}
            json_write(path, metadata)
        jobs = []
        for item in data:
            prepare_assignment(item, args.output)
            for component in COMPONENTS:
                stage = args.output / item['id'] / component
                if stage.exists():
                    verified_stage(stage)
                    print(f"[resume] {item['id']}/{component}", flush=True)
                else:
                    jobs.append((item, str(args.output), component, sig['n_permutations']))
        print(f'[{mode}] {len(jobs)} remaining components; {min(args.workers, max(1, len(jobs)))} workers.', flush=True)
        start = time.monotonic()
        if jobs:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(worker, *job) for job in jobs]
                for future in as_completed(futures):
                    future.result()
        require(all((sha(Path(value['path'])) == value['sha256'] for value in fingerprints.values())), 'An input changed during computation; completed stages retain their original run signature.')
        build_comparisons(data, args.output, mode)
        metadata.update({'completed': True, 'completed_utc': utc(), 'last_invocation_seconds': time.monotonic() - start, 'comparison_outputs': file_manifest(args.output / 'comparisons')})
        json_write(path, metadata)
        verify_run(data, args.output, sig)
        print(f'[{mode}] complete: {args.output}', flush=True)
if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
