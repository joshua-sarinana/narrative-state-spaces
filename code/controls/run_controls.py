"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import control_methods as m
PACKAGE = Path(__file__).resolve().parent
ASSIGNMENTS = ('D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39')
PARTICIPANTS = ('p01', 'p02', 'p03', 'p04', 'p05')
KEYS = ['participant_id_norm', 'hippo_id_norm']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    m.write_json(temporary, value)
    temporary.replace(path)

def file_manifest(folder):
    return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'COMPLETE.json' and (not any((x.startswith('.') for x in p.relative_to(folder).parts)))}

def check_complete(folder, signature):
    marker = folder / 'COMPLETE.json'
    m.require(marker.is_file(), f'Incomplete output folder: {folder}. Choose a new --output folder if it was manually modified.')
    record = json.loads(marker.read_text())
    m.require(record['signature'] == signature, f'Different run signature in {folder}; choose a new --output folder.')
    m.require(record['files'] == file_manifest(folder), f'Completed outputs changed in {folder}; choose a new --output folder.')
    return record

def load_data(baseline, assignment):
    folder = baseline / assignment
    read = lambda name: pd.read_csv(folder / name)
    spec = json.loads((folder / 'inputs/Assignment.json').read_text())
    k = int(spec['K'])
    labels = [f'B{i}' for i in range(1, k + 1)]
    events = read('inputs/event_memberships_features_coordinates.csv')
    edges = read('inputs/event_transition_records.csv')
    manifest = read('inputs/participant_sequence_manifest.csv')
    m.require(spec['assignment_id'] == assignment and len(events) == 256 and (len(edges) == 251), 'Unexpected assignment or event/edge totals.')
    m.require(not events.duplicated(KEYS).any() and (not manifest.duplicated(KEYS).any()), 'Duplicate event keys.')
    m.require(set(events.state) == set(range(1, k + 1)), 'Unexpected state labels.')
    m.require(tuple(sorted(manifest.participant_id_norm.unique())) == PARTICIPANTS, 'Unexpected participant set.')
    a = events[KEYS + ['state']].sort_values(KEYS).reset_index(drop=True)
    b = manifest[KEYS + ['state']].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    m.require(np.isfinite(events[[f'Dim{i}' for i in range(13)]].to_numpy(float)).all(), 'Nonfinite coordinates.')
    sequences, expected_edges = ({}, [])
    for participant, frame in manifest.groupby('participant_id_norm', sort=True):
        frame = frame.sort_values('sequence_position')
        m.require(np.array_equal(frame.sequence_position, np.arange(1, len(frame) + 1)), 'Invalid sequence positions.')
        states = frame.state.to_numpy(int)
        sequences[participant] = (states - 1).astype(np.int16)
        ids = frame.hippo_id_norm.tolist()
        expected_edges.extend(((participant, ids[i], ids[i + 1], int(states[i]), int(states[i + 1])) for i in range(len(frame) - 1)))
    edge_cols = ['participant_id_norm', 'source_event', 'target_event', 'source_state', 'target_state']
    m.require(sorted(expected_edges) == sorted(edges[edge_cols].itertuples(index=False, name=None)), 'Sequence manifest does not reproduce authoritative event edges.')

    def matrix(relative):
        frame = pd.read_csv(folder / relative, index_col=0)
        m.require(frame.index.tolist() == labels and frame.columns.tolist() == labels, f'Matrix label order changed: {relative}')
        return frame.to_numpy(float)
    c = matrix('inputs/transition_counts.csv')
    p = matrix('inputs/transition_probabilities_empirical.csv')
    successor = matrix('geometry_090/successor_matrix.csv')
    np.testing.assert_array_equal(c, m.counts_from_sequences(sequences, k))
    np.testing.assert_allclose(p, m.row_normalize(c), rtol=0, atol=1e-14)
    np.testing.assert_allclose(successor, m.successor(p), rtol=0, atol=1e-11)
    recurrence = read('recurrence/basin_recurrence_significance.csv')
    np.testing.assert_allclose(recurrence.set_index('basin').loc[labels].successor_centrality, successor.mean(0), rtol=0, atol=1e-11)
    geometry = read('geometry_090/distance_correspondence.csv')
    for d in m.DIMENSIONS:
        ref = geometry.loc[geometry.geometry_D.eq(d) & geometry.representation.eq('full_profile')]
        m.require(len(ref) == 1 and ref.iloc[0].n_permutations == m.N_PERM and (ref.iloc[0].seed == m.SEED), 'Unexpected geometry reference settings.')
        upper = np.triu_indices(k, 1)
        r = np.corrcoef(m.distances(m.centroids(events, d, k))[upper], m.distances(successor)[upper])[0, 1]
        np.testing.assert_allclose(r, ref.iloc[0].pearson_r, rtol=0, atol=1e-11)
    return {'id': assignment, 'K': k, 'labels': labels, 'events': events, 'edges': edges, 'sequence_manifest': manifest, 'sequences': sequences, 'C': c, 'P': p, 'geometry': geometry, 'recurrence': recurrence}

def run_component(baseline, output, assignment, omitted, signature):
    component = 'sequence_controls' if omitted is None else f'participant_exclusion/omit_{omitted}'
    target = output / assignment / component
    if target.exists():
        check_complete(target, signature)
        return f'reused {assignment}/{component}'
    stage = output / '.work' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        data = load_data(baseline, assignment)
        if omitted is None:
            m.persistence_and_visit_controls(data, stage)
        else:
            m.omission_control(data, stage, omitted)
    m.write_json(stage / 'COMPLETE.json', {'signature': signature, 'assignment_id': assignment, 'component': component, 'completed_utc': now(), 'seconds': time.perf_counter() - started, 'files': file_manifest(stage)})
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return f'completed {assignment}/{component} ({time.perf_counter() - started:.1f}s)'

def aggregate(output, signature):
    target = output / 'comparisons'
    if target.exists():
        check_complete(target, signature)
        return
    stage = output / '.work' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    tables = {name: [] for name in ('participant_exclusion_geometry', 'participant_exclusion_recurrence', 'persistence_overall', 'persistence_by_basin', 'visit_geometry', 'visit_recurrence')}
    for assignment in ASSIGNMENTS:
        for participant in PARTICIPANTS:
            folder = output / assignment / 'participant_exclusion' / f'omit_{participant}'
            for name, filename in [('participant_exclusion_geometry', 'geometry_correspondence.csv'), ('participant_exclusion_recurrence', 'basin_recurrence.csv')]:
                tables[name].append(pd.read_csv(folder / filename))
        folder = output / assignment / 'sequence_controls'
        for name, filename in [('persistence_overall', 'overall_persistence.csv'), ('persistence_by_basin', 'basin_persistence.csv'), ('visit_geometry', 'visit_geometry_correspondence.csv'), ('visit_recurrence', 'visit_basin_recurrence.csv')]:
            tables[name].append(pd.read_csv(folder / filename))
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}
    for name, frame in tables.items():
        m.write_csv(frame, stage / f'{name}.csv')
    summary = []
    for assignment in ASSIGNMENTS:
        geometry = tables['participant_exclusion_geometry']
        geometry = geometry.loc[geometry.assignment_id.eq(assignment) & geometry.geometry_D.eq(4)]
        visits = tables['visit_geometry']
        v = visits.loc[visits.assignment_id.eq(assignment) & visits.geometry_D.eq(4)].iloc[0]
        persistence = tables['persistence_overall']
        pers = persistence.loc[persistence.assignment_id.eq(assignment)].iloc[0]
        row = {'assignment_id': assignment, 'K': int(v.K), 'geometry_D': 4, 'gamma': m.GAMMA, 'full_event_r': v.event_adjacency_r, 'full_event_p': v.event_adjacency_p, 'visit_r': v.pearson_r, 'visit_p': v.permutation_p, 'visit_delta_r': v.delta_r_visit_minus_event, 'self_transitions': int(pers.self_transitions), 'visit_transitions': int(pers.visit_transitions), 'global_self_transition_p': pers.self_transition_upper_tail_p}
        for condition, prefix in [('fixed_full_geometry', 'fixed_centroids'), ('omission_specific_geometry', 'retained_centroids')]:
            sub = geometry.loc[geometry.geometry_condition.eq(condition)]
            valid = sub.loc[sub.status.eq('ok')]
            row[prefix + '_omissions_estimable'] = len(valid)
            row[prefix + '_omission_r_min'] = valid.pearson_r.min()
            row[prefix + '_omission_r_max'] = valid.pearson_r.max()
            row[prefix + '_omissions_p_lt_005'] = int((valid.permutation_p < 0.05).sum())
            if len(valid):
                influential = valid.loc[valid.delta_r_from_full.abs().idxmax()]
                row[prefix + '_largest_abs_delta_participant'] = influential.participant_omitted
                row[prefix + '_largest_delta_r'] = influential.delta_r_from_full
        summary.append(row)
    m.write_csv(pd.DataFrame(summary), stage / 'assignment_control_summary.csv')
    (stage / 'READ_RESULTS.md').write_text((PACKAGE / 'INTERPRETATION.md').read_text(), encoding='utf-8')
    m.write_json(stage / 'COMPLETE.json', {'signature': signature, 'completed_utc': now(), 'files': file_manifest(stage)})
    stage.rename(target)

@contextmanager
def run_lock(output):
    import fcntl
    with (output / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'A controls run is already active in {output}.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project-root', type=Path, help='Optional 0_REFIT location; inferred from the installed package by default.')
    parser.add_argument('--baseline-run', type=Path, help='Optional completed downstream run location.')
    parser.add_argument('--output', type=Path, help='Optional separate output location.')
    args = parser.parse_args()
    m.require(args.workers > 0, '--workers must be positive.')
    root = (args.project_root or PACKAGE.parent.parent).resolve()
    baseline = (args.baseline_run or root / 'outputs/downstream/local').resolve()
    output = (args.output or root / 'outputs/downstream/controls').resolve()
    m.require(output != baseline and baseline not in output.parents and (output not in baseline.parents), 'Controls output must be separate from the completed baseline run.')
    m.require(baseline.is_dir(), f'Completed downstream run not found: {baseline}')
    expected = {'files': {p: sha256(baseline / p) for p in json.loads((PACKAGE / 'required_inputs.json').read_text())}}
    for relative, wanted in expected['files'].items():
        path = baseline / relative
        m.require(path.is_file(), f'Required downstream output missing: {path}')
        m.require(sha256(path) == wanted, f'Reviewed downstream output differs: {path}. Use the completed run supplied for this controls package.')
    base_meta = json.loads((baseline / 'RUN.json').read_text())
    m.require(base_meta.get('completed') is True and base_meta['signature']['mode'] == 'full' and (base_meta['signature']['n_permutations'] == 10000) and (base_meta['signature']['seed'] == 42), 'The baseline must be the completed full downstream run (10,000 permutations, seed 42).')
    environment = {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(), 'packages': {name: version(name) for name in ('numpy', 'pandas', 'threadpoolctl')}}
    protocol = json.loads((PACKAGE / 'PROTOCOL.json').read_text())
    signature = {'protocol': protocol, 'inputs': expected['files'], 'environment': environment, 'package_files': {name: sha256(PACKAGE / name) for name in ('run_controls.py', 'control_methods.py', 'PROTOCOL.json', 'required_inputs.json', 'INTERPRETATION.md')}}
    signature_hash = digest(signature)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        started = time.perf_counter()
        runfile = output / 'RUN.json'
        if runfile.exists():
            metadata = json.loads(runfile.read_text())
            m.require(metadata['signature_hash'] == signature_hash, 'Controls code, inputs, settings, or environment changed. Choose a new --output folder to keep runs separate.')
        else:
            visible = [p for p in output.iterdir() if not p.name.startswith('.')]
            m.require(not visible, 'Output folder contains files from an unrecognized run; choose a new --output folder.')
            metadata = {'created_utc': now(), 'signature_hash': signature_hash, 'signature': signature, 'baseline_run': str(baseline), 'completed': False}
            atomic_json(runfile, metadata)
        jobs = [(assignment, omitted) for assignment in ASSIGNMENTS for omitted in (*PARTICIPANTS, None)]
        pending = []
        for assignment, omitted in jobs:
            folder = output / assignment / ('sequence_controls' if omitted is None else f'participant_exclusion/omit_{omitted}')
            if folder.exists():
                check_complete(folder, signature_hash)
            else:
                pending.append((assignment, omitted))
        m.require(not metadata.get('completed') or not pending, 'A completed run is missing component outputs; choose a new --output folder.')
        print(f'Five frozen assignments; gamma=0.90; 10,000 permutations; seed=42.', flush=True)
        print(f'{len(jobs) - len(pending)}/30 components already complete; {len(pending)} to run with {args.workers} workers.', flush=True)
        print(f'Outputs: {output}', flush=True)
        if pending:
            metadata['completed'] = False
            atomic_json(runfile, metadata)
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                futures = [pool.submit(run_component, baseline, output, assignment, omitted, signature_hash) for assignment, omitted in pending]
                for i, future in enumerate(as_completed(futures), 1):
                    print(f'[{len(jobs) - len(pending) + i}/30] {future.result()}', flush=True)
        aggregate(output, signature_hash)
        metadata.update({'completed': True, 'completed_utc': now(), 'last_invocation_seconds': time.perf_counter() - started, 'components_completed': 30, 'comparison_files': file_manifest(output / 'comparisons')})
        atomic_json(runfile, metadata)
        print(f'COMPLETE: {output}', flush=True)
        print(f"Summary: {output / 'comparisons/assignment_control_summary.csv'}", flush=True)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Repeat the same run command to resume completed components.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
