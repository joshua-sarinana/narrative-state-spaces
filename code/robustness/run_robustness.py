"""Deterministic analysis of the frozen 256-event dataset."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import uuid
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import numerics as n
import frozen_inputs as f
import robustness_methods as methods
PACKAGE = Path(__file__).resolve().parent

def jobs():
    return [(aid, kind, omit) for aid in f.ASSIGNMENTS for kind in ('event_order', 'profiles', 'visit_exclusion', 'routing_exclusion') for omit in (f.PARTICIPANTS if kind.endswith('exclusion') else (None,))]

def component_path(output, aid, kind, omit):
    return output / aid / kind / (f'omit_{omit}' if omit else 'full_sample')

def run_component(baseline, controls, output, aid, kind, omit, signature):
    target = component_path(output, aid, kind, omit)
    if target.exists():
        f.check_complete(target, signature)
        return f'reused {aid}/{kind}/{omit}'
    stage = output / '.work' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        data = f.load_data(baseline, aid)
        if kind == 'event_order':
            methods.order_control(data, stage)
        elif kind == 'profiles':
            methods.profile_controls(data, stage)
        elif kind == 'visit_exclusion':
            methods.crossed_visits(data, controls, stage, omit)
        elif kind == 'routing_exclusion':
            methods.routing_omission(data, baseline, stage, omit)
        else:
            raise ValueError(kind)
    n.write_json(stage / 'COMPLETE.json', {'signature': signature, 'assignment_id': aid, 'component': kind, 'participant_omitted': omit, 'completed_utc': f.now(), 'seconds': time.perf_counter() - started, 'files': f.file_manifest(stage)})
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return f"completed {aid}/{kind}/{omit or 'full_sample'} ({time.perf_counter() - started:.1f}s)"

def aggregate(baseline, controls, output, signature):
    target = output / 'comparisons'
    if target.exists():
        f.check_complete(target, signature)
        return
    stage = output / '.work' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    tables = {name: [] for name in ['crossed_visit_geometry', 'crossed_visit_recurrence', 'event_order', 'profile_definitions', 'feature_routing_exclusions', 'conditioned_routing_exclusions', 'condition_support_changes']}
    for aid in f.ASSIGNMENTS:
        tables['event_order'].append(pd.read_csv(component_path(output, aid, 'event_order', None) / 'order_correspondence.csv'))
        tables['profile_definitions'].append(pd.read_csv(component_path(output, aid, 'profiles', None) / 'profile_correspondence.csv'))
        for omit in f.PARTICIPANTS:
            folder = component_path(output, aid, 'visit_exclusion', omit)
            tables['crossed_visit_geometry'].append(pd.read_csv(folder / 'geometry_correspondence.csv'))
            tables['crossed_visit_recurrence'].append(pd.read_csv(folder / 'basin_recurrence.csv'))
            folder = component_path(output, aid, 'routing_exclusion', omit)
            for name, filename in [('feature_routing_exclusions', 'feature_routing.csv'), ('conditioned_routing_exclusions', 'conditioned_routing.csv'), ('condition_support_changes', 'condition_support_changes.csv')]:
                tables[name].append(pd.read_csv(folder / filename))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    for name, frame in tables.items():
        n.write_csv(frame, stage / f'{name}.csv')
    columns = ['assignment_id', 'participant_omitted', 'transition_unit', 'geometry_D', 'geometry_condition', 'pearson_r', 'permutation_p', 'status']
    event_lopo = pd.read_csv(controls / 'comparisons/participant_exclusion_geometry.csv')
    event_lopo['transition_unit'] = 'event_adjacency'
    visit_full = pd.read_csv(controls / 'comparisons/visit_geometry.csv')
    visit_full['participant_omitted'] = 'none'
    visit_full['geometry_condition'] = 'full_sample_geometry'
    pooled = []
    pooledrec = []
    for aid in f.ASSIGNMENTS:
        ref = pd.read_csv(baseline / aid / 'geometry_090/distance_correspondence.csv')
        ref = ref.loc[ref.representation.eq('full_profile')]
        ref = ref.rename(columns={'two_sided_common_label_permutation_p': 'permutation_p'})
        ref = ref.assign(assignment_id=aid, participant_omitted='none', transition_unit='event_adjacency', geometry_condition='full_sample_geometry', status='ok')
        pooled.append(ref)
        rec = pd.read_csv(baseline / aid / 'recurrence/basin_recurrence_significance.csv').rename(columns={'successor_centrality_upper_tail_p': 'upper_tail_p', 'successor_centrality_bh_q': 'bh_q'})
        rec = rec.assign(assignment_id=aid, participant_omitted='none', transition_unit='event_adjacency')
        pooledrec.append(rec)
    n.write_csv(pd.concat([pd.concat(pooled)[columns], event_lopo[columns], visit_full[columns], tables['crossed_visit_geometry'][columns]], ignore_index=True), stage / 'four_conditions_geometry.csv')
    rec_event = pd.read_csv(controls / 'comparisons/participant_exclusion_recurrence.csv').rename(columns={'upper_tail_permutation_p': 'upper_tail_p', 'bh_q_within_assignment_condition': 'bh_q'}).assign(transition_unit='event_adjacency')
    rec_visit = pd.read_csv(controls / 'comparisons/visit_recurrence.csv').rename(columns={'upper_tail_permutation_p': 'upper_tail_p', 'bh_q_within_assignment_condition': 'bh_q'}).assign(transition_unit='basin_visit')
    rec_cross = tables['crossed_visit_recurrence'].rename(columns={'bh_q_within_assignment_omission': 'bh_q'}).assign(transition_unit='basin_visit')
    rcols = ['assignment_id', 'participant_omitted', 'transition_unit', 'basin', 'successor_centrality', 'upper_tail_p', 'bh_q']
    n.write_csv(pd.concat([pd.concat(pooledrec)[rcols], rec_event[rcols], rec_visit[rcols], rec_cross[rcols]], ignore_index=True), stage / 'four_conditions_recurrence.csv')
    summary = []
    for aid in f.ASSIGNMENTS:
        cross = tables['crossed_visit_geometry']
        cross = cross.loc[cross.assignment_id.eq(aid) & cross.geometry_D.eq(4)]
        full = visit_full.loc[visit_full.assignment_id.eq(aid) & visit_full.geometry_D.eq(4)].iloc[0]
        for condition in ['fixed_full_geometry', 'omission_specific_geometry']:
            sub = cross.loc[cross.geometry_condition.eq(condition)]
            valid = sub.loc[sub.status.eq('ok')]
            summary.append({'assignment_id': aid, 'analysis': 'participant_exclusion_after_visit_collapse', 'variant': condition, 'full_sample_effect': full.pearson_r, 'full_sample_p': full.permutation_p, 'omission_effect_min': valid.pearson_r.min(), 'omission_effect_max': valid.pearson_r.max(), 'omissions_estimable': len(valid), 'omissions_nominal_p_lt_005': int((valid.permutation_p < 0.05).sum()), 'omissions_positive_effect': int((valid.pearson_r > 0).sum())})
        features = tables['feature_routing_exclusions']
        features = features.loc[features.assignment_id.eq(aid)]
        for mode, sub in features.groupby('profile_mode', sort=False):
            summary.append({'assignment_id': aid, 'analysis': 'feature_routing_participant_exclusion', 'variant': mode, 'full_sample_effect': sub.iloc[0].full_sample_r, 'full_sample_p': sub.iloc[0].full_sample_p, 'omission_effect_min': sub.pearson_r.min(), 'omission_effect_max': sub.pearson_r.max(), 'omissions_estimable': len(sub), 'omissions_nominal_p_lt_005': int((sub.pearson_permutation_p_two_sided < 0.05).sum()), 'omissions_positive_effect': int((sub.pearson_r > 0).sum())})
    n.write_csv(pd.DataFrame(summary), stage / 'assignment_sensitivity_summary.csv')
    (stage / 'READ_RESULTS.md').write_text((PACKAGE / 'METHODS.md').read_text(), encoding='utf-8')
    inventory = [{'analysis': 'participant exclusion with visit collapse', 'configurations': 25, 'manuscript_relationship': 'extension of Tables S4D and S4E'}, {'analysis': 'narrated, reversed, and shuffled event order', 'configurations': 5, 'manuscript_relationship': 'carry forward Table S4F'}, {'analysis': 'P, I+0.90P, M, and M-I profiles', 'configurations': 5, 'manuscript_relationship': 'carry forward Table S4G'}, {'analysis': 'participant exclusion for feature and conditioned routing', 'configurations': 25, 'manuscript_relationship': 'extension of Table S3B-C'}]
    n.write_csv(pd.DataFrame(inventory).assign(status='completed'), stage / 'analysis_inventory.csv')
    n.write_json(stage / 'COMPLETE.json', {'signature': signature, 'completed_utc': f.now(), 'files': f.file_manifest(stage)})
    stage.rename(target)

@contextmanager
def run_lock(output):
    import fcntl
    with (output / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another run is already active in {output}.') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--controls-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    n.require(args.workers > 0, '--workers must be positive.')
    root = (args.project_root or PACKAGE.parent.parent).resolve()
    baseline = (args.baseline_run or root / 'outputs/downstream/local').resolve()
    controls = (args.controls_run or root / 'outputs/downstream/controls').resolve()
    output = (args.output or root / 'outputs/downstream/robustness').resolve()
    for path in [baseline, controls]:
        n.require(path.is_dir(), f'Required completed run is missing: {path}')
        n.require(output != path and path not in output.parents and (output not in path.parents), 'Output must be separate from both input runs.')
    required = json.loads((PACKAGE / 'required_inputs.json').read_text())
    expected = {label: {p: f.sha256(source / p) for p in required[label]} for label, source in [('baseline', baseline), ('controls', controls)]}
    for label, source in [('baseline', baseline), ('controls', controls)]:
        for relative, wanted in expected[label].items():
            path = source / relative
            n.require(path.is_file(), f'Missing required input: {path}')
            n.require(f.sha256(path) == wanted, f'Reviewed input has changed: {path}. Use the completed runs reviewed for this package.')
        record = json.loads((source / 'RUN.json').read_text())
        n.require(record.get('completed') is True, f'{source} is incomplete.')
    n.require(json.loads((baseline / 'RUN.json').read_text())['signature']['mode'] == 'full', 'A full baseline run is required.')
    n.require(json.loads((controls / 'RUN.json').read_text())['components_completed'] == 30, 'The 30-component controls run is required.')
    environment = {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(), 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}
    protocol = json.loads((PACKAGE / 'PROTOCOL.json').read_text())
    signature = {'inputs': expected, 'protocol': protocol, 'environment': environment, 'package': {p.name: f.sha256(p) for p in sorted(PACKAGE.glob('*.py'))}, 'methods': f.sha256(PACKAGE / 'METHODS.md')}
    signature_hash = f.digest(signature)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        start = time.perf_counter()
        runfile = output / 'RUN.json'
        if runfile.exists():
            record = json.loads(runfile.read_text())
            n.require(record['signature_hash'] == signature_hash, 'Code, inputs, settings, or environment changed; use a new --output folder.')
        else:
            n.require(not [p for p in output.iterdir() if not p.name.startswith('.')], 'Output folder contains unrecognized files; use a new --output folder.')
            record = {'created_utc': f.now(), 'completed': False, 'signature_hash': signature_hash, 'signature': signature, 'baseline_run': str(baseline), 'controls_run': str(controls)}
            f.atomic_json(runfile, record)
        pending = []
        for aid, kind, omit in jobs():
            target = component_path(output, aid, kind, omit)
            if target.exists():
                f.check_complete(target, signature_hash)
            else:
                pending.append((aid, kind, omit))
        n.require(not record.get('completed') or not pending, 'A completed run is missing component outputs. Choose a new --output folder.')
        print(f'Five frozen assignments; 60 components; 10,000 permutations; seed 42; gamma 0.90.', flush=True)
        print(f'{60 - len(pending)}/60 already complete; {len(pending)} to run with {args.workers} workers.', flush=True)
        print(f'Outputs: {output}', flush=True)
        if pending:
            record['completed'] = False
            f.atomic_json(runfile, record)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(run_component, baseline, controls, output, aid, kind, omit, signature_hash) for aid, kind, omit in pending]
                for i, future in enumerate(as_completed(futures), 1):
                    print(f'[{60 - len(pending) + i}/60] {future.result()}', flush=True)
        aggregate(baseline, controls, output, signature_hash)
        record.update({'completed': True, 'components_completed': 60, 'completed_utc': f.now(), 'last_invocation_seconds': time.perf_counter() - start, 'comparison_files': f.file_manifest(output / 'comparisons')})
        f.atomic_json(runfile, record)
        print(f'COMPLETE: {output}', flush=True)
        print(f"Combined geometry: {output / 'comparisons/four_conditions_geometry.csv'}", flush=True)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Repeat the same command to resume completed components.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
