"""Compare computed outputs with frozen references and declared metric tolerances."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
STRICT_ATOL=1e-10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compare_summary(root,spec):
    row={'check':spec['output'],'passed':False,'strict_passed':False,'accepted_with_metric_tolerance':False,'detail':'','columns':[]}
    try:
        actual=pd.read_csv(root/spec['output'],keep_default_na=False)
        observed={'row_count':len(actual)}
        expected={'row_count':spec['expected_row_count']}
        for col,count in spec.get('expected_true_counts',{}).items():
            observed[col]=int(actual[col].astype(str).str.lower().eq('true').sum())
            expected[col]=count
        row['passed']=row['strict_passed']=observed==expected
        row['detail']=json.dumps({'expected':expected,'actual':observed})
    except Exception as error:row['detail']=str(error)
    return row

def compare_table(root,spec,strict_numerics=False):
    if spec.get('kind')=='summary':return compare_summary(root,spec)
    row={'check':spec['output'],'passed':False,'strict_passed':False,'accepted_with_metric_tolerance':False,'detail':'','columns':[]}
    try:
        if 'reference' in spec:
            expected=pd.read_csv(root/spec['reference'],keep_default_na=False)
            expected=expected.rename(columns={v:k for k,v in spec.get('reference_columns',{}).items()})
        else:
            expected=pd.DataFrame(spec['expected_rows'],columns=spec['columns'])
        actual=pd.read_csv(root/spec['output'],keep_default_na=False)
        cols=spec['columns'];actual=actual[cols];expected=expected[cols]
        if spec.get('sort_by'):
            actual=actual.sort_values(spec['sort_by']).reset_index(drop=True);expected=expected.sort_values(spec['sort_by']).reset_index(drop=True)
        if actual.shape!=expected.shape:raise ValueError(f'shape {actual.shape}, expected {expected.shape}')
        exact=set(spec.get('exact_columns',[]))
        for col in cols:
            left=pd.to_numeric(expected[col],errors='coerce').astype(float);right=pd.to_numeric(actual[col],errors='coerce').astype(float)
            numeric=left.notna() & right.notna()
            strict_atol=0.0 if col in exact else spec.get('atol',STRICT_ATOL)
            allowed_atol=strict_atol if strict_numerics or col in exact else spec.get('atol_overrides',{}).get(col,strict_atol)
            categorical_ok=expected.loc[~numeric,col].astype(str).equals(actual.loc[~numeric,col].astype(str))
            strict_ok=bool(categorical_ok and (not numeric.any() or np.allclose(left[numeric],right[numeric],atol=strict_atol,rtol=0)))
            allowed_ok=bool(categorical_ok and (not numeric.any() or np.allclose(left[numeric],right[numeric],atol=allowed_atol,rtol=0)))
            difference=(left[numeric]-right[numeric]).abs()
            largest=float(difference.max()) if len(difference) else 0.0
            row['columns'].append({'column':col,'passed':allowed_ok,'strict_passed':strict_ok,'max_abs_difference':largest,'strict_atol':strict_atol,'allowed_atol':allowed_atol,'cells_exceeding_strict_tolerance':int((difference>strict_atol).sum()),'reason':spec.get('tolerance_reason','') if allowed_atol>strict_atol else ''})
        row['strict_passed']=all(c['strict_passed'] for c in row['columns'])
        row['passed']=all(c['passed'] for c in row['columns'])
        row['accepted_with_metric_tolerance']=row['passed'] and not row['strict_passed']
        bad=[c['column'] for c in row['columns'] if not c['passed']]
        row['detail']='mismatched columns: '+', '.join(bad) if bad else 'accepted under declared metric tolerance' if row['accepted_with_metric_tolerance'] else ''
    except Exception as error:row['detail']=str(error)
    return row

def compare_specs(root,specs,strict_numerics=False):
    rows=[compare_table(root,s,strict_numerics) for s in specs]
    by_output={s['output']:s for s in specs}
    for spec,row in zip(specs,rows):
        if not row['accepted_with_metric_tolerance']:continue
        for dependency in spec.get('requires_strict_outputs',[]):
            guard=by_output.get(dependency)
            if guard is None or not compare_table(root,guard,True)['passed']:
                row.update(passed=False,accepted_with_metric_tolerance=False,detail='metric allowance blocked: strict prerequisite failed or missing: '+dependency)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--completed-only',action='store_true',help='Diagnostic partial check; never certifies complete reproduction.')
    ap.add_argument('--strict-numerics',action='store_true',help='Disable metric allowances and show the original 1e-10 comparison.')
    ap.add_argument('--group',choices=['all','analysis'],default='all',help='Required execution scope; analysis excludes figure rendering.')
    a=ap.parse_args();rows=[]
    stages=json.loads((ROOT/'provenance/RUN_CONFIG.json').read_text())
    stages=[s for s in stages if a.group=='all' or s['group']=='analysis'];done=set()
    release=json.loads((ROOT/'provenance/RELEASE_MANIFEST.json').read_text())
    integrity=all((ROOT/rel).is_file() and sha(ROOT/rel)==digest for rel,digest in release.items() if rel.split('/')[0] in ['code','data','provenance','reference_outputs'])
    rows.append({'check':'released code, inputs and reference integrity','passed':integrity,'detail':'' if integrity else 'a released file is missing or changed'})
    fixed={str(p.relative_to(ROOT)):sha(p) for top in ['code','data','provenance'] for p in sorted((ROOT/top).rglob('*')) if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    signature=hashlib.sha256(json.dumps(fixed,sort_keys=True).encode()).hexdigest()
    for s in stages:
        p=ROOT/'outputs/execution'/f'{s["id"]}.json'
        if not p.exists():
            if not a.completed_only:rows.append({'check':s['id'],'passed':False,'detail':'missing completion record'})
            continue
        record=json.loads(p.read_text());ok=record.get('signature')==signature and record.get('stage')==s['id'] and record.get('status')=='complete' and bool(record.get('files')) and all((ROOT/k).is_file() and sha(ROOT/k)==v for k,v in record['files'].items())
        rows.append({'check':s['id']+' output integrity','passed':ok,'detail':''});done.add(s['id'])
    specs=json.loads((ROOT/'reference_outputs/quantitative_checks.json').read_text())['checks']
    specs=[s for s in specs if s['stage'] in {stage['id'] for stage in stages}]
    table_rows=compare_specs(ROOT,specs,a.strict_numerics)
    if a.completed_only:table_rows=[row for row,spec in zip(table_rows,specs) if spec['stage'] in done]
    rows.extend(table_rows)
    full=len(done)==len(stages) and bool(rows) and all(r['passed'] for r in rows) and not a.completed_only
    report={'status':('complete_reproduction_passed' if a.group=='all' else 'complete_analysis_reproduction_passed') if full else 'partial_checks' if a.completed_only and all(r['passed'] for r in rows) else 'failed','full_reproduction':full and a.group=='all','complete_requested_scope':full,'requested_scope':a.group,'numeric_policy':'strict_1e_10' if a.strict_numerics else 'declared_metric_tolerances','completed_stages':len(done),'required_stages':len(stages),'tables_accepted_with_metric_tolerance':sum(r.get('accepted_with_metric_tolerance',False) for r in rows),'checks':rows}
    out=ROOT/'outputs/verification';out.mkdir(parents=True,exist_ok=True)
    suffix='_STRICT' if a.strict_numerics else ''
    (out/f'VERIFICATION_REPORT{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    flat=[{k:v for k,v in r.items() if k!='columns'} for r in rows]
    pd.DataFrame(flat).to_csv(out/f'verification_checks{suffix}.csv',index=False)
    details=[{'table':r['check'],**c} for r in rows for c in r.get('columns',[])]
    pd.DataFrame(details).to_csv(out/f'verification_columns{suffix}.csv',index=False)
    print(report['status'],f'{sum(r["passed"] for r in rows)}/{len(rows)} checks; {len(done)}/{len(stages)} stages; {report["tables_accepted_with_metric_tolerance"]} tables accepted with metric allowances')
    if not full and not a.completed_only:sys.exit(1)
    if any(not r['passed'] for r in rows):sys.exit(1)
if __name__=='__main__':main()
