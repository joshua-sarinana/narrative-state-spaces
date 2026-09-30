"""Execute the declared stages with checked resumption and separate outputs."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(paths):
    files={}
    for rel in paths:
        p=ROOT/rel
        if not p.exists():raise FileNotFoundError(f'Missing completed-stage output: {rel}')
        for f in (sorted(p.rglob('*')) if p.is_dir() else [p]):
            if f.is_file() and '__pycache__' not in f.parts and not any(v.startswith('.') for v in f.relative_to(ROOT).parts):files[str(f.relative_to(ROOT))]=sha(f)
    if not files:raise ValueError('Stage produced no files.')
    return files
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--list',action='store_true');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--through');ap.add_argument('--group',choices=['all','analysis'],default='all');a=ap.parse_args()
    stages=json.loads((ROOT/'provenance/RUN_CONFIG.json').read_text());stages=[s for s in stages if a.group=='all' or s['group']=='analysis']
    if a.list:
        for s in stages:print(s['id'],s['group'])
        return
    if a.workers<1:raise ValueError('--workers must be positive.')
    release=json.loads((ROOT/'provenance/RELEASE_MANIFEST.json').read_text())
    for rel,wanted in release.items():
        if rel.split('/')[0] in ['code','data','provenance'] and (not (ROOT/rel).is_file() or sha(ROOT/rel)!=wanted):
            raise RuntimeError(f'Release input or code changed: {rel}. Use the declared frozen release.')
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Reproduction requires Python 3.12 and the pinned requirements.')
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        name,version=line.split('==')
        if importlib.metadata.version(name)!=version:raise RuntimeError(f'Expected {name}=={version}; install requirements.txt in the reproduction environment.')
    if a.through:
        ids=[s['id'] for s in stages]
        if a.through not in ids:raise ValueError('Unknown --through stage.')
        stages=stages[:ids.index(a.through)+1]
    fixed={str(p.relative_to(ROOT)):sha(p) for top in ['code','data','provenance'] for p in sorted((ROOT/top).rglob('*')) if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    sig=hashlib.sha256(json.dumps(fixed,sort_keys=True).encode()).hexdigest()
    env=os.environ.copy();env.update(REFIT_RUN_ROOT=str(ROOT/'outputs/assignments'),REFIT_WORKERS=str(a.workers),PYTHONDONTWRITEBYTECODE='1')
    for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[name]='1'
    records=ROOT/'outputs/execution';records.mkdir(parents=True,exist_ok=True)
    for s in stages:
        marker=records/(s['id']+'.json')
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['signature']!=sig or old['files']!=manifest(s['outputs']):raise RuntimeError(f'Cannot reuse modified stage {s["id"]}. Use a separate clean repository for a changed run.')
            print('Verified existing '+s['id'],flush=True);continue
        start=time.monotonic();log=records/(s['id']+'.log');print('Running '+s['id'],flush=True)
        with log.open('a') as stream:
            for cmd in s['commands']:
                result=subprocess.run([sys.executable]+[str(a.workers) if v=='{workers}' else v for v in cmd],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'{s["id"]} failed (exit {result.returncode}); inspect {log}. No completion record written.')
        record={'stage':s['id'],'status':'complete','signature':sig,'seconds':time.monotonic()-start,'python':platform.python_version(),'files':manifest(s['outputs'])}
        temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(marker)
        print(f'Completed {s["id"]} ({record["seconds"]:.1f} s)',flush=True)
    print('Requested stages complete. Run code/verify_reproduction.py for reference comparisons.')
if __name__=='__main__':main()
