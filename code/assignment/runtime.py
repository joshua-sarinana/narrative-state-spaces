"""Deterministic analysis of the frozen 256-event dataset."""
import json
import multiprocessing as mp
import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def run_root():
    value = os.environ.get('REFIT_RUN_ROOT')
    if not value:
        raise RuntimeError('Use python run_refit.py; a run directory has not been declared.')
    return Path(value).resolve()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

@contextmanager
def worker_pool():
    existing = {p.pid for p in mp.active_children()}
    pool = ProcessPoolExecutor(max_workers=int(os.environ['REFIT_WORKERS']), mp_context=mp.get_context('spawn'))
    try:
        yield pool
    except BaseException:
        for process in mp.active_children():
            if process.pid not in existing:
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
