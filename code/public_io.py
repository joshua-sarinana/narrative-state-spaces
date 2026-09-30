"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

def open_table(path, *args, **kwargs):
    if Path(path).suffix.lower() == '.csv':
        return SimpleNamespace(sheet_names=['public_event_table'])
    return pd.ExcelFile(path, *args, **kwargs)

def read_table(path, sheet_name=0, **kwargs):
    if Path(path).suffix.lower() == '.csv':
        df = pd.read_csv(path, low_memory=False)
        for source, alias in [('participant_id_norm', 'participant_id'), ('hippo_id_norm', 'hippo_id')]:
            if source in df and alias not in df:
                df[alias] = df[source]
        return df
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
