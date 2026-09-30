"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import numerics as m
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
