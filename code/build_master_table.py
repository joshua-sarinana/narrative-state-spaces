"""Integrate public inputs and computed results without reading reference outputs."""
from pathlib import Path
import argparse
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['participant_id_norm', 'hippo_id_norm']
SOLUTIONS = {
    'basin_D4_10B': 'D4_K10_L9_U47',
    'basin_D4_9B': 'D4_K9_L8_U49',
    'basin_D4_6B': 'D4_K6_L8_U50',
    'basin_D2_9B': 'D2_K9_L8_U50',
    'basin_D2_8B': 'D2_K8_L8_U39',
}
EXTRA_COLUMNS = ['segment_id'] + [f'Dim{i}' for i in range(13)] + [
    'macro_cluster_D4_K4', 'chain_id', 'step', 'macro_cluster_prev',
    'macro_cluster_next', 'macro_seq_role', 'macro_seen_before',
    'macro_return_event'] + list(SOLUTIONS)

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def index_rows(rows, expected_keys=None):
    index = {tuple(row[k] for k in KEYS): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate participant/event key')
    if expected_keys is not None and set(index) != expected_keys:
        raise ValueError('Event keys differ between inputs')
    return index

def build(input_path, coordinates_path, macro_path, basins_path, output_path):
    events = read_csv(input_path)
    if len(events) != 256 or len(events[0]) != 19:
        raise ValueError('Expected the released 256-row, 19-field input')
    event_index = index_rows(events)
    keys = set(event_index)
    coordinates = index_rows(read_csv(coordinates_path), keys)
    macro = index_rows(read_csv(macro_path), keys)
    basins = index_rows(read_csv(basins_path), keys)
    result = []
    for event in events:
        key = tuple(event[k] for k in KEYS)
        row = dict(event)
        row['segment_id'] = row['uid'].split('::', 1)[1]
        row.update({f'Dim{i}': coordinates[key][f'Dim{i}'] for i in range(13)})
        row['macro_cluster_D4_K4'] = str(int(macro[key]['cluster_k4_d4']))
        row['chain_id'] = row['participant_id_norm'] + ':0'
        row.update({col: str(int(basins[key][run])) for col, run in SOLUTIONS.items()})
        result.append(row)
    index = {row['hippo_id_norm']: row for row in result}
    if len(index) != len(result):
        raise ValueError('Event identifiers must be globally unique')
    starts = [row for row in result if not row['hippo_prev_global']]
    if Counter(row['participant_id_norm'] for row in starts) != Counter({p:1 for p in {r['participant_id_norm'] for r in result}}):
        raise ValueError('Expected one narrative-chain start per participant')
    visited = set()
    for start in starts:
        row = start
        seen = set()
        step = 0
        while row is not None:
            eid = row['hippo_id_norm']
            if eid in visited:
                raise ValueError('Cycle or shared event in narrative chains')
            visited.add(eid)
            previous_id, next_id = row['hippo_prev_global'], row['hippo_next_global']
            if (previous_id and previous_id not in index) or (next_id and next_id not in index):
                raise ValueError('Sequence link refers to a missing event')
            previous, following = index.get(previous_id), index.get(next_id)
            for neighbor, reciprocal in [(previous, 'hippo_next_global'), (following, 'hippo_prev_global')]:
                if neighbor and (neighbor['participant_id_norm'] != row['participant_id_norm'] or neighbor[reciprocal] != eid):
                    raise ValueError('Nonreciprocal or cross-participant sequence link')
            label = row['macro_cluster_D4_K4']
            prev_label = previous['macro_cluster_D4_K4'] if previous else None
            next_label = following['macro_cluster_D4_K4'] if following else None
            row['step'] = str(step)
            row['macro_cluster_prev'] = prev_label + '.0' if prev_label else ''
            row['macro_cluster_next'] = next_label + '.0' if next_label else ''
            row['macro_seq_role'] = ('core' if prev_label == label and next_label == label else
                                     'exit' if prev_label == label else
                                     'entry' if next_label == label else 'single')
            row['macro_seen_before'] = str(label in seen)
            row['macro_return_event'] = str(label in seen and prev_label != label)
            seen.add(label)
            step += 1
            row = following
    if len(visited) != len(result) or sum(bool(r['hippo_next_global']) for r in result) != 251:
        raise ValueError('Narrative chains do not cover 256 events and 251 adjacencies')
    fields = list(events[0]) + EXTRA_COLUMNS
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(result)
    print(f'Built {len(result)} event-states, {len(fields)} fields and 251 narrative adjacencies: {output_path}')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/event_level_analysis_input.csv')
    parser.add_argument('--coordinates', type=Path, default=ROOT/'outputs/mca/tables/mca_full_54d_row_coordinates.csv')
    parser.add_argument('--macro', type=Path, default=ROOT/'outputs/macro_canonical/tables/d4_k4_event_memberships_REFIT.csv')
    parser.add_argument('--basins', type=Path, default=ROOT/'outputs/assignments/results/Final_Event_Assignments.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/integration/master_regime_table_eventlevel_REFIT.csv')
    args = parser.parse_args()
    build(args.input, args.coordinates, args.macro, args.basins, args.output)

if __name__ == '__main__':
    main()
