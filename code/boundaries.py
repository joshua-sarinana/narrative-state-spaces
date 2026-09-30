"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse as _dispatch_argparse
import sys as _dispatch_sys

def run_boundary_summary():
    import argparse
    import json
    import re
    import traceback
    from pathlib import Path
    import pandas as pd
    from public_io import open_table, read_table
    N_EVENTS = 256
    N_PARTICIPANTS = 5
    N_SEGMENTS = 21
    N_TRANSITIONS = 251
    N_WITHIN = 235
    N_CROSS = 16
    SHEET = 'hippo_all_events'
    CORE_FEATURES = ['primary_hipp_mech', 'context_type_primary', 'anchor_text', 'anchor_type', 'conj_rep']
    SUMMARY_ROWS = [('Specific anchor changed', 'anchor_text_changed'), ('Primary mechanism changed', 'primary_hipp_mech_changed'), ('Anchor category changed', 'anchor_type_changed'), ('Primary context changed', 'context_type_primary_changed'), ('Representational domain changed', 'conj_rep_changed'), ('Multiple event-defining features changed', 'multiple_core_features_changed'), ('Specific anchor changed while anchor category remained stable', 'anchor_changed_category_stable'), ('Specific anchor was the only event-defining feature that changed', 'anchor_only_core_feature_changed')]

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f'QA FAIL: {message}')

    def norm_id(value) -> str:
        if pd.isna(value):
            return ''
        return str(value).strip().lower()

    def norm_value(value) -> str:
        if pd.isna(value):
            return 'none'
        text = re.sub('\\s+', ' ', str(value).strip()).lower()
        return text if text else 'none'

    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument('--event-xlsx', required=True, type=Path)
        p.add_argument('--out-dir', required=True, type=Path)
        return p.parse_args()

    def load_events(path: Path) -> pd.DataFrame:
        require(path.is_file(), f'event workbook not found: {path}')
        raw = read_table(path, sheet_name=SHEET)
        raw.columns = [str(c).strip() for c in raw.columns]
        required = ['participant_id', 'hippo_id', 'uid', 'hippo_prev_global', 'hippo_next_global', *[f for f in CORE_FEATURES if f != 'anchor_text'], 'anchor_changed_to_next_within_segment']
        missing = [c for c in required if c not in raw.columns]
        require(not missing, f'{SHEET} missing required columns: {missing}')
        use = raw[required].copy()
        use['participant_id'] = use['participant_id'].map(norm_id)
        use['hippo_id'] = use['hippo_id'].map(norm_id)
        use['uid'] = use['uid'].map(norm_id)
        use['hippo_prev_global'] = use['hippo_prev_global'].map(norm_id)
        use['hippo_next_global'] = use['hippo_next_global'].map(norm_id)
        for feature in CORE_FEATURES:
            if feature != 'anchor_text':
                use[feature] = use[feature].map(norm_value)
        require(len(use) == N_EVENTS, f'rows={len(use)}, expected {N_EVENTS}')
        require(not use[['participant_id', 'hippo_id']].duplicated().any(), 'duplicate participant_id × hippo_id rows')
        require(use['participant_id'].nunique() == N_PARTICIPANTS, f"participants={use['participant_id'].nunique()}, expected {N_PARTICIPANTS}")
        require(use['uid'].nunique() == N_SEGMENTS, f"uid segments={use['uid'].nunique()}, expected {N_SEGMENTS}")
        require((use['participant_id'] != '').all(), 'blank participant_id')
        require((use['hippo_id'] != '').all(), 'blank hippo_id')
        require((use['uid'] != '').all(), 'blank uid')
        return use

    def build_transition_table(events: pd.DataFrame) -> pd.DataFrame:
        lookup = events.set_index(['participant_id', 'hippo_id'], drop=False)
        rows = []
        for source in events.itertuples(index=False):
            if source.hippo_next_global == '':
                continue
            target_key = (source.participant_id, source.hippo_next_global)
            require(target_key in lookup.index, f'unresolved global next-event pointer: {target_key}')
            target = lookup.loc[target_key]
            row = {'participant_id': source.participant_id, 'source_event': source.hippo_id, 'target_event': source.hippo_next_global, 'source_uid': source.uid, 'target_uid': target['uid'], 'within_segment': bool(source.uid == target['uid'])}
            changed_count = 0
            for feature in CORE_FEATURES:
                if feature == 'anchor_text':
                    flag = source.anchor_changed_to_next_within_segment
                    changed = bool(not pd.isna(flag) and int(flag) == 1)
                else:
                    src_value = getattr(source, feature)
                    dst_value = target[feature]
                    changed = bool(src_value != dst_value)
                row[f'{feature}_changed'] = changed
                changed_count += int(changed)
            row['n_core_features_changed'] = changed_count
            row['multiple_core_features_changed'] = bool(changed_count > 1)
            row['anchor_changed_category_stable'] = bool(row['anchor_text_changed'] and (not row['anchor_type_changed']))
            row['anchor_only_core_feature_changed'] = bool(row['anchor_text_changed'] and changed_count == 1)
            rows.append(row)
        transitions = pd.DataFrame(rows)
        require(len(transitions) == N_TRANSITIONS, f'participant-global transitions={len(transitions)}, expected {N_TRANSITIONS}')
        return transitions

    def make_summaries(within: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        table_rows = []
        for label, column in SUMMARY_ROWS:
            n = int(within[column].sum())
            table_rows.append({'boundary_characteristic': label, 'n': n, 'percent_of_235': round(100.0 * n / N_WITHIN, 1)})
        full_rows = []
        for feature in CORE_FEATURES:
            column = f'{feature}_changed'
            n = int(within[column].sum())
            only = int((within[column] & within['n_core_features_changed'].eq(1)).sum())
            full_rows.append({'feature': feature, 'feature_role': 'core_event_identity', 'n_changed': n, 'pct_of_235_boundaries': 100.0 * n / N_WITHIN, 'n_only_core_feature_changed': only})
        return (pd.DataFrame(table_rows), pd.DataFrame(full_rows))

    def run(args: argparse.Namespace) -> None:
        events = load_events(args.event_xlsx)
        transitions = build_transition_table(events)
        within = transitions.loc[transitions['within_segment']].copy()
        cross = transitions.loc[~transitions['within_segment']].copy()
        require(len(within) == N_WITHIN, f'within-segment boundaries={len(within)}, expected {N_WITHIN}')
        require(len(cross) == N_CROSS, f'cross-segment joins={len(cross)}, expected {N_CROSS}')
        table_s1c, full_frequency = make_summaries(within)
        within.to_csv(args.out_dir / 'within_segment_boundary_feature_changes.csv', index=False)
        table_s1c.to_csv(args.out_dir / 'Table_S1D_boundary_feature_change_summary.csv', index=False)
        full_frequency.to_csv(args.out_dir / 'feature_change_frequency_full.csv', index=False)
        cross.to_csv(args.out_dir / 'cross_segment_joins_EXCLUDED_FROM_S1D_QA.csv', index=False)
        report = ['# Table S1D boundary-feature co-change', '', 'This is a descriptive co-change audit of the established event partition.', '', f'- Events: **{len(events)}**', f"- Participants: **{events['participant_id'].nunique()}**", f"- Processing segments (`uid`): **{events['uid'].nunique()}**", f'- Participant-global transitions: **{len(transitions)}**', f'- Within-segment boundaries analyzed: **{len(within)}**', f'- Cross-segment joins excluded: **{len(cross)}**', '', '## Table S1D', '', '```text', table_s1c.to_string(index=False), '```', '', '## Interpretation constraint', '', 'These values describe which event-level fields differ across already-established boundaries. They do not establish that any field independently caused a boundary.']
        (args.out_dir / 'TABLE_S1D_BOUNDARY_FEATURE_CHANGES_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        metadata = {'analysis': 'repaired Table S1D boundary-feature co-change export', 'event_xlsx': str(args.event_xlsx), 'sheet': SHEET, 'events': len(events), 'participants': int(events['participant_id'].nunique()), 'processing_segments_uid': int(events['uid'].nunique()), 'participant_global_transitions': len(transitions), 'within_segment_boundaries': len(within), 'cross_segment_joins_excluded': len(cross), 'core_features': CORE_FEATURES, 'comparison_normalization': 'missing/blank->none; strip; collapse whitespace; lowercase', 'event_partition_modified': False, 'annotation_inference_performed': False, 'mca_used': False, 'kmeans_used': False, 'basin_used': False, 'successor_used': False, 'chain_step_adjacency_reconstructed': False}
        (args.out_dir / 'RUN_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print('[OK] Table S1D export complete.')
        print(f'[OK] Events: {len(events)} / {N_EVENTS}')
        print(f"[OK] Participants: {events['participant_id'].nunique()} / {N_PARTICIPANTS}")
        print(f"[OK] Processing segments (uid): {events['uid'].nunique()} / {N_SEGMENTS}")
        print(f'[OK] Participant-global transitions: {len(transitions)} / {N_TRANSITIONS}')
        print(f'[OK] Within-segment boundaries: {len(within)} / {N_WITHIN}')
        print(f'[OK] Cross-segment joins excluded: {len(cross)} / {N_CROSS}')
        print('[OK] No chain/step adjacency reconstruction used')
        print('[OK] No downstream analysis used')
        print('')
        print(table_s1c.to_string(index=False))
        print(f'\n[OK] Output: {args.out_dir.resolve()}')

    def main() -> None:
        args = parse_args()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            run(args)
        except Exception:
            failure = traceback.format_exc()
            (args.out_dir / 'FAILURE.txt').write_text(failure, encoding='utf-8')
            raise
    main()
STAGES = {'boundary_summary': run_boundary_summary}

def _dispatch():
    parser = _dispatch_argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args, remainder = parser.parse_known_args()
    _dispatch_sys.argv = [_dispatch_sys.argv[0], *remainder]
    STAGES[args.stage]()
if __name__ == '__main__':
    _dispatch()
