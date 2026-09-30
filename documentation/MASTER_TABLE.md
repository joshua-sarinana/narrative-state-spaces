# master_regime_table_eventlevel_REFIT

Public derived integration and verification table for the 256-event analysis.

- Rows: 256 event-states
- Columns: 46
- Geometry: Dim0–Dim12 from the fixed full-sample MCA solution
- Macro organization: canonical 4-D four-cluster membership (`macro_cluster_D4_K4`)
- Narrative order: reconstructed from the 251 authoritative participant-global directed edges
- Basin assignments: all five frozen REFIT solutions
- Noncanonical refinement-overlay fields, obsolete micro-state fields, and unused legacy network summaries are excluded.

The quantitative workflow begins with `data/event_level_analysis_input.csv`. This master table integrates its fields with computed coordinates and memberships. `code/build_master_table.py` reconstructs the table from public inputs and generated results without reading the reference table. `provenance/MASTER_TABLE.json` supplies the public source paths and field definitions.

## Sequence-field definitions

These fields describe **narrative order**, not the timing of the experiences being recalled. They use the canonical macro-cluster assignments, not basin-state assignments. A **run** is a consecutive stretch of event-states assigned to the same macro-cluster; a run can contain one event-state.

| Field | Definition |
| --- | --- |
| `chain_id` | Identifier for a participant's directed narrative chain. In this export, it is `participant_id_norm` followed by `:0`, with one chain per participant. Chains span transcript segments. |
| `step` | Zero-based position within the chain: the first event-state is 0, the second is 1, and so on. Positions follow the authoritative directed links; event IDs must not be sorted to infer order. |
| `macro_cluster_prev` | Macro-cluster of the immediately preceding event-state in the same chain. Blank at the beginning of the chain. |
| `macro_cluster_next` | Macro-cluster of the immediately following event-state in the same chain. Blank at the end of the chain. |
| `macro_seq_role` | Position within a consecutive run of the same macro-cluster: `single`, `entry`, `core`, or `exit`, as defined below. |
| `macro_seen_before` | `True` if the current macro-cluster occurred at any earlier step in the same chain; otherwise `False`. Earlier event-states in the current run count. |
| `macro_return_event` | `True` for the first event-state of a run when its macro-cluster occurred earlier in the same chain and the immediately preceding event-state belongs to a different macro-cluster. Otherwise `False`. It marks re-entry after an intervening macro-cluster, not continued membership in the current run. |

`hippo_prev_global` and `hippo_next_global` identify the adjacent event-states themselves; `macro_cluster_prev` and `macro_cluster_next` give their macro-cluster memberships. Blank adjacent-event fields mark chain endpoints. The adjacent macro-cluster columns serialize integer labels as values such as `1.0`; these remain categorical labels.

### Values of `macro_seq_role`

| Value | Definition |
| --- | --- |
| `single` | The run contains only this event-state. Neither adjacent event-state belongs to its macro-cluster. |
| `entry` | First event-state in a run of at least two. The next event-state belongs to its macro-cluster; the previous event-state does not. |
| `core` | Interior event-state in a run of at least three. Both adjacent event-states belong to its macro-cluster. |
| `exit` | Last event-state in a run of at least two. The previous event-state belongs to its macro-cluster; the next event-state does not. |

A missing neighbor at a chain endpoint is treated as not belonging to the current macro-cluster. For example, a chain that begins with a two-event run starts with `entry`, followed by `exit`. These labels are deterministic sequence descriptors, not interpretive annotations or event-boundary rules, and do not reinstate the excluded legacy `structural_role` field.

### Example

For the macro-cluster sequence **1 → 1 → 2 → 1 → 1 → 1**, the fields are:

| `step` | Macro-cluster | `macro_seq_role` | `macro_seen_before` | `macro_return_event` |
| --- | --- | --- | --- | --- |
| 0 | 1 | entry | False | False |
| 1 | 1 | exit | True | False |
| 2 | 2 | single | False | False |
| 3 | 1 | entry | True | True |
| 4 | 1 | core | True | False |
| 5 | 1 | exit | True | False |

The return is marked only at step 3. At steps 4 and 5, the sequence continues within that same macro-cluster.

## Public identifiers and field changes

Participant keys use p01 through p05. Event, processing-segment, chain and adjacent-event keys use the same neutral identifiers as the public event-level input. The private identifier crosswalk is excluded. These replacement keys preserve joins and narrative order.

The duplicate `hippo_id` and `uid_norm` fields are removed. The auxiliary `hippo_weight_tag` and `n_theme_assignments_event` fields are omitted because the public calculations do not use them. The latter equals `n_excerpts_event` in all supplied rows. The fields `conj_rep` and `anchor_changed_to_next_within_segment` are added from the public input so that all 19 source fields are present.

All retained feature values, Dim0–Dim12 coordinates, macro-cluster memberships, basin-state memberships and sequence descriptors are preserved. The added fields retain their source values. The supplied master table matched the existing public numerical references exactly for all 13 coordinates and all six memberships.

## Rebuilding

Run the package pipeline through `14b_master_table`, or run `python code/build_master_table.py` after the MCA, macro-cluster and basin-selection outputs exist. The generated table is written to `outputs/integration/master_regime_table_eventlevel_REFIT.csv`. The stored table in `reference_outputs/` is used for comparison only.

The 11 active feature variables are defined in the data dictionary and annotation codebook. `conj_rep` and the anchor-change flag support boundary summaries and do not enter MCA. The five basin columns identify separate solutions; equal integer labels across different columns do not establish the same basin-state.
