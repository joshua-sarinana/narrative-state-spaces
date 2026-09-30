# Frozen-assignment downstream protocol

This package consumes the reviewed, full local assignment run for chronological_256_v2. It executes all five frozen alternatives: D4_K10_L9_U47, D4_K9_L8_U49, D4_K6_L8_U50, D2_K9_L8_U50, and D2_K8_L8_U39. It performs no fitting, omission screen, basin selection, or relabeling based on downstream results. State labels are local to each assignment.

## Inputs and sequence

Use all 256 corrected events and all 251 authoritative participant-global edges, including same-basin and cross-segment transitions. Traverse those edges into five participant sequences and require exact agreement with the transition count matrix. Do not infer order from event IDs, dataframe order, or segment numbering; do not bridge removed events. Source-event features condition transitions.

Feature columns come from the frozen assignment input CSV, preserving literal `none` categories. Historical `basin_K*` columns in that CSV are discarded. Memberships come only from the selected assignment folders. D4/D10/D13 coordinates come from the original verified full MCA coordinate file. No MCA refitting, rescaling, whitening, or feature weighting is performed. Features and geometry have already contributed to assignment construction; downstream associations do not constitute held-out validation.

The empirical transition matrix is P[i,j] = C[i,j]/sum_j C[i,j]. There is no transition smoothing or added pseudocount. A zero-outgoing basin aborts the full-chain analysis rather than fabricating a transition. The selected assignments all have outgoing support. Conditional matrices retain zero rows where a condition has no source support.

## Analyses

| Component | Definition and null model |
|---|---|
| Feature routing, primary3 | Token-count profiles: combined primary/secondary mechanism presence (once per mechanism per event), primary context, event-level meta-theme. Cosine similarity of feature profiles versus cosine similarity of empirical P rows. |
| Feature routing, all11 | Separate categorical tokens for all 11 active features; primary and secondary mechanisms remain separate. Same routing comparison. |
| Feature-routing inference | Pearson and Spearman correlations across K(K-1)/2 unordered basin pairs. Common basin-label permutation of the routing similarity matrix. Two-sided absolute-statistic tail, +1 correction. |
| Conditioned routing | Primary context, combined mechanism presence, and primary meta-theme. Retain conditions with at least 10 source transitions. Compare flattened row-normalized conditional matrices by mean off-diagonal cosine similarity. |
| Supported-source conditioned routing | For each condition pair, use only source basins with at least 3 outgoing transitions in each condition. Support masks are recomputed in every permutation. Unavailable condition pairs are omitted from that mean and reported through the shared-source table. |
| Conditioned-routing inference | Globally shuffle condition payloads across the same frozen source transitions. Mechanism primary/secondary pairs move jointly, preserving overlap. Lower-tail cosine test, +1 correction. **This is not a within-participant or within-source-basin shuffle.**  |
| Successor geometry | M = solve(I - gamma P, I), including the identity term. Primary gamma 0.90; sensitivities 0.70, 0.80, 0.95. Euclidean distances between full successor rows versus Euclidean basin-centroid distances. |
| Geometry roles | Common primary comparator D4 for **all** assignments, including those fitted in Dim0 x Dim3. D10 and D13 retain the participant-robust and full-retained reference roles. No geometry is chosen by downstream significance. |
| Geometry inference | Common basin-label permutation of successor distance matrices; two-sided absolute Pearson statistic, +1 correction. The same seeded bank is reused across geometry dimensions, gamma, and component count within each K. |
| PCA and alignment sensitivities | PCA has K-1 components. Export every truncation and verify complete-PCA distances equal full-profile distances. Dim0 x Dim3 versus successor PCA2 Procrustes uses centering, unit Frobenius scaling, orthogonal alignment with reflections allowed, and a lower-tail label-permutation test. |
| Recurrence | At gamma 0.90, successor centrality is the mean of each destination column of M across equally weighted starting basins. Also report incoming transition counts and stationary probability. |
| Recurrence inference | Shuffle basin labels within each participant's global sequence, preserving that participant's basin occupancy and number of adjacency positions. Seeded, upper-tail +1 tests. This null differs from the geometry label permutation. |
| Stationary distribution | Historical constrained least-squares estimator, followed by nonnegative clipping and normalization. Record eigenvalue-one multiplicity and residuals. Nonunique cases do not identify a unique stationary distribution. |

Full runs use 10,000 Monte Carlo draws per test and seed 42. Sampling may repeat label permutations, particularly at K6; draws are not represented as unique exact permutations. Computation caches repeated label permutations without changing their frequency in the null. Seeds are reset for each analysis family and assignment. Smoke uses 199 draws in a separate directory and is only an execution check.

The array implementation of conditioned routing preserves payload order and the original random permutation sequence. An undefined observed supported-source statistic produces `not_estimable` and a missing p-value. This prevents a spurious smallest p-value when no observed condition pair has adequate shared support. 

## Multiplicity and interpretation

BH families: three condition families separately for full-matrix and supported-source tests within an assignment; K basin tests separately for each recurrence metric within an assignment. Feature-routing and geometry p-values retain the stated definitions without a new correction.

There is no cross-assignment or study-wide multiplicity correction here. The five partitions use the same events and are dependent sensitivity analyses, not five replications. Report effect sizes and sensitivity across all five, not the best p-value. Do not infer an overall positive result merely because one assignment yields significance. Raw recurrence centrality also depends on K and is not an absolute cross-K score.

The minimum 10 **conditioned transitions** and minimum 3 **source transitions per condition pair** are existing support heuristics for these downstream tests. They are distinct from the relaxed 8/9-event basin occupancy limits already used in assignment selection; this package does not reimpose a 10-event minimum on basins.

## Additional analyses

Participant exclusions, visit collapse, narrative order and profile-definition controls are executed by the controls and robustness stages. Cross-solution inference and endpoint-fixed event-distance/field tests have separate stages and protocols.
