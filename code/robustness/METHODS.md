# Methods and interpretation

This package operates on the five frozen assignments D4_K10_L9_U47, D4_K9_L8_U49, D4_K6_L8_U50, D2_K9_L8_U50, and D2_K8_L8_U39. The completed baseline supplies 256 event records, their MCA coordinates, 251 authoritative within-participant global edges, features, and memberships. The previously reviewed controls supply event-omission and pooled-visit references. Both input runs are read-only.

All inferred quantities are conditional on those representations. Basin membership and MCA are not refitted. D4, comprising Dim0 through Dim3, is the primary centroid geometry for every assignment; D10 and D13 are references. This also applies to the two assignments fitted on Dim0 × Dim3. State numbers are local to each assignment.

Successor matrices solve (I − 0.90P)M = I from unsmoothed row-normalized counts. Zero-outgoing rows remain zero, with their identity contribution in M; no self-loop or uniform row is inserted. Geometry uses Euclidean distances between complete profile rows and between event-weighted basin centroids. Missing centroids make the relevant geometry comparison explicitly not estimable.

The common-label null applies the same permutation to the rows and columns of a profile-distance or routing-similarity matrix. It preserves the matrix's internal relationships while randomizing correspondence with the fixed comparator. Each K uses a seed-42 bank of 10,000 Monte Carlo draws with replacement, retaining duplicate draws. Geometry and feature-routing tests use two-sided absolute correlations and +1 p-value correction. Correlation changes between conditions are descriptive; separate p-values are not tests comparing those changes.

## Participant exclusion with visit collapse

For each assignment and participant, omit that participant's entire sequence, then collapse each remaining maximal consecutive same-basin run to one visit. At the count level this equals the retained event count matrix with the diagonal zeroed. Exported visit edges use the original event IDs at each run boundary; no edge bridges participants. Basin centroid calculations continue to use event records, not visit-weighted records.

Recalculate the visit transition matrix and successor profiles. Compare with both full-sample centroids (`fixed_full_geometry`) and centroids recalculated from retained events in the existing coordinates (`omission_specific_geometry`). Both are reported against full-sample visit results and the corresponding event-adjacency omission results.

Visit recurrence is the mean of each successor column over equally weighted starting basins. For each omission, reset seed 42, shuffle original retained event labels independently within each participant, THEN collapse runs for every draw. Each participant's event occupancy is preserved; the number of visits can change. Upper-tail +1 tests receive BH correction across K basins separately for each assignment and omission. Full-visit and omitted-event recurrence references are included.

At fixed gamma, one discount step now means one visit instead of one event. This changes the effective horizon in event units. The crossed control tests sensitivity to participant omission under that transition definition; it does not isolate a causal dwell effect.

## Event order and manuscript Table S4E

Use the full event-adjacency system. Reverse each complete participant sequence separately and verify that reversed counts equal the transpose of narrated counts. Recompute row normalization and successor profiles; reversing a normalized P directly would be incorrect.

Independently shuffle complete event positions within each participant 10,000 times, preserving event identity, coordinates, features, membership, and participant occupancy. Counts only require the attached frozen state label, so the implementation operates on indexed label arrays without changing any event attributes. Recompute P and M for each draw and correlate full-profile distances with D4/D10/D13 centroids. Shuffling order leaves centroid membership and feature counts unchanged.

The narrated and reversed observed correlations are each compared with the SAME order-shuffle null using an upper-tail +1 test, matching the manuscript method. Separately, common basin-label permutations give a two-sided alignment test for each observed order. These nulls answer different questions. The package makes no direct significance claim about the difference between narrated and reversed correlations.

## Profile definitions and manuscript Table S4F

On full event-adjacency P, compare P, I + 0.90P, standard M = (I − 0.90P)⁻¹, and M − I. D4 is primary; D10/D13 are reference comparisons. All use the same common-label permutation procedure. Identity alone has constant distance √2 between distinct rows, so its geometry correlation is undefined and is not assigned a p-value.

The identity-removed profile is the unscaled M − I. A positive common scalar, such as the normalized future-only convention, changes distance magnitudes but not their Pearson correlation in exact arithmetic. These selected definitions carry forward the manuscript's reported controls; this is not a new full finite-horizon sweep or horizon-selection procedure.

## Participant-exclusion routing

Retain original event adjacency after removing one participant. Recalculate feature count profiles from retained events and P from retained original edges. The primary feature profile combines mechanism presence (deduplicated primary/secondary mechanisms per event), primary context, and event-level meta-theme. The reference profile uses all 11 MCA-active variables. Correlate cosine feature similarity with cosine routing-profile similarity using Pearson and Spearman correlations and common-label permutations. No additional fixed-full-feature-profile variant is introduced.

Conditioned routing retains the original definitions: source-event primary context, joint primary/secondary mechanism presence, and source-event meta-theme. Reapply the existing threshold of at least 10 source transitions per condition within each omission. For supported-source comparisons, require at least three outgoing transitions in both conditions at a source basin. These thresholds concern downstream support and are separate from the 8/9-event basin-size limits used during assignment selection.

For each family, globally shuffle condition payloads across retained source transitions. Mechanism pairs move jointly before overlapping condition groups are rebuilt. The shuffled unit remains a retained source transition, not a participant and not a source basin. This preserves the specified null and its limitations. Support masks are recalculated for every draw. The full-system statistic averages cosine similarities of flattened conditional probability matrices; the supported-source statistic uses only the shared adequately supported rows for each condition pair, flattened before cosine calculation.

Lower-tail tests use +1 correction. For an undefined observed supported-source statistic, report not estimable with a missing p-value. A finite supported-source p-value uses its finite null draws, and their number is reported. Fewer than two eligible conditions makes the family not estimable. The support-change table explicitly reports every retained/dropped condition and its retained transition count.

BH correction covers the three feature families separately for full-system and supported-source tests within each assignment and omission. If a family is not estimable, it occupies a p=1 placeholder solely during the three-test adjustment; its own reported p and q remain missing. This keeps the declared family size fixed. Full-sample references are supplied alongside omission effects. Changing condition eligibility changes the aggregate being compared; inspect the support table before attributing a shift exclusively to routing structure.

The numerical routines reuse ranks for Spearman label permutations because matrix relabeling only reorders the same pair values. Supported-source cosine calculations use direct normalized dot products. These optimizations are checked against the original SciPy/scikit-learn calculations and do not alter the defined null or support rules.

## Reading the combined results

`four_conditions_geometry.csv` contains all full-sample event, omitted-participant event, full-sample visit, and omitted-participant visit results. `four_conditions_recurrence.csv` contains the corresponding basin-wise centrality tests. `event_order.csv`, `profile_definitions.csv`, and routing tables provide the remaining controls. `assignment_sensitivity_summary.csv` reports descriptive omission ranges and counts of nominal p < .05; these counts are not a new inferential test or assignment-selection criterion.

Geometry, order, and feature-routing p-values remain nominal. Local BH correction does not claim study-wide correction across assignments, omissions, dimensions, or metrics. All five partitions use the same events and overlapping participant subsets; they are dependent sensitivity views. The package records numerical evidence but does not automatically rewrite manuscript claims, declare participant-level replication, infer attractors, or select a new partition.
