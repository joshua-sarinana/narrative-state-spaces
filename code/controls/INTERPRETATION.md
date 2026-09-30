# Reading the controls

All five assignments are retained without selecting a winner from downstream p-values. D4 geometry (Dim0 through Dim3) and gamma 0.90 are primary for these controls; D10 and D13 are reference views. This common D4 comparison also applies to the two assignments originally fitted on Dim0 × Dim3. Basin numbers identify states only within their own assignment; matching basin numbers across assignments does not establish matching membership.

## Participant exclusion

`participant_exclusion_geometry.csv` reports every participant omitted, every geometry dimension, and both centroid conditions. Each omission removes that participant's entire sequence, preserving the remaining within-participant edges and frozen assignments.

- `fixed_full_geometry` retains centroids computed from all events and recalculates transition probabilities and successor profiles after omission. It isolates transition sensitivity conditional on full-sample geometry.
- `omission_specific_geometry` also recalculates centroids using the retained events in the existing MCA coordinates. It does not refit MCA or cluster membership.

Each geometry test correlates Euclidean distances between complete successor profiles with Euclidean distances between basin centroids. Its two-sided null applies the same random basin permutation to the rows and columns of the successor-distance matrix. There are 10,000 Monte Carlo draws with replacement, including repeated permutations for small K; repeated draws are not deduplicated in inference. The seed-42 bank is shared across geometries and omissions for a given K.

The reported changes in correlation are descriptive sensitivity measures, not formal tests comparing dependent correlations. The largest absolute change identifies an influential omission for examination; it does not justify excluding that participant. These are five leave-one-participant-out sensitivity checks, not out-of-sample prediction or a population-level participant bootstrap.

`participant_exclusion_recurrence.csv` reports mean successor-column centrality for every basin under each omission. Each null shuffles the retained participants' event labels separately, preserving each participant's basin occupancy and event count. Upper-tail p-values receive BH correction across K basins separately for each assignment and omission. Seed 42 is reset for each omission.

## Self-transition persistence

`persistence_overall.csv` tests the total number of same-basin event edges against a within-participant event-label shuffle. `persistence_by_basin.csv` reports the primary local family of same-basin counts and the secondary diagnostic family of empirical self-transition probabilities. Each family receives its own BH correction across K basins. Global tests are reported as nominal p-values.

Event labels are shuffled independently within each participant over the authoritative sequence positions, preserving participant-specific basin occupancy. A lower-tail test of the number of remaining visit transitions is exactly the complement of the global self-count test; these are the same test, not independent support. Structured persistence does not by itself justify removing self transitions from the primary event-adjacency analysis.

## Visit collapse

Consecutive events in the same basin become a single visit, separately within each participant. At the first-order transition-count level, this equals the event count matrix with its diagonal set to zero. A separate diagonal-removal analysis would duplicate this count-level control. Visit-level transition probabilities and successor profiles are then recalculated.

`visit_geometry.csv` compares visit-based successor geometry with the original full-event centroids. `visit_recurrence.csv` tests visit-based successor centrality. For its null, shuffle the original event labels within each participant FIRST, THEN collapse consecutive repeats for each draw. Shuffling the already-collapsed visits would answer a different question. Event occupancy is preserved; the number of visits and visit transitions may change. Persistence and visit controls reuse identical seed-42 shuffled event sequences, preserving the specified calculations while avoiding duplicate work.

The visit discount step is one basin visit, whereas the event discount step is one event. Keeping gamma at 0.90 changes the effective horizon in event units. Treat this as sensitivity to transition definition, not a causal separation of dwell and movement. Event adjacency remains the primary unit.

## Numerical and interpretation limits

Transition probabilities are empirical and unsmoothed. Zero-outgoing rows stay zero. Successor matrices solve `(I - gamma P) M = I`, including any substochastic rows. Such a row contributes its identity term and no future transitions; it is not filled with a self-loop or uniform transitions. Zero-support and absent-basin diagnostics are exported. If omission removes all members of a basin, the frozen K is retained; omission-specific geometry is explicitly not estimable instead of inventing a centroid.

Permutation p-values use the +1 correction. Geometry p-values are two-sided; recurrence and persistence p-values are upper-tailed. BH families are local to the stated assignment and condition. No global correction is claimed across assignments, omissions, geometry dimensions, or metrics. The assignments and omissions are dependent sensitivity views, not independent replications. A shift across p=.05 is not itself evidence that effects differ.

These controls assess successor geometry, successor-column recurrence, and persistence. They do not rerun feature-routing or conditioned-routing associations under participant omission, and do not cross every participant omission with visit collapse. Those would be additional analyses. Frozen assignments make this a conditional sensitivity audit; full representation/assignment refitting under participant omission is outside this package.

`assignment_control_summary.csv` puts the primary D4 results side by side. Detailed tables include all omissions, all reference geometries, all basins, and all permutation draws. Participant event/visit manifests and boundary event IDs make removed repetitions and retained transitions traceable.
