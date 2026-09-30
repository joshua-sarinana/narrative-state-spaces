# REFIT flow and macro-to-basin tests

Statistics, null and correction families fixed before this package's first
result inspection, 2026-09-26. The simulation count was subsequently increased
for Monte Carlo precision as documented below; no statistic or null changed.
These are exploratory additional analyses of the existing five participants.

## Frozen inputs

Use all 256 current REFIT events and all 251 authoritative within-participant
links. Reconstruct each narrative by following those links; never sort event
identifiers to infer chronology. Use the frozen D4/K4 macro-clusters and the
five selected local assignments. D2 means the Dim0 × Dim3 projection; D4
means Dim0 through Dim3. Macro-clusters remain D4/K4 for every comparison.
Verify reviewed SHA-256 hashes and independent event, edge and coordinate
sources. Do not refit geometry, reassign events or select basin solutions.

## Common randomization

For each participant independently, hold the first and last event fixed and
uniformly permute the interior whole event records. Features, coordinates,
macro-clusters and all basin labels remain attached to their events. Rebuild
the adjacent links. This preserves each participant's source-event and
destination-event sets and therefore their exact source/destination margins.
It also preserves the outgoing support of the frozen assignments.

Use the same 1,000,000 randomizations for all six primary statistics, with NumPy
PCG64 seed 20260926. An upper-tail Monte Carlo p-value is
(1 + number of null statistics at least as large as observed)/(1,000,000 + 1).
Numerical ties use a 1e-12 tolerance scaled by max(1, |observed statistic|).
Report raw p-values, Holm correction across the five macro-to-basin tests,
and Holm correction across all six new tests. Use the latter for the main
significance statements, at 0.05. This is not a correction across all tests
in the manuscript. Do not add tests after viewing the results.

Precision amendment: the initial fixed 10,000-draw execution yielded a
six-test Holm p-value of 0.048895 for D4_K10_L9_U47, close to 0.05. To reduce
Monte Carlo uncertainty, increase to a fixed 1,000,000 draws for every test,
using the same seed and preserving the first 10,000 draws. This simulation
budget was fixed before running the extension; do not stop when a desired
significance label is obtained. Export raw p-value Monte Carlo standard errors.

The null is random interior narrative order conditional on each participant's
event set and endpoints. It is stronger than a null of zero particular
association: it removes temporal persistence and other interior order
structure. Rejection establishes departure from this random-order reference;
it does not identify a unique mechanism, eliminate temporal confounding, or
provide five hundred independent observations. The fits and selected
solutions are treated as frozen; uncertainty and all aspects of upstream
selection are not propagated. These are conditional exploratory results for
the observed narratives, not a population-level replication study.

## Empirical displacement field

Use only Dim0 × Dim3, matching the current Figure 2 empirical field: a 25 × 25
grid spanning all event coordinates, padded by 5%; source count and summed
displacements Gaussian-smoothed at sigma=0.75, mode='nearest'; visible cells
require smoothed source count >=0.05. Divide smoothed displacement sums by
smoothed source counts to obtain the mean one-step displacement vector.

Random order itself produces an inward tendency because source coordinates
constrain the possible displacement. Therefore center the observed field on
the exact expected field under the endpoint-fixed null, rather than zero.
For a participant with m interior events and their coordinate sum S, the
first event has expected successor S/m. Interior event i has expected
successor (S - x_i + x_last)/m. Apply the same smoothing and count division.

The one global statistic is the count-weighted mean squared distance between
the observed vector field and this exact null-expected field, over the fixed
visible cells. Weights are smoothed source counts normalized to sum to one.
The statistic is in squared raw MCA-coordinate units. Recompute it for each
randomized chain. Verify the accelerated linear calculation against direct
histogram-and-Gaussian smoothing. Grid cells are not independent samples.
There are no cell-wise, context-wise, streamline-wise or arrow-wise tests.
The global result cannot be assigned to an individual context-specific S3
field or interpreted as evidence for continuous trajectories or attractors.

## Macro-cluster at source → basin at destination

For each selected assignment compute conditional mutual information
I(C_t; B_(t+1) | participant), using natural logarithms. Compute the empirical
mutual information within each participant's 4 × K count matrix and average
it with weights equal to that participant's link count divided by 251.
Include self-label/persistence links. Compare the statistic with the same
interior-order null. Small-sample positive bias is handled by the permutation
comparison, not by interpreting raw mutual information alone; also report
the null mean and observed-minus-null-mean difference.

This tests temporal macro-to-next-basin dependence under the random-order
reference. It does not test centroid distance versus routing, isolate
cross-macro transitions, show that macro identity adds to current basin
identity, or identify which individual macro-to-basin links are reliable.

## Incremental prediction and support audit

Descriptively compare predictors of the next basin using (i) current macro
identity and (ii) current macro plus current basin identity. Leave one whole
participant out when estimating transition probabilities. The feature
geometry and labels remain frozen from the full event set: this is held-out
evaluation of the routing model, not an end-to-end unseen-participant test.

Use a fixed total global prior mass of 1 spread uniformly across K targets:
q_j=(n_j+1/K)/(N+1). Use a fixed total shrinkage mass alpha=10 for both
conditional models:

P0(j|c)=(n_cj+10*q_j)/(n_c+10)

P1(j|c,b)=(n_cbj+10*P0(j|c))/(n_cb+10).

No tuning or model selection. An unseen source combination falls back to
the macro-only predictor. Average log[P1/P0] over held-out links and report
each participant separately. Positive gain means improved prediction for
the frozen representations. Report how many held-out links have source
combinations supported by at least eight training links from at least three
training participants, and how many have zero training links. Those support
cutoffs are descriptive heuristics, not validation criteria.

Do not compute a p-value for this incremental comparison using unrestricted
order shuffles: that null destroys the macro-level routing structure that
the incremental test would need to preserve. No significance claim is made
for incremental prediction in this package. A valid conditional null or an
independent adequately supported validation set is needed for that claim.
