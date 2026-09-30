# Numerical comparison policy

Numerical agreement is evaluated by field. The default absolute tolerance is 1e-10 and relative tolerance is zero. Integer and Boolean fields are compared exactly. Labels, selections, ranks and memberships must agree; accepting a small score difference does not authorize a different model or projection.

| Tables | Fields with an allowance | Absolute tolerance |
| --- | --- | ---: |
| `kmeans_consensus_selected_k_by_dimension.csv` | `silhouette_norm`, `inverse_db_norm`, `consensus_score` | 1e-8 |
| `macro_model_summary.csv`; `full_geometry_macro_grid.csv` | `davies_bouldin`, `inverse_db_norm`, `macro_k2_k6_consensus_score` | 1e-8 |
| `full_sample_projection_winners_by_D.csv`; `full_sample_projection_candidates_D2_D13.csv` | `mean_trustworthiness`, `trustworthiness_k5`, `trustworthiness_k10`, `trustworthiness_k20` | 5e-5 |

These allowances apply only to the named fields. P values, confidence limits, effect estimates in routing/successor analyses, and the remaining fields keep their original comparison rules. No geometric-fidelity field receives an allowance in this release.

## Rationale and prerequisites

Some distance calculations obtain a squared distance by subtracting dot products of similar magnitude. Floating-point cancellation can give a tiny positive distance for an event that is exactly its singleton cluster's centroid. Taking the square root amplifies this discrepancy; Davies–Bouldin averages and subsequent min-max normalization can propagate it. Controlled calculations with direct coordinate differences preserve the cluster memberships while removing this numerical artifact. The 1e-8 allowance is restricted to the affected descriptive scores, which do not determine a different selected cluster count in the evaluated runs.

Trustworthiness uses neighbor ranks. Tied and nearly tied distances can exchange order without a meaningful displacement of event coordinates, changing the rank penalty by a small discrete amount. The 5e-5 allowance is the established projection-diagnostic tolerance. Projection names, axes, selections and reported ranks must still agree. This condition matters where two candidate projections have similar scores.

Every allowance requires the full coordinate table to pass at 1e-10. Macro-model allowances additionally require the event-membership table to pass strictly. If a required table is missing or differs, the allowance is blocked.

These are declared computational acceptance criteria, not statistical significance thresholds or a guarantee across every numerical platform. They do not change the values calculated or reported by any analysis stage.

## Reporting and geometric fidelity

The verifier records the observed maximum error, strict and allowed tolerances, and whether each column passes both criteria. A table accepted under a metric allowance is marked `accepted_with_metric_tolerance`; its original strict result remains visible. Use `--strict-numerics` to disable all allowances. Its report is saved separately with a `_STRICT` suffix.

Geometric-fidelity diagnostics explicitly exclude the event itself and apply the fixed numerical-tie rule described in `GEOMETRIC_FIDELITY.md`. Corrected reference values are independently checked against refitted coordinates. These fields retain the original 1e-10 output-comparison tolerance; no fidelity allowance was added. The ranking rule changes how numerical ties are resolved, not the underlying coordinates. Complete reproduction still requires every stage, including the full basin-fitting workflow, to finish and pass.

Technical background: [scikit-learn Euclidean distances](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html) documents cancellation in the dot-product formulation; [trustworthiness](https://scikit-learn.org/1.6/modules/generated/sklearn.manifold.trustworthiness.html) defines the rank penalties.
