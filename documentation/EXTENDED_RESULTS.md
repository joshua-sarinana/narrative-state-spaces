# Extended results

These 13 tables retain publication rounding. They provide the extended results cited in Supplement S7. Numerical reproduction uses the full-precision calculated outputs identified in RESULT_TO_CODE.md. These tables are not pipeline inputs.

## Analysis workbook field inventory

| Field group | Current column(s) | Analytical role |
| --- | --- | --- |
| Identifiers | participant_id_norm hippo_id_norm hippo_id uid | Normalized participant and event identifiers, source event identifier, and participant–processing-segment unit. |
| Sequence links | hippo_prev_global hippo_next_global | Preceding and following event-states within each participant narrative; 251 next-event adjacencies. |
| Annotation metadata | hippo_weight_tag | Annotation tag retained as metadata; not an active MCA variable. |
| Theme-assignment count | n_theme_assignments_event | Number of theme assignments associated with each event-state. |
| Excerpt count | n_excerpts_event | Number of excerpts assigned to each event-state; sums to 442. |

Table S1A lists the 11 active MCA variables. This inventory lists the nine additional fields in Sheet1 of the original analysis workbook. MCA coordinates and macro-cluster or basin-state memberships are generated outputs, not additional columns in this input. This identified 20-column analysis workbook is distinct from the de-identified, 19-field public input and data dictionary described in Supplement S7.

## Corpus unit inventory

| Element | n | Role |
| --- | --- | --- |
| Participants | 5 | Participant-global narrative sequences |
| Excerpt units | 442 | Source units mapped to event-states |
| Processing segments | 21 | LLM-inference provenance units |
| Event-states | 256 | Rows in the current MCA-ready table |
| Within-segment boundaries | 235 | Adjacent events within the same processing segment |
| Between-segment links | 16 | Successive processing segments within each participant narrative |
| Participant-global adjacencies | 251 | Next-event links used for transition analyses |

## Extended macro-cluster comparisons

| MCA dimensions | Macro-clusters | Silhouette | Bootstrap ARI | Minimum events per macro-cluster | Minimum participants | Maximum participant share | Mean participant-exclusion ARI | Mean held-out corrected R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 0.498 | 0.708 | 76 | 5 | 0.342 | 0.807 | 0.053 |
| 3 | 3 | 0.414 | 0.776 | 45 | 5 | 0.347 | 0.705 | 0.057 |
| 2 | 4 | 0.498 | 0.714 | 1 | 1 | 1.000 | 0.607 | 0.050 |
| 3 | 4 | 0.428 | 0.681 | 1 | 1 | 1.000 | 0.818 | 0.056 |
| 4 | 4 | 0.349 | 0.735 | 17 | 4 | 0.471 | 0.720 | 0.054 |
| 5 | 4 | 0.296 | 0.669 | 17 | 4 | 0.471 | 0.636 | 0.055 |
| 6 | 4 | 0.259 | 0.589 | 17 | 4 | 0.471 | 0.606 | 0.055 |
| 7 | 4 | 0.219 | 0.545 | 17 | 4 | 0.471 | 0.598 | 0.056 |
| 10 | 3 | 0.252 | 0.520 | 17 | 4 | 0.471 | 0.694 | 0.053 |
| 10 | 4 | 0.165 | 0.474 | 17 | 4 | 0.471 | 0.592 | 0.055 |
| 13 | 2 | 0.185 | 0.543 | 84 | 5 | 0.345 | 0.908 | 0.049 |
| 13 | 4 | 0.122 | 0.470 | 30 | 5 | 0.343 | 0.503 | 0.051 |

ARI is the adjusted Rand index. These comparisons cover selected dimensionalities and macro-cluster counts from the two- through thirteen-dimensional, two- through six-cluster grid. Eligibility required at least 0.05 of the event-states and at least two participants in each macro-cluster. The two- and three-dimensional four-cluster partitions failed both criteria. Selection considered geometric separation, event-resampling stability, participant support, participant-exclusion stability and held-out construct alignment. Transition and successor outcomes were excluded. The neighboring four- and five-dimensional four-cluster partitions had matched-profile mean Pearson r = 0.954 and assignment ARI = 0.673.

Among eligible four-cluster partitions, the four-dimensional representation had the greatest silhouette and bootstrap ARI. It also had greater mean participant-exclusion ARI and similar held-out corrected R² relative to the ten- and thirteen-dimensional four-cluster partitions. Higher-dimensional representations retained more overall distance information (Table S2B). The thirteen-dimensional two-cluster partition had high participant-exclusion agreement. These descriptive comparisons support the selected four-dimensional, four-cluster representation for the intended macro-cluster analysis. They do not test basin fitting in ten or thirteen dimensions.

## Basin construction settings

| Setting | Value |
| --- | --- |
| Fitting spaces | 4-D macro-geometry; 2-D Dim0 × Dim3 plane |
| MCA scaling | Original raw scaling; no whitening |
| Assignment unit | Identical 11-feature profiles kept indivisible and event weighted |
| Candidate basin counts | 5–20 |
| Lower and upper event bounds | Lower bounds of 8, 9, or 10; upper bounds of every integer from 25 through 50 |
| Outgoing eligibility | At least 8 original outgoing adjacencies per basin-state |
| Screen starts | 10 independent starts plus preceding-cap warm starts |
| Full-sample refinement | 100 independent starts plus retained screen solution |
| Solver / update | Binary mixed-integer fixed-centroid assignment; weighted-centroid updates |
| Optimization limit | Fixed-centroid relative gap 10⁻⁹; 30 s solver cap; up to 200 iterations |
| Final shortlist | Up to three 4-D and two 2-D solutions; diversity ARI < 0.95 |

Basin assignments use constrained clustering. Gaussian smoothing in a display does not define membership. Fixed-centroid solver tolerances do not certify a global clustering optimum; finite-start optimization and count constraints contribute to uncertainty.

## Conditioned routing complete

| Basin solution | Comparison | Feature family | Retained conditions | Observed cosine similarity | Null mean | Lower-tail p | BH q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin | Full-system | Primary context | 4 | 0.127 | 0.292 | < 0.001 | < 0.001 |
| 4-D 10-Basin | Full-system | Mechanism presence | 6 | 0.295 | 0.413 | < 0.001 | < 0.001 |
| 4-D 10-Basin | Full-system | Meta-theme | 11 | 0.153 | 0.175 | 0.031 | 0.031 |
| 4-D 10-Basin | Supported-source | Primary context | 4 | 0.511 | 0.501 | 0.547 | 0.547 |
| 4-D 10-Basin | Supported-source | Mechanism presence | 6 | 0.537 | 0.596 | 0.134 | 0.403 |
| 4-D 10-Basin | Supported-source | Meta-theme | 11 | 0.433 | 0.440 | 0.436 | 0.547 |
| 4-D 9-Basin | Full-system | Primary context | 4 | 0.175 | 0.346 | < 0.001 | < 0.001 |
| 4-D 9-Basin | Full-system | Mechanism presence | 6 | 0.392 | 0.460 | 0.008 | 0.008 |
| 4-D 9-Basin | Full-system | Meta-theme | 11 | 0.181 | 0.216 | 0.007 | 0.008 |
| 4-D 9-Basin | Supported-source | Primary context | 4 | 0.672 | 0.549 | 0.956 | 0.956 |
| 4-D 9-Basin | Supported-source | Mechanism presence | 6 | 0.581 | 0.635 | 0.151 | 0.452 |
| 4-D 9-Basin | Supported-source | Meta-theme | 11 | 0.533 | 0.475 | 0.921 | 0.956 |
| 4-D 6-Basin | Full-system | Primary context | 4 | 0.223 | 0.531 | < 0.001 | < 0.001 |
| 4-D 6-Basin | Full-system | Mechanism presence | 6 | 0.463 | 0.622 | < 0.001 | < 0.001 |
| 4-D 6-Basin | Full-system | Meta-theme | 11 | 0.284 | 0.371 | < 0.001 | < 0.001 |
| 4-D 6-Basin | Supported-source | Primary context | 4 | 0.754 | 0.641 | 0.973 | 0.973 |
| 4-D 6-Basin | Supported-source | Mechanism presence | 6 | 0.734 | 0.715 | 0.667 | 0.973 |
| 4-D 6-Basin | Supported-source | Meta-theme | 11 | 0.594 | 0.532 | 0.960 | 0.973 |
| 2-D 9-Basin | Full-system | Primary context | 4 | 0.182 | 0.323 | < 0.001 | < 0.001 |
| 2-D 9-Basin | Full-system | Mechanism presence | 6 | 0.320 | 0.440 | < 0.001 | < 0.001 |
| 2-D 9-Basin | Full-system | Meta-theme | 11 | 0.188 | 0.201 | 0.195 | 0.195 |
| 2-D 9-Basin | Supported-source | Primary context | 4 | 0.461 | 0.528 | 0.179 | 0.269 |
| 2-D 9-Basin | Supported-source | Mechanism presence | 6 | 0.466 | 0.618 | 0.001 | 0.003 |
| 2-D 9-Basin | Supported-source | Meta-theme | 11 | 0.476 | 0.460 | 0.663 | 0.663 |
| 2-D 8-Basin | Full-system | Primary context | 4 | 0.250 | 0.375 | < 0.001 | < 0.001 |
| 2-D 8-Basin | Full-system | Mechanism presence | 6 | 0.341 | 0.488 | < 0.001 | < 0.001 |
| 2-D 8-Basin | Full-system | Meta-theme | 11 | 0.203 | 0.225 | 0.052 | 0.052 |
| 2-D 8-Basin | Supported-source | Primary context | 4 | 0.579 | 0.500 | 0.904 | 0.969 |
| 2-D 8-Basin | Supported-source | Mechanism presence | 6 | 0.616 | 0.599 | 0.661 | 0.969 |
| 2-D 8-Basin | Supported-source | Meta-theme | 11 | 0.460 | 0.397 | 0.969 | 0.969 |

## Condition support summary

| Basin solution / family | Conditions | Source transitions per condition | Active source basin-states per condition | Source basin-states with ≥3 transitions | Nonzero routing cells per condition |
| --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin / Primary context | 4 | 14–143 | 1–7 | 1–6 | 7–43 |
| 4-D 10-Basin / Mechanism presence | 6 | 13–219 | 4–10 | 3–10 | 12–72 |
| 4-D 10-Basin / Meta-theme | 11 | 10–46 | 5–10 | 1–6 | 9–29 |
| 2-D 9-Basin / Primary context | 4 | 14–143 | 2–6 | 1–5 | 9–38 |
| 2-D 9-Basin / Mechanism presence | 6 | 13–219 | 3–9 | 3–8 | 10–63 |
| 2-D 9-Basin / Meta-theme | 11 | 10–46 | 5–9 | 1–6 | 10–29 |

Conditions required at least 10 source transitions. Supported-source comparisons required at least three outgoing transitions from the same source basin-state under each compared condition. BH q values adjust across the three feature families separately for full-system and supported-source comparisons within each solution. These local adjustments do not control a study-wide family across solutions and participant omissions. Support masks were recomputed within each condition-label shuffle.
Context and meta-theme labels partition the 251 source transitions before filtering. Mechanism-presence groups can overlap. This table reports support ranges across retained conditions in the main and 2-D nine-basin solutions. In the main solution, only three of six context-condition pairs have any shared source satisfying the support rule; 15/15 mechanism pairs and 35/55 meta-theme pairs contribute. The significant supported-source mechanism result in the 2-D nine-basin solution is nonsignificant after omitting P01 (q = 0.076) or P05 (q = 0.157).
Condition-specific counts, routing matrices, and support masks are retained in the numerical outputs. Global condition-label permutations do not preserve participant-specific condition frequencies. Omission analyses reapply eligibility rules, so the retained conditions can change.

## Successor discount geometry sweep

| Basin solution | γ | 4-D r / nominal p | 10-D r / nominal p | 13-D r / nominal p |
| --- | --- | --- | --- | --- |
| 4-D 10-Basin | 0.70 | 0.409 / 0.097 | 0.406 / 0.122 | 0.405 / 0.125 |
| 4-D 10-Basin | 0.80 | 0.418 / 0.088 | 0.419 / 0.110 | 0.418 / 0.112 |
| 4-D 10-Basin | 0.90 | 0.427 / 0.080 | 0.432 / 0.100 | 0.431 / 0.101 |
| 4-D 10-Basin | 0.95 | 0.431 / 0.076 | 0.438 / 0.094 | 0.438 / 0.095 |
| 4-D 9-Basin | 0.70 | 0.393 / 0.136 | 0.376 / 0.178 | 0.374 / 0.182 |
| 4-D 9-Basin | 0.80 | 0.404 / 0.122 | 0.389 / 0.159 | 0.387 / 0.163 |
| 4-D 9-Basin | 0.90 | 0.415 / 0.110 | 0.401 / 0.143 | 0.400 / 0.146 |
| 4-D 9-Basin | 0.95 | 0.420 / 0.105 | 0.408 / 0.135 | 0.406 / 0.138 |
| 4-D 6-Basin | 0.70 | 0.822 / 0.009 | 0.792 / 0.012 | 0.792 / 0.012 |
| 4-D 6-Basin | 0.80 | 0.827 / 0.008 | 0.797 / 0.012 | 0.797 / 0.012 |
| 4-D 6-Basin | 0.90 | 0.832 / 0.005 | 0.803 / 0.011 | 0.802 / 0.011 |
| 4-D 6-Basin | 0.95 | 0.834 / 0.005 | 0.805 / 0.011 | 0.805 / 0.011 |
| 2-D 9-Basin | 0.70 | 0.453 / 0.036 | 0.452 / 0.044 | 0.425 / 0.066 |
| 2-D 9-Basin | 0.80 | 0.454 / 0.035 | 0.453 / 0.043 | 0.425 / 0.066 |
| 2-D 9-Basin | 0.90 | 0.455 / 0.035 | 0.453 / 0.042 | 0.425 / 0.066 |
| 2-D 9-Basin | 0.95 | 0.455 / 0.035 | 0.454 / 0.041 | 0.425 / 0.066 |
| 2-D 8-Basin | 0.70 | 0.678 / 0.001 | 0.655 / 0.003 | 0.648 / 0.003 |
| 2-D 8-Basin | 0.80 | 0.681 / 0.001 | 0.658 / 0.002 | 0.652 / 0.003 |
| 2-D 8-Basin | 0.90 | 0.684 / 0.001 | 0.662 / 0.002 | 0.655 / 0.003 |
| 2-D 8-Basin | 0.95 | 0.685 / 0.001 | 0.663 / 0.002 | 0.657 / 0.003 |

## Procrustes all solutions

| Basin solution | Successor PC1 fraction | Successor PC2 fraction | Cumulative fraction | Procrustes disparity | Lower-tail p | Permutations |
| --- | --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin | 0.200 | 0.161 | 0.361 | 0.992 | 0.193 | 10,000 |
| 4-D 9-Basin | 0.212 | 0.202 | 0.414 | 0.833 | 0.112 | 10,000 |
| 4-D 6-Basin | 0.277 | 0.256 | 0.533 | 0.398 | 0.066 | 10,000 |
| 2-D 9-Basin | 0.206 | 0.185 | 0.391 | 0.943 | 0.209 | 10,000 |
| 2-D 8-Basin | 0.232 | 0.182 | 0.415 | 0.617 | 0.044 | 10,000 |

## Participant exclusion complete

| Basin solution / transition unit | Participant omitted | Events retained | Transitions retained | Fixed-centroid r / p | Recomputed-centroid r / p |
| --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin / event | None | 256 | 251 | 0.427 / 0.080 | 0.427 / 0.080 |
| 4-D 10-Basin / event | P01 | 227 | 223 | 0.444 / 0.064 | 0.398 / 0.116 |
| 4-D 10-Basin / event | P02 | 200 | 196 | 0.404 / 0.092 | 0.583 / 0.005 |
| 4-D 10-Basin / event | P03 | 190 | 186 | 0.246 / 0.350 | 0.277 / 0.279 |
| 4-D 10-Basin / event | P04 | 222 | 218 | 0.401 / 0.081 | 0.417 / 0.069 |
| 4-D 10-Basin / event | P05 | 185 | 181 | 0.492 / 0.030 | 0.327 / 0.198 |
| 4-D 10-Basin / visit | None | 256 | 199 | 0.585 / < 0.001 | 0.585 / < 0.001 |
| 4-D 10-Basin / visit | P01 | 227 | 176 | 0.570 / 0.001 | 0.584 / < 0.001 |
| 4-D 10-Basin / visit | P02 | 200 | 156 | 0.544 / < 0.001 | 0.527 / 0.002 |
| 4-D 10-Basin / visit | P03 | 190 | 148 | 0.430 / 0.036 | 0.428 / 0.036 |
| 4-D 10-Basin / visit | P04 | 222 | 173 | 0.615 / < 0.001 | 0.622 / < 0.001 |
| 4-D 10-Basin / visit | P05 | 185 | 143 | 0.550 / 0.001 | 0.524 / 0.006 |
| 2-D 9-Basin / event | None | 256 | 251 | 0.455 / 0.035 | 0.455 / 0.035 |
| 2-D 9-Basin / event | P01 | 227 | 223 | 0.309 / 0.198 | 0.294 / 0.227 |
| 2-D 9-Basin / event | P02 | 200 | 196 | 0.530 / 0.010 | 0.544 / 0.006 |
| 2-D 9-Basin / event | P03 | 190 | 186 | 0.478 / 0.020 | 0.457 / 0.028 |
| 2-D 9-Basin / event | P04 | 222 | 218 | 0.296 / 0.101 | 0.295 / 0.100 |
| 2-D 9-Basin / event | P05 | 185 | 181 | 0.494 / 0.024 | 0.473 / 0.035 |
| 2-D 9-Basin / visit | None | 256 | 202 | 0.552 / 0.002 | 0.552 / 0.002 |
| 2-D 9-Basin / visit | P01 | 227 | 180 | 0.533 / 0.002 | 0.542 / 0.002 |
| 2-D 9-Basin / visit | P02 | 200 | 156 | 0.551 / < 0.001 | 0.537 / 0.001 |
| 2-D 9-Basin / visit | P03 | 190 | 154 | 0.410 / 0.019 | 0.393 / 0.025 |
| 2-D 9-Basin / visit | P04 | 222 | 176 | 0.541 / 0.002 | 0.533 / 0.003 |
| 2-D 9-Basin / visit | P05 | 185 | 142 | 0.514 / 0.004 | 0.527 / 0.003 |

## Transition units complete

| Basin solution | Transition unit | Space | Transitions | Self-transitions | Pearson r | Nominal p | Δr from event adjacency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin | Event | 4-D | 251 | 52 | 0.427 | 0.080 | 0.000 |
| 4-D 10-Basin | Basin visit | 4-D | 199 | 0 | 0.585 | < 0.001 | 0.158 |
| 4-D 9-Basin | Event | 4-D | 251 | 52 | 0.415 | 0.110 | 0.000 |
| 4-D 9-Basin | Basin visit | 4-D | 199 | 0 | 0.651 | < 0.001 | 0.236 |
| 4-D 6-Basin | Event | 4-D | 251 | 72 | 0.832 | 0.005 | 0.000 |
| 4-D 6-Basin | Basin visit | 4-D | 179 | 0 | 0.803 | 0.002 | -0.028 |
| 2-D 9-Basin | Event | 4-D | 251 | 49 | 0.455 | 0.035 | 0.000 |
| 2-D 9-Basin | Basin visit | 4-D | 202 | 0 | 0.552 | 0.002 | 0.097 |
| 2-D 8-Basin | Event | 4-D | 251 | 48 | 0.684 | 0.001 | 0.000 |
| 2-D 8-Basin | Basin visit | 4-D | 203 | 0 | 0.691 | < 0.001 | 0.007 |

Participant-exclusion rows rebuild P and M after each omission while retaining the original MCA coordinates and basin memberships. Fixed centroids and centroids recomputed from retained events isolate complementary influences. Both columns report r / nominal p at γ = 0.90 using 4-D centroid distances. They do not represent participant-specific systems or full-pipeline refits. “Events retained” continues to count event-states even in the visit rows.
Basin-visit adjacency collapses maximal consecutive runs within a basin-state. In the main solution, 256 event-states form 204 visits and 199 visit adjacencies; the 2-D nine-basin comparison forms 207 visits and 202 visit adjacencies. One discounted step becomes one transition between consecutive basin visits. Δr is descriptive and does not test the difference between correlations. All five omissions remain nominally significant for each selected solution under visit adjacency and both centroid definitions. Figure S6A–B displays the main and 2-D nine-basin full-sample comparisons using event-weighted centroids.

## Narrated reversed order complete

| Basin solution / order | Geometry | Pearson r | Label-permutation p | Shuffle-null mean r | Shuffle-null 2.5% | Shuffle-null 97.5% | Order-shuffle p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin / narrated | 4-D | 0.427 | 0.080 | 0.123 | -0.256 | 0.504 | 0.062 |
| 4-D 10-Basin / reversed | 4-D | 0.432 | 0.073 | 0.123 | -0.256 | 0.504 | 0.059 |
| 4-D 9-Basin / narrated | 4-D | 0.415 | 0.110 | 0.159 | -0.243 | 0.561 | 0.113 |
| 4-D 9-Basin / reversed | 4-D | 0.399 | 0.131 | 0.159 | -0.243 | 0.561 | 0.126 |
| 4-D 6-Basin / narrated | 4-D | 0.832 | 0.005 | 0.234 | -0.402 | 0.741 | 0.003 |
| 4-D 6-Basin / reversed | 4-D | 0.800 | 0.015 | 0.234 | -0.402 | 0.741 | 0.008 |
| 2-D 9-Basin / narrated | 4-D | 0.455 | 0.035 | 0.166 | -0.249 | 0.530 | 0.071 |
| 2-D 9-Basin / reversed | 4-D | 0.499 | 0.017 | 0.166 | -0.249 | 0.530 | 0.041 |
| 2-D 8-Basin / narrated | 4-D | 0.684 | 0.001 | 0.115 | -0.325 | 0.518 | 0.002 |
| 2-D 8-Basin / reversed | 4-D | 0.672 | 0.001 | 0.115 | -0.325 | 0.518 | 0.002 |

## Profile definitions complete

| Basin solution | Profile definition | Pearson r | Nominal p |
| --- | --- | --- | --- |
| 4-D 10-Basin | P | 0.516 | 0.028 |
| 4-D 10-Basin | I + 0.90P | 0.377 | 0.137 |
| 4-D 10-Basin | M | 0.427 | 0.080 |
| 4-D 10-Basin | M − I | 0.513 | 0.029 |
| 4-D 9-Basin | P | 0.539 | 0.008 |
| 4-D 9-Basin | I + 0.90P | 0.349 | 0.190 |
| 4-D 9-Basin | M | 0.415 | 0.110 |
| 4-D 9-Basin | M − I | 0.578 | 0.005 |
| 4-D 6-Basin | P | 0.834 | 0.008 |
| 4-D 6-Basin | I + 0.90P | 0.798 | 0.016 |
| 4-D 6-Basin | M | 0.832 | 0.005 |
| 4-D 6-Basin | M − I | 0.871 | 0.006 |
| 2-D 9-Basin | P | 0.408 | 0.068 |
| 2-D 9-Basin | I + 0.90P | 0.453 | 0.039 |
| 2-D 9-Basin | M | 0.455 | 0.035 |
| 2-D 9-Basin | M − I | 0.436 | 0.061 |
| 2-D 8-Basin | P | 0.749 | 0.002 |
| 2-D 8-Basin | I + 0.90P | 0.690 | < 0.001 |
| 2-D 8-Basin | M | 0.684 | 0.001 |
| 2-D 8-Basin | M − I | 0.741 | 0.001 |

Separate label-permutation and order-shuffle p values test different null hypotheses and do not test differences between correlations. Label tests are two-sided. Order tests are upper-tail and rebuild P and M after independently shuffling event order within each participant. The order shuffles allow endpoints to change, unlike the endpoint-fixed null in Tables S3D–F. The narrated and reversed order comparison uses 4-D distances. The main and 2-D nine-basin narrated-order tests are nonsignificant; the 4-D six-basin and 2-D eight-basin tests are nominally significant. Figure S6D displays the narrated and reversed correlations relative to the shuffle-null ranges.
The profile-definition comparison uses 4-D basin-centroid distances and Euclidean distances between the indicated profiles at γ = 0.90. Standard M includes current-state identity and expected discounted future occupancy; M − I removes identity. The P comparison is distinct from the cosine feature–routing test in Table S3B. Separate p values do not establish differences between profile correlations or the necessity of accumulated multistep structure. Figure S6C displays these profile comparisons across all five solutions.
Profile controls comprise the four listed profile definitions; no finite-horizon multiple-testing adjustment is applied.

## Basin recurrence

| Basin solution | Transition unit | Significant basin-state | Successor centrality | BH q | Minimum BH q |
| --- | --- | --- | --- | --- | --- |
| 4-D 10-Basin | Event | B9 | 0.754 | 0.015 | 0.015 |
| 4-D 10-Basin | Basin visit | B8 | 0.862 | 0.003 | 0.003 |
| 4-D 9-Basin | Event | None | — | — | 0.096 |
| 4-D 9-Basin | Basin visit | B2 | 1.996 | 0.019 | 0.019 |
| 4-D 6-Basin | Event | B6 | 0.828 | 0.011 | 0.011 |
| 4-D 6-Basin | Basin visit | B1 | 2.188 | 0.004 | 0.004 |
| 2-D 9-Basin | Event | None | — | — | 0.112 |
| 2-D 9-Basin | Basin visit | None | — | — | 0.355 |
| 2-D 8-Basin | Event | B8 | 0.812 | 0.016 | 0.016 |
| 2-D 8-Basin | Basin visit | None | — | — | 0.608 |

Successor centrality is the mean of a complete successor-matrix column, equally weighting starting basin-states; it is not a probability. The upper-tail null independently shuffles original event order within each participant. Visit nulls collapse consecutive same-basin runs after every shuffle. BH correction covers all basin-states separately within each basin solution and transition unit. “None” indicates no basin-state with q < 0.05. Minimum q is reported even when no basin-state survives. The main solution’s event-state adjacency result for B9 survives three of five omissions; its basin-visit result for B8 survives four. Basin identifiers are local to each solution.
