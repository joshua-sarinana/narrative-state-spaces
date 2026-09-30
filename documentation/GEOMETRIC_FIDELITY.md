# Geometric fidelity

Each prefix of 1–54 MCA dimensions is compared with the complete 54-dimensional solution. Pearson distance correlation, stress, inertia and row cos² use unrounded coordinates and distances.

## Neighborhood definition

For each event, its own index is removed explicitly before any neighbor selection. Other events with identical feature profiles remain eligible neighbors; they are distinct analytical units.

Euclidean distances are computed from coordinate differences. Sorted distances are partitioned into numerical-tie blocks. Each block includes distances no more than 1e-10 above the block's smallest distance, with zero relative tolerance. A chain of pairwise close distances does not expand a block beyond this width. Within a block, neighbors are ordered lexicographically by the stable pair `(participant_id_norm, hippo_id_norm)`. Thus permuting table rows does not change membership in a neighborhood. Identifier order resolves an otherwise ambiguous tie; it has no interpretation as additional geometric separation.

The same rule applies in both the complete and truncated spaces. Exactly k other events define each neighborhood, for k = 5, 10 and 20. Recall is the intersection size divided by k. Jaccard is the intersection size divided by the union size. Trustworthiness uses the standard rank penalty for neighbors entering the truncated neighborhood from outside the complete-space neighborhood. Its full-space ranks and truncated neighbors follow the same tie convention.

Spearman distance correspondence assigns average ranks within numerical-tie blocks of condensed pairwise distances. It does not break these correlation ties using event identifiers. Continuous distance statistics and the coordinates themselves are not rounded or altered.

## Reproducibility

The method identifier and tolerance are in `provenance/GEOMETRIC_FIDELITY_METHOD.json` and the MCA run metadata. The 1e-10 tie threshold defines how distances are ranked; it is distinct from the verifier's unchanged 1e-10 absolute output-comparison tolerance. No relaxed output allowance applies to geometric fidelity.

Figure S1B uses Pearson distance correlation, trustworthiness at k = 10, and neighbor recall at k = 10. The full diagnostic table also reports Spearman correspondence and Jaccard. These descriptive diagnostics do not feed into parallel-analysis retention, projection selection, macro-cluster selection or basin fitting. Projection-selection diagnostics are calculated by their separately specified procedure.

Reference fidelity values are generated from the saved full coordinate export and checked against a new MCA fit of the frozen annotations. Analysis stages independently calculate their outputs; they never read reference tables. Regression tests cover explicit self-exclusion with duplicates, stable event-key ties, row permutation, complete-space identity, and agreement with standard formulas when distances are not tied.

Run the regression checks from the repository root with:

```bash
python -m unittest discover -s code/mca -p test_fidelity.py
```
