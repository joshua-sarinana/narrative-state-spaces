# Release revision

The 256-event scientific analysis code and source CSV are retained. The public master table now uses the existing neutral identifiers and contains 46 fields that can be reconstructed from the public workflow. Its coordinates, macro-cluster memberships and five basin assignments match the previous release references exactly.

The 66 separate reference CSVs are replaced by the master table and one quantitative-checks JSON containing 52 checks. The 32 saved analysis exports are omitted. Six duplicate codebook CSVs are omitted because the workbook contains all six sheets. The 25 publication supplementary CSVs remain represented by the manuscript's supplementary-table document. The 13 extended publication tables are preserved in EXTENDED_RESULTS.md, including their statistical notes. The ambiguous Model column in the extended macro-cluster comparison is split into MCA dimensions and Macro-clusters.

Full-precision data and numerical tolerances remain separate from publication rounding. The pipeline generates the detailed outputs. The compact verifier retains the full 54-axis coordinate check, event memberships, selected-solution checks, transition and successor matrices, principal results and specified sensitivity comparisons. It does not retain every exhaustive diagnostic comparison from the previous package.

The runner no longer requires reference files to execute calculations. The verifier checks the compact reference files. A new analysis stage rebuilds the master table without reading either reference file. The analysis-only verifier does not require figure rendering.

## Supplement S7 alignment

Replace the two paragraphs beginning “Publication tables are supplied as CSV files” and “Extended tables cited above are in tables/extended/” with the following text. The surrounding S7 paragraphs can remain.

The supplementary-table document provides the publication tables. The public reproducibility package consolidates the complete or relocated tables in documentation/EXTENDED_RESULTS.md, retaining their reported rounding and statistical notes. This file includes the original analysis-workbook field inventory, corpus unit inventory, extended macro-cluster comparisons, basin-construction settings, complete conditioned-routing results and condition support, the successor discount and geometry sweep, all five Procrustes comparisons, participant-exclusion estimates, transition-unit comparisons, complete order-control and profile-definition results, and exploratory basin recurrence. The public 19-field input has a separate dictionary at data/DATA_DICTIONARY.csv.

The public master table integrates the 256 event-states with their feature variables, neutral identifiers, narrative order, Dim0–Dim12 coordinates, macro-cluster membership and all five basin-state assignments. It is rebuilt from the public input and computed analysis outputs. Its field definitions and source mappings are documented in documentation/MASTER_TABLE.md and provenance/MASTER_TABLE.json. The stored master table and reference_outputs/quantitative_checks.json provide compact verification targets. Detailed analysis outputs are generated under outputs/, with their producers identified in documentation/RESULT_TO_CODE.md.

## Verification scope of this revision

The validation report documents the checks executed for this revision. Repackaging and comparison with inherited references do not constitute a fresh full run of basin selection and every downstream permutation analysis.
