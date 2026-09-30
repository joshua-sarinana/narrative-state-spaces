# Reproduction and verification

`provenance/RUN_CONFIG.json` is the run-order authority. Each stage writes computed outputs. The separate verifier reads `reference_outputs/quantitative_checks.json` and compares those outputs with `reference_outputs/`.

Identifiers, category labels, matrix labels, selections and counts must match exactly. Floating-point comparisons default to absolute tolerance 1e-10 and zero relative tolerance. Explicit field-specific allowances are declared in `reference_outputs/quantitative_checks.json` and explained in `documentation/NUMERICAL_TOLERANCES.md`. The verifier preserves both the strict result and acceptance under a declared allowance. It never changes a tolerance automatically, substitutes reference values, or treats missing results as a pass. Use `--strict-numerics` for the original uniform numerical comparison.

MCA axes are sign-indeterminate. The fixed orientation file records one nonzero event coordinate and its intended sign for each axis. Applying these sign conventions preserves geometry; it does not supply coordinate magnitudes. Refitted omission geometries are aligned using the documented fitting procedure.

Floating-point distance cancellation and tied neighbor ranks can change some descriptive diagnostics even when coordinates agree to 1e-10. The declared allowances apply to specified clustering scores and projection trustworthiness fields, with unchanged selections required. Geometric-fidelity diagnostics explicitly exclude self-neighbors and use the fixed tie rule in `GEOMETRIC_FIDELITY.md`; their original 1e-10 tolerance remains unchanged. See `NUMERICAL_TOLERANCES.md`.

The full basin screen and perturbation analyses must finish before end-to-end reproduction can be claimed. Downstream checks conditional on released memberships are a different verification scope. A stage log, build check, or a successful partial run is not a complete-run certificate.

The public input intentionally omits original narratives. This release permits reproduction of the quantitative analysis from frozen annotations; it cannot independently validate every source-text interpretation or regenerate historical LLM responses.

The current source implements 5,000 within-participant permutations for full-sample MCA and for each participant omission, the declared basin fitting budgets, 10,000 downstream permutations, and 1,000,000 endpoint-fixed permutations for each sequence-analysis package. Seeds and test families remain in the analysis modules and their protocols. The event-distance test is a separate primary comparison; the field/macro-to-next-basin package applies Holm across its six specified tests.

The reference directory contains two files: the public master table and the quantitative-checks JSON. The latter contains comparison definitions and selected expected values, including the full 54-axis coordinate target. The master table supplies the canonical macro-cluster and five basin memberships, along with the integrated-table check. All existing strict-coordinate prerequisites and declared metric tolerances are retained.

The exhaustive basin-screen table is checked by requested, fitted and eligible configuration counts, together with the retained candidate and final-selection tables. Detailed recurrence, condition-support and exhaustive projection diagnostics remain generated outputs. They are not all stored as reference targets in this compact release. The supplied reference targets are inherited numerical benchmarks; repackaging them does not constitute a fresh complete run.

Use `python code/verify_reproduction.py --group analysis` after `python code/run_pipeline.py --group analysis` to check the numerical workflow without requiring figure rendering. The default verifier requires all configured stages. `--completed-only` remains a partial diagnostic and never reports complete reproduction.
