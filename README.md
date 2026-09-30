# Event Memory and Narrative Dynamics

Public computational release for *Event Memory and Narrative Dynamics: A Large Language Model Framework for Multidimensional State Spaces* by Joshua Sariñana. This package is the 256-event compact release, revision 1.

## Start here

`data/event_level_analysis_input.csv` is the source table for the numerical workflow. It contains 256 event-states, 19 fields, five participants, 21 processing segments and 251 directed narrative adjacencies. Excerpt counts sum to 442. The scripts construct the representational spaces and memberships, then calculate transitions, successor profiles, statistical comparisons and figure panels.

The de-identified master table in `reference_outputs/master_regime_table_eventlevel_REFIT.csv` integrates the 19 input fields with Dim0–Dim12 coordinates, macro-cluster membership, narrative-order descriptors and all five retained basin assignments. It is a derived inspection and verification table. The pipeline rebuilds it from the source input and computed outputs.

## Run

Use Python 3.12 and the pinned dependencies in a dedicated environment. Run these commands from the extracted package directory.

```bash
python -m pip install -r requirements.txt
python code/run_pipeline.py --list
python code/run_pipeline.py --workers 4 --group analysis
python code/verify_reproduction.py --group analysis
```

To include computational figure panels, omit `--group analysis` from both commands. The pipeline reuses completed stages only when their recorded output hashes and source configuration match.

For a short input and boundary check, run `python code/run_pipeline.py --through 01_boundaries --group analysis`, followed by `python code/verify_reproduction.py --completed-only --group analysis`. This is a partial check. The full workflow includes the basin screen, omission fits, 10,000-draw downstream analyses and 1,000,000-draw sequence analyses. Full execution can be lengthy.

## Files

| Location | Purpose |
| --- | --- |
| `data/` | Event-level source input and data dictionary |
| `code/` | Numerical methods, orchestration, master-table builder, verification and figure code |
| `provenance/` | Execution order, specified omission samples, MCA sign convention, master-table provenance and file integrity records |
| `reference_outputs/` | Two verification files: the integrated master table and expected numerical checks |
| `documentation/` | Annotation codebook, aggregation and field definitions, result-to-code map, verification guidance and consolidated extended results |
| `prompts/` | Model-independent constraints for prospective annotation |
| `licenses/` | Software and data/documentation license texts |

`outputs/` is created during execution. All intermediate analysis tables, perturbation diagnostics and figure products are written there. No saved output table is used to fill a computed result. The runner validates the code, input and configuration independently of the reference files; the verifier checks reference-file integrity when comparing results.

## Representation and sequence

The primary geometry is four-dimensional. Ten- and thirteen-dimensional geometries provide reference comparisons. The retained solutions are 4-D 10-Basin, 4-D 9-Basin, 4-D 6-Basin, 2-D 9-Basin and 2-D 8-Basin. The 2-D solutions are fitted in Dim0 × Dim3. The fitting space, distance geometry and displayed dimensions have separate roles.

Use `hippo_prev_global` and `hippo_next_global` to reconstruct narrative order. Sorting event identifiers is not an ordering rule. Links include 16 between-segment joins. Repeated feature profiles remain separate event-states and are kept indivisible during basin fitting.

## Scope and documentation

Original participant and event identifiers, narrative text, excerpts, specific-anchor text and private mappings are excluded. Neutral identifiers preserve joins across the released files. Numerical reproduction starts from frozen annotations. The codebook and prospective prompt constraints document the annotation system; they do not reproduce historical model conversations.

`documentation/RESULT_TO_CODE.md` maps manuscript results to their producers. `documentation/MASTER_TABLE.md` explains the integrated table. `documentation/EXTENDED_RESULTS.md` retains the 13 extended publication tables cited by the supplementary methods. `documentation/VERIFICATION.md` defines comparison coverage and distinguishes partial checks from full reproduction.

Figure scripts export individual panels and supporting data. Manual composition and the conceptual Figure 1 are outside numerical reproduction. Code is MIT licensed; data and documentation are CC BY 4.0. Citation and availability metadata remain pending deposition.
