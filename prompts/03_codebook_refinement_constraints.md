# Codebook refinement: minimum prompt constraints

## Scope

This optional template supports adaptation of the framework to a new corpus. It must be used only on designated development material before the production annotation pass. It is not part of reproduction of the released study and must not modify the released `Annotation_Codebook.xlsx`.

## Required inputs

- the current versioned codebook;
- a designated development subset and its provisional annotations;
- the field definitions and boundary roles that must remain fixed;
- a required change-ledger schema.

## Constraints that must appear in the complete prompt

1. Treat every output as a proposal requiring human review. Never modify the codebook or existing annotations automatically.
2. Use semantic coverage, definitional clarity, and observed ambiguity in the development material as evidence. Do not use MCA geometry, cluster or basin structure, transition statistics, successor results, or desired hypotheses.
3. Prefer an existing category when its definition adequately covers the evidence. Do not add a category merely because wording differs.
4. Propose a new category only when recurrent evidence cannot be represented without systematic information loss. Distinguish a genuinely new construct from a synonym, subtype, or corpus-specific example.
5. A proposed merge must identify the conceptual overlap and confirm that the merged definition preserves distinctions required by the analysis.
6. A proposed split must state the discriminating rule and show that independent annotators could apply it from textual evidence.
7. Preserve the roles of boundary-defining fields and descriptor-only fields. Codebook refinement cannot convert a descriptor into a boundary criterion without a separately justified methodological revision.
8. Do not include participant names, quotations, autobiographical examples, or other potentially identifying material in public definitions.
9. Preserve a backward mapping from every retired or changed label to the preceding version. Flag one-to-many or ambiguous mappings for human adjudication.
10. Freeze and version the approved codebook before production annotation. Apply the same version across all participants in the production corpus.

## Minimum change-ledger output

`proposal_id | field | current_category | proposed_category | change_type | proposed_definition | inclusion_rule | exclusion_rule | evidence_summary | backward_mapping | affected_records_estimate | reviewer_decision | reviewer_rationale | target_version`

Allowed `change_type` values are `add`, `merge`, `split`, `rename`, `clarify`, and `retire`.

## Required validation

- every proposal has a semantic rationale and supporting development evidence;
- no proposal is based on downstream analytical results;
- no identifying text appears in public definitions;
- backward mappings are complete;
- rejected proposals remain recorded;
- the approved version is frozen before production annotation begins.
