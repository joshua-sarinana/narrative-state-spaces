# HPC-informed event construction: minimum prompt constraints

## Task

Group the fixed, ordered thematic excerpts into event states and assign the HPC-informed categorical fields defined in `Annotation_Codebook.xlsx`. An event state is a theory-constrained textual representation of a stable organizing configuration; it is not a measured hippocampal state or a demonstrated dynamical attractor.

## Required inputs

- the fixed excerpt table produced by thematic annotation;
- the mechanism, context, anchor, representation, orientation, perspective, reference-frame, landmark, and boundary definitions from `Annotation_Codebook.xlsx`;
- the required event and excerpt-to-event output schemas.

## Constraints that must appear in the complete prompt

1. Do not edit, paraphrase, reorder, split, or merge the input excerpts. Each excerpt must map to exactly one event, and each event must contain one or more consecutive excerpts.
2. Preserve an event while one dominant configuration adequately characterizes the combined span. A new sentence, thematic label, example, emotional shift, or secondary mechanism is insufficient by itself to create a boundary.
3. Retain a boundary when adjacent material cannot be represented adequately by one organizing configuration because the dominant primary mechanism, primary context, specific organizing anchor, or anchor category changes.
4. A representational-domain change may contribute only when the new domain is explicitly supported and dominates at least one complete excerpt. It should not create a boundary from a weak or incidental change.
5. Orientation, perspective, reference frame, and landmark describe an event but do not independently establish a boundary.
6. Meta-theme and sentiment are joined after the event partition is fixed. They characterize events and must not determine event boundaries.
7. Mechanism describes the operation organizing the span; context describes its organizing environment rather than its topic. Specific anchor identifies the fine-grained organizing referent. A mentioned person or place does not automatically replace the organizing anchor.
8. Perspective identifies the focal experiencer or observer. Reference frame identifies organization relative to the experiencer or environment. Do not infer either from emotional tone alone.
9. Ground every feature and boundary in the supplied excerpts. Do not invent referents, actions, settings, temporal relations, or unexpressed internal states.
10. Use exact categorical spellings from the supplied codebook. Apply its explicit `none` and missing-value conventions rather than creating new null categories.
11. Keep participant, processing-segment, event, and excerpt identities distinct. Event identifiers must be unique and sortable within each segment.
12. Do not use MCA coordinates, cluster or basin membership, transition counts, successor results, or any other downstream analysis to construct or revise events.

## Minimum outputs

Produce two linked tables.

### Event table

`participant_code | segment_id | event_id | primary_hpc_mechanism | secondary_hpc_mechanism | primary_context | secondary_context | specific_anchor_local | anchor_type | representational_domain | orientation | perspective | reference_frame | landmark | event_rationale | uncertainty_flag`

### Excerpt-to-event map

`participant_code | segment_id | excerpt_id | event_id | excerpt_order`

Specific-anchor text and evidence rationales may contain identifying narrative information and should remain in the protected annotation environment. A disclosure-reviewed event table can later replace them with sufficient statistics or omit them.

## Required validation

- every input excerpt maps once and only once;
- each event contains a consecutive excerpt span;
- event and excerpt order are preserved;
- every boundary satisfies at least one permitted boundary rule;
- descriptor-only changes do not create boundaries;
- all categorical values match the codebook;
- all assignments and boundaries remain traceable to source evidence;
- no downstream quantitative result influenced annotation.
