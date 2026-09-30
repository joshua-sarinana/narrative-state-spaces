# Thematic annotation: minimum prompt constraints

## Task

Construct an ordered excerpt backbone from one interview processing segment and assign evidence-grounded thematic and sentiment fields using the supplied codebook. The output will become the fixed input to HPC-informed event construction.

## Required inputs

- one processing segment with stable participant/interviewer and source-turn identifiers;
- a non-identifying participant code and segment identifier;
- the thematic-domain and sentiment definitions from `Annotation_Codebook.xlsx`;
- the required structured-output schema.

## Constraints that must appear in the complete prompt

1. Analyze interviewee speech only. Interviewer speech may supply elicitation context but is neither coded evidence nor an analytical excerpt.
2. Preserve source order and verbatim wording. Do not paraphrase, reorder, complete, or invent speech.
3. An excerpt is a contiguous span expressing an independently interpretable proposition or a coherent set of related propositions. Nonpropositional speech need not form a separate excerpt.
4. Preserve one meaning-bearing unit across explanations, examples, qualifications, and shifts in emphasis when they serve the same proposition. Sentence boundaries and the availability of multiple labels do not independently require a split.
5. Fix excerpt spans and identifiers before interpreting themes. Thematic interpretation must not reshape the excerpt backbone.
6. Assign one shared meta-theme from the supplied codebook. More specific superordinate and sub-theme labels may be created for the new corpus, but they must remain linked to the shared domain and source excerpt.
7. Mark thematic meaning as `Explicit` only when directly stated. Use `Implicit` only when supported by the excerpt's propositions or semantic relationships. Do not infer unexpressed experiences, diagnoses, intentions, or mental states.
8. Assign sentiment as `Positive`, `Negative`, `Mixed`, or `Neutral`. `Mixed` requires salient positive and negative content without clear predominance. `Neutral` is informational or affectively unmarked content. Judge the expressed stance, not the annotator's evaluation of the topic.
9. Use exact categorical spellings from the supplied codebook. Do not create substitute labels for an existing category.
10. Flag insufficient or ambiguous evidence explicitly. Do not resolve uncertainty by inventing detail.

## Minimum output record

Each row must contain:

`participant_code | segment_id | excerpt_id | source_turn_start | source_turn_end | excerpt_text | superordinate_theme | sub_theme | meta_theme | thematic_type | sentiment | evidence_rationale | uncertainty_flag`

Excerpt identifiers must be unique and sortable in source order. Every coded assignment must retain an explicit link to its supporting excerpt.

## Required validation

- all excerpt text is verbatim and contiguous;
- excerpt order matches source order;
- participant and interviewer speech are not conflated;
- identifiers are unique;
- categorical values match the codebook;
- every thematic and sentiment assignment is traceable to evidence;
- no downstream geometry, cluster, transition, or successor result influenced annotation.
