# Event construction and event-level aggregation

The annotation codebook and constraint templates describe the thematic and event-construction stages. The frozen public event table is the starting point for quantitative reproduction.

Thematic annotation supplied an ordered excerpt backbone. Event construction grouped consecutive excerpts within each participant and processing segment. A change in the dominant mechanism, primary context, specific organizing anchor or anchor category could establish a boundary. A change in representational domain could contribute only when explicitly supported and dominant for at least one complete excerpt. Orientation, perspective, frame and landmark were descriptive fields. The specific anchor was event-specific, rather than a persistent entity identifier across the interview.

The original excerpt-to-event mapping assigned each of 442 excerpts to exactly one of 256 event-states. Thematic records joined the mapping on participant and excerpt identifiers; participant and event identifiers linked the result to the event-level annotations. Meta-theme and sentiment were aggregated separately within each event by selecting the most frequent non-missing value. Ties were resolved lexicographically using the values’ string representations. Event-structure features were assigned to the complete event and were not obtained by voting over excerpt labels.

The shared meta-theme vocabulary grouped 62 canonical superordinate-theme labels into 12 domains by their semantic meanings. Mapping keys trimmed leading/trailing whitespace, collapsed internal whitespace and converted to lowercase. A many-to-one lookup filled missing meta-theme entries while retaining existing nonblank assignments. The codebook documents the resulting ontology.

`hippo_prev_global` and `hippo_next_global` preserve reciprocal participant-specific sequences. They supply 251 links, including 16 links between processing segments. The public `uid` marks segment provenance. There are 235 within-segment boundaries; descriptive feature co-change summaries exclude the 16 segment joins.

The public table contains 19 fields. Eleven are active MCA variables: primary and secondary mechanism, primary and secondary context, anchor category, frame, orientation, perspective, landmark, meta-theme and sentiment. Empty/null-like analysis entries are normalized to the literal category `none`; this normalization does not authorize unsupported annotations in prospective use. Specific-anchor text is excluded. `conj_rep` and `anchor_changed_to_next_within_segment` support boundary summaries and are not MCA variables.

The anchor-change flag records only whether normalized anchor text differed at an existing within-segment boundary. Normalization trims/collapses whitespace and converts to lowercase. The flag is blank at participant tails and segment joins. It discloses no anchor identity. Feature co-changes describe established boundaries and do not identify independent causes of segmentation.

Retain public row order. The omission samples in `provenance/PERTURBATIONS.json` use zero-based row indices. For omitted event-states, retain only original links with both endpoints present; never bridge across an omitted event. The separate two-event stress omission leaves 254 event-states and 247 links and does not reconstruct a different manuscript dataset.
