# Public annotation reimplementation templates

These files specify the minimum constraints that a complete prompt must preserve when applying the annotation framework with a current language model. They are model-independent reimplementation templates, not historical production prompts and not a record of prompt development.

Because model behavior, interfaces, context limits, and structured-output systems change, users must incorporate these constraints into a complete implementation appropriate to the selected model. That implementation must also define how source material is supplied, how the requested schema is enforced, how invalid outputs are retried or rejected, how long inputs are segmented, and how protected material is handled.

The released `documentation/Annotation_Codebook.xlsx` is the authoritative vocabulary and field specification. A model must receive the relevant codebook definitions with each annotation task. The templates do not authorize silent changes to that codebook.

## Execution order

1. For a new corpus only, the optional codebook-refinement template may be applied to designated development material. Human reviewers must approve and freeze a new codebook version before annotation begins.
2. Apply the thematic template to construct the ordered excerpt backbone and assign thematic fields.
3. Apply the HPC-informed event template to the fixed excerpts. It may group consecutive excerpts into events but may not edit, reorder, or resegment the excerpt backbone.

For reproduction of the released study, do not run codebook refinement and do not regenerate the annotations. Begin with `data/event_level_analysis_input.csv` and execute the quantitative pipeline.

These templates constrain the method but cannot make different models or model versions produce identical annotations. Any new application must report the model, interface, date, model-specific instructions, schema-enforcement method, segmentation procedure, and codebook version actually used.
