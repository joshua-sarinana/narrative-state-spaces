# Analytical nomenclature

| Term | Meaning in this study |
| --- | --- |
| Event-state | One constructed event and its feature representation; 256 in the frozen dataset |
| Basin-state | A group of nearby event-states used to estimate transitions |
| Basin solution | A complete partition of the same 256 event-states |
| Macro-cluster | A broader representational group; the main macro-organization has four clusters in the 4-D geometry |
| Representational geometry | MCA coordinates used to represent annotated features |
| Projection | A reduced display; the principal display uses Dim0 × Dim3 |
| One-step routing profile | Empirical probabilities of the next basin-state |
| Complete successor profile | A row of the successor matrix, including its full set of basin-state coordinates, at the specified discount |

| Display name | File identifier | Event-count bounds |
| --- | --- | --- |
| 4-D 10-Basin | `D4_K10_L9_U47` | 9–47 |
| 4-D 9-Basin | `D4_K9_L8_U49` | 8–49 |
| 4-D 6-Basin | `D4_K6_L8_U50` | 8–50 |
| 2-D 9-Basin | `D2_K9_L8_U50` | 8–50 |
| 2-D 8-Basin | `D2_K8_L8_U39` | 8–39 |

The dimension in a solution name is its fitting dimensionality. The 2-D solutions are fitted on Dim0 and Dim3. Comparisons of centroid distances can use 4-D, 10-D or 13-D coordinates for any retained solution. These roles must remain distinct. `K` is retained in technical keys and k-means arguments; basin names in prose and figures use numerals and “Basin.” Within-solution basin identifiers are B1, B2, and so forth. Matching basin numbers across solutions does not imply matching membership.

Feature annotations are descriptions of narrative content informed by memory and navigation models. They are not measurements of hippocampal activity. Sequence relations are observed narrative adjacencies. Successor profiles model discounted multistep occupancy; their correspondence with geometry does not by itself establish neural prediction, causal mechanism or future behavioral accuracy.
