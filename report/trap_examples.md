# SciFact trap examples

These examples come from the runs with 20 traps per type, seed 42 and k = 5.
The descriptions summarize synthetic edits to SciFact abstracts. The trap and
source IDs let readers locate the full texts in the generated `trap_chunks.json`
files and in `data/chunks.json`. See the [dataset attribution and licence](../README.md#data).

## Near-copy

| Trap chunk | Target | Source chunk | Example edit |
|---|---:|---|---|
| `trap_near_copy_s42_2#0` | 1029 | `13940200#0` | In the abstract on type 1 diabetes and IL2RA, two instances of “lower” became “higher” and a numeric 1 became 10. |
| `trap_near_copy_s42_4#0` | 1370 | `2425364#0` | In the review of maternal vitamin D and pregnancy outcomes, three instances of “increased” became “decreased”. |
| `trap_near_copy_s42_10#0` | 649 | `12789595#0` | In the abstract on computer-assisted medical education, “enhanced”, “more” and “higher” became “impaired”, “less” and “lower”. |
| `trap_near_copy_s42_17#0` | 1041 | `16626264#0` | In the abstract on the histone variant Htz1, “less”, “activated” and “promotes” became “more”, “inhibited” and “suppresses”. |
| `trap_near_copy_s42_19#0` | 1298 | `11718220#0` | In the compression-stocking trial abstract, the generator changed 2518 to 1259, 1 to 10, and 64 to 32. |

## Contradiction

| Trap chunk | Target | Source chunk | Example edit |
|---|---:|---|---|
| `trap_contradiction_s42_0#0` | 589 | `10984005#0` | In the ADHD medication and cardiovascular events abstract, a statement of association was negated and increase terms were replaced with decrease terms. |
| `trap_contradiction_s42_4#0` | 1370 | `2425364#0` | In the maternal vitamin D review, “is associated with” became “is not associated with” and “increased” became “decreased” three times. |
| `trap_contradiction_s42_10#0` | 649 | `12789595#0` | In the medical education abstract, directional edits turned enhancement into impairment and changed comparative terms. The same edits also appear in the near-copy run. |
| `trap_contradiction_s42_16#0` | 1049 | `12486491#0` | The trap denies that the authors show enrichment of Rpl38 expression and rewrites their conclusion about ribosomal proteins as a refutation. |
| `trap_contradiction_s42_18#0` | 1150 | `11369420#0` | The trap denies that the authors show Tetraspanin 3 is a Musashi 2 target and reverses directional terms in the description of the leukemia experiments. |

## Mix

| Trap chunk | Target | Source chunks | Topics joined |
|---|---:|---|---|
| `trap_mix_s42_1#0` | 1187 | `52873726#0`, `36860856#0` | TEAD regulation in the Hippo signaling pathway and a 40 Hz auditory potential recorded from the scalp. |
| `trap_mix_s42_2#0` | 1029 | `11899391#0`, `60206680#0` | IL2RA variation in type 1 diabetes and the R language for analysis and graphics. |
| `trap_mix_s42_10#0` | 649 | `12789595#0`, `19205326#0` | Computer-assisted medical education and long-term alendronate use for osteoporosis. |
| `trap_mix_s42_17#0` | 1041 | `25254425#0`, `44264297#0` | Nucleosome stability and network meta-analysis for indirect treatment comparisons. |
| `trap_mix_s42_19#0` | 1298 | `11718220#0`, `2883827#0` | A compression-stocking trial after stroke and Cytoscape features for data integration and visualization. |
