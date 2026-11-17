# Pairfocus full500 physical-adjacency final dataset

Generated: 2026-08-14T09:26:45

This dataset starts from the trusted full500 pairfocus runs and replaces only the pairs whose hop-1 neighbourhoods were wrong under the physical shared-node road-adjacency definition.

Affected pairs: 16, 20, 26, 27
Corrected neighbourhood table: `results/pairfocus_neighbourhood_adjacency_audit/link_pair_neighbourhood_physical_candidate.csv`

## Layout

- `result_dirs/`: copied result directories with corrected pair files overlaid.
- `manifests/`: one CSV manifest per model, plus a combined manifest.
- `replaced_files.csv`: every file replaced from a correctional run.
- `validation_summary.json`: coverage checks performed while building this dataset.

## Model result directories

| Model | Seed | Final result dir | Corrected source |
|---|---:|---|---|
| LINREG | 9400 | `result_dirs/horn_network_pairfocus_h1_thr5p0_LINREG_without_neigh_seed9400_20260728_190312` | `results/horn_network_pairfocus_h1_thr5p0_LINREG_without_neigh_seed9400_20260813_135741` |
| MLP | 9500 | `result_dirs/horn_network_pairfocus_h1_thr5p0_MLP_without_neigh_seed9500_20260728_190642` | `results/horn_network_pairfocus_h1_thr5p0_MLP_without_neigh_seed9500_20260813_135741` |
| GN | 1 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed1_20260807_101343` | `results/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed1_20260813_135741` |
| GN | 2 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed2_20260807_230417` | `results/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed2_20260813_161733` |
| GN | 3 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed3_20260808_120057` | `results/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed3_20260813_183204` |
| GN | 4 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed4_20260809_010908` | `results/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed4_20260813_205056` |
| GN | 5 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed5_20260809_143202` | `results/horn_network_pairfocus_h1_thr5p0_GN_without_neigh_seed5_20260813_224901` |
| GCNII | 9700 | `result_dirs/horn_network_pairfocus_h1_thr5p0_GCNII_without_neigh_seed9700_20260803_132458` | `results/horn_network_pairfocus_h1_thr5p0_GCNII_without_neigh_seed9700_20260813_135741` |

## Validation summary

- Corrected model checks passed: LINREG, MLP, GN, GCNII
- Final model checks passed: LINREG, MLP, GN, GCNII
- Replaced files: 640
