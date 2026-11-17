# Pairfocus Full500 Physical-Adjacency Final CSV Dataset

Source dataset: `results/pairfocus_full500_physadj_final_20260814`
Physical neighbourhood table: `results/pairfocus_neighbourhood_adjacency_audit/link_pair_neighbourhood_physical_candidate.csv`

## Files

- `runs.csv`: one row per model/pair/orientation/replicate run (1400 rows).
- `clean_rules.csv`: one row per cleaned Horn-rule line (5982 rows).
- `pair_orientation_summary.csv`: one row per model/pair/orientation aggregate (280 rows).
- `neighbourhoods.csv`: one row per physical link pair (35 rows).
- `source_manifest.csv`: source/final result-directory ledger copied from the final dataset.
- `replacement_ledger.csv`: file-level overlay ledger copied from the final dataset.

`model` preserves the implementation key (`MLP`), while `model_label` gives the paper-facing label (`FF`).
`TRUE` means the run terminated before the configured iteration cap under the final EQ procedure; capped rows are cap-time hypotheses.
