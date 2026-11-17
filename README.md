# Pairfocus Horn-Rule Extraction — Reproduction Package

This self-contained folder reproduces the **pairfocus** experiment from the paper
*"Extracting Rules via Queries from ML Models for Long Term Traffic Congestion
Prediction"*. It applies the sampling HORN algorithm to four trained traffic
models (LINREG, MLP/FF, GN, GCNII) to extract Horn rules describing each model's
congestion behaviour around a focus road-link pair and its one-hop neighbourhood.

The notebook here runs a **small smoke test** (5 learner iterations, 1 run, 1 link
pair, all four models) so you can verify the pipeline end-to-end. The same runner
reproduces the full experiment by changing a few arguments (see below).

This code accompanies the paper above (see [Citation](#citation)). Large language
models were used extensively as coding aids; see the paper's LLM disclosure.

## Folder layout

```
pairfocus_reproduction/
├── environment.yml                     conda environment specification
├── LICENSE                             MIT License (see also Third-party work below)
├── published_results/                  extracted rules reported in the paper (see below)
├── run_pairfocus_experiment.ipynb      smoke-test notebook (start here)
├── README.md
├── src/                                all Python code (single import dir)
│   ├── run_network_pairfocus.py        experiment runner (paths point inside this folder)
│   ├── hornAlgorithm.py                HORN learning loop + model-as-oracle
│   ├── eq_sampling.py                  seeded + random equivalence-query sampling
│   ├── pairfocus_flow.py               2-bit flow-state encoding helpers
│   ├── intepretor.py                   binary-vector <-> model-input mapping
│   ├── runFunctions.py                 rule formatting, cleaning, CSV output
│   ├── run_experiment.py               model architectures + build_model()
│   ├── link_id_converter.py            link-index <-> pair-label mapping (LINK_PAIRS)
│   └── reclean_horn_csvs.py            optional: re-clean rule CSVs of a result dir
├── models/                             trained checkpoints (without_neigh variant)
│   ├── model_state_linreg.pt
│   ├── model_state_mlp.pt
│   ├── model_state_gn.pt
│   ├── model_state_gcnii.pt
│   ├── used_hparams.json               hyper-parameters used to build each model
│   └── experiment_metrics.csv          reference training/eval metrics
├── data/
│   ├── pairfocus_static_inputs.pkl     static features, edge_index, edge_attr, hours (~18 KB)
│   ├── exttest_capacity_scaler.pkl     capacity StandardScaler
│   ├── exttest_speed_scaler.pkl        speed StandardScaler
│   ├── link_pairs.csv                  35 link pairs + geometry (reference)
│   ├── link_pair_neighbourhood.csv     original hop-1/hop-2 neighbourhood table
│   └── link_pair_neighbourhood_physical_candidate.csv   corrected physical-adjacency table
└── results/                            experiment output is written here at runtime
```

## Neighbourhood table — which one to use

Two neighbourhood tables are included:

* `link_pair_neighbourhood_physical_candidate.csv` — the **physical shared-node
  adjacency** table. This is the one used for the paper's final results; it
  corrects the hop-1 neighbourhoods of pairs 16, 20, 26, 27.
* `link_pair_neighbourhood.csv` — the original table (kept for provenance).

The notebook uses the **physical candidate** table by default via the
`PAIRFOCUS_NEIGHBOURHOOD_CSV` environment variable. The default focus pair for
the smoke test (pair 33) has an identical neighbourhood in both tables.

## Setup

```bash
conda env create -f environment.yml
conda activate horn-pairfocus-repro
jupyter lab   # or: jupyter notebook
```

Then open `run_pairfocus_experiment.ipynb` and run the cells top to bottom.

Everything runs on CPU and is quick. The full ~2 GB MATSim training dataset is
**not** included: the HORN extraction only needs the fixed map/topology arrays
(static features, `edge_index`, `edge_attr`, hours), which are stored in the
small `data/pairfocus_static_inputs.pkl`. The four trained model checkpoints are
in `models/`.

## Reproducing the full experiment

The smoke test uses `--iterations 5 --runs 1 --pairs 33`. The pairfocus comparison
in the paper uses, per model:

* all 35 link pairs (`--pairs all`),
* both orientations (`--orientation both`),
* 5 runs per pair and orientation,
* a 500 learner-iteration cap (`--iterations 500`),
* 50 seeded positives and 500 random final equivalence samples
  (`--n-positive-per-eq 50 --n-eq-samples 500`),
* hop radius 1 and a 5.0 m/s congestion threshold.

The five runs were obtained differently per model:

| Model  | Invocation                          | Seed(s)    |
|--------|-------------------------------------|------------|
| LINREG | one process with `--runs 5`         | 9400       |
| MLP    | one process with `--runs 5`         | 9500       |
| GCNII  | one process with `--runs 5`         | 9700       |
| GN     | five processes, each `--runs 1`     | 1, 2, 3, 4, 5 |

Example (LINREG, full budget):

```bash
cd src
PAIRFOCUS_NEIGHBOURHOOD_CSV=../data/link_pair_neighbourhood_physical_candidate.csv \
python run_network_pairfocus.py \
    --model LINREG --pairs all --orientation both --hops 1 --threshold 5.0 \
    --iterations 500 --n-eq-samples 500 --n-positive-per-eq 50 \
    --runs 5 --seed 9400
```

For GN, repeat with `--runs 1 --seed N` for N = 1..5. Result directories appear
under `results/`.

Note on exact reproduction: the stochastic equivalence-query sampling depends on
the seed, library versions and the order in which runs consume the random stream.
The reported results for link pairs 16, 20, 26 and 27 were produced in separate
runs after the neighbourhood correction described above, so a single new pass over
all pairs will give statistically equivalent but not necessarily identical rules.
The LINREG results are the most stable (they converge in a handful of iterations);
MLP, GN and GCNII mostly hit the 500-iteration cap.

## Published results

`published_results/` contains the Horn-rule output behind the paper's results
(four models, 35 link pairs, two orientations, five runs each, 500-iteration cap;
1,400 runs in total):

* `rules_final_full500/result_dirs/` — one directory per model and seed with
  `output_data/hornRules` (raw rules), `hornRulesClean` (entailed rules removed),
  `hornRulesCleanCounts`, and `metadata` (per-iteration log; the last row is `TRUE`
  if the learner terminated before the cap). `run_meta.json` records the settings.
  GN has five directories (seeds 1-5); the others hold five runs in one directory.
* `rules_final_full500/manifests/` and `replaced_files.csv` — provenance. Runs for
  link pairs 16, 20, 26 and 27 were regenerated with the corrected neighbourhood
  table and overlaid on the original runs; `replaced_files.csv` lists every file
  replaced, and `run_meta_original_before_physadj_overlay.json` keeps the original
  settings.
* `csv_export/` — the same data as flat CSV files (`runs.csv`, `clean_rules.csv`,
  `pair_orientation_summary.csv`, ...) for analysis without the directory layout.

Paths in the provenance JSON files were rewritten to be relative to this package.

## Citation

If you use this code, please cite:

> G. H. Wesenberg and A. Ozaki. *Extracting Rules via Queries from ML Models for
> Long Term Traffic Congestion Prediction*. Proceedings of the 8th Northern Lights
> Deep Learning Conference (NLDL), PMLR 352, 2027.

## Third-party work and references

**HORN implementation.** The HORN learner (`src/hornAlgorithm.py`,
`src/intepretor.py`, `src/runFunctions.py`) is adapted from the code base of
S. Blum, https://github.com/ZofieB/MT_UiB, which is the experiments repository
for:

* S. Blum, R. Koudijs, A. Ozaki, S. Touileb. *Learning Horn envelopes via queries
  from language models*. International Journal of Approximate Reasoning 171
  (2024), 109026. https://doi.org/10.1016/j.ijar.2023.109026

That repository's `Horn.py` is in turn taken from the implementation of
J. Jøsang, C. Persia and A. Ozaki (*Extracting Horn Theories from Neural Networks
with Queries and Counterexamples*). The upstream repository does not declare a
license.

The algorithm itself is from:

* D. Angluin, M. Frazier, L. Pitt. *Learning conjunctions of Horn clauses*.
  Machine Learning 9 (1992), 147-164. https://doi.org/10.1007/BF00992675

Changes relative to that implementation: the language-model oracle was replaced
by direct inference with the traffic models, equivalence queries are simulated by
seeded-positive plus random sampling (`src/eq_sampling.py`), and the pairfocus
binarisation is in `src/run_network_pairfocus.py` and `src/pairfocus_flow.py`.

**Model families** implemented in `src/run_experiment.py`:

* GN follows the graph-network block design of P. W. Battaglia et al.,
  *Relational inductive biases, deep learning, and graph networks*, arXiv:1806.01261 (2018).
* GCNII follows M. Chen, Z. Wei, Z. Huang, B. Ding, Y. Li, *Simple and Deep Graph
  Convolutional Networks*, ICML 2020 (PMLR 119:1725-1735).

**Data context.** The models were trained on traffic scenarios simulated with
MATSim (A. Horni, K. Nagel, K. W. Axhausen (eds.), *The Multi-Agent Transport
Simulation MATSim*, Ubiquity Press, 2016, https://doi.org/10.5334/baw) using a
synthetic population from S. Flügel et al., *Towards activity-based demand
modelling for the Greater Oslo Area*. The simulation data and training set are
not part of this package; `data/pairfocus_static_inputs.pkl` holds only the fixed
road-network arrays needed to query the trained models.

**Software dependencies** (versions pinned in `environment.yml`):

| Library | Version | License |
|---|---|---|
| Python | 3.10.14 | PSF |
| PyTorch | 2.8.0 | BSD-3-Clause |
| NumPy | 2.2.6 | BSD-3-Clause |
| SciPy | 1.15.3 | BSD-3-Clause |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| SymPy | 1.14.0 | BSD-3-Clause |
| pandas | 2.3.2 | BSD-3-Clause |
| Matplotlib | 3.10.8 | Matplotlib (PSF-based) |
| tqdm | 4.67.1 | MPL-2.0 / MIT |
| JupyterLab | 4.4.7 | BSD-3-Clause |

## License

The original contributions of this repository are released under the MIT License
(see `LICENSE`). The three adapted files listed under "HORN implementation" above
derive from upstream code that has no declared license, so their reuse outside
this repository should be cleared with the upstream authors.
