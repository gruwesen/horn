"""
Per-pair Horn clause extraction – pair-focus 2-bit capacity encoding (pairfocus).

One-hop neighbourhood around a focus pair.  Every local link (focus pair +
neighbourhood) carries a 2-bit capacity flag:
      10 -> low  capacity (800  veh/hr)
      01 -> high capacity (2000 veh/hr)
      00 -> any  capacity (1500 veh/hr, neutral default)
The 11 state is invalid and never produced.

The OUTPUT is the focus pair only.  Each direction of the focus pair gets a
2-bit congestion flag at a dedicated hour:
      10 -> low  congestion (free)
      01 -> high congestion (congested)
    00 -> unknown / no positive flow fact
The two directions use opposite hours.  Two orientations are run per pair:
      A: link_N at h8 (morning),  link_S at h16 (evening)
      B: link_N at h16,           link_S at h8

Output layout (4 bits):  [free_N, cong_N, free_S, cong_S]

Oracle-positive samples set exactly one bit per direction.  Random negatives may
also use 00 for a direction, which means unknown/no positive flow fact.  The 11
state is invalid and never sampled.

Capacity values:  LOW=800, HIGH=2000, ANY/OUT=1500 veh/hr.

Usage:
  python3 run_network_pairfocus.py --pairs 1 --orientation both --iterations 2000
  python3 run_network_pairfocus.py --pairs central --runs 1
"""

import sys
print("Loading dataset, scalers, and model...", flush=True)

import csv
import functools
import json
import os
import random
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sympy import true, false as sym_false

VALID_MODEL_KEYS = {"GN", "GCNII", "MLP", "LINREG", "SIMPLEGNN"}


def _preparse_model_key(argv):
    model_key = os.environ.get("PAIRFOCUS_MODEL", "GN").upper()
    for idx, arg in enumerate(argv[1:], start=1):
        if arg == "--model" and idx + 1 < len(argv):
            model_key = argv[idx + 1].upper()
        elif arg.startswith("--model="):
            model_key = arg.split("=", 1)[1].upper()
    if model_key not in VALID_MODEL_KEYS:
        raise ValueError(f"Unsupported model {model_key!r}; expected one of {sorted(VALID_MODEL_KEYS)}")
    return model_key

HERE         = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = HERE.parent                              # self-contained package root
DATA_DIR     = (PROJECT_ROOT / "data").resolve()
MODELS_DIR   = (PROJECT_ROOT / "models").resolve()

# All Python modules live side-by-side in src/, so a single path entry suffices.
sys.path.insert(0, str(HERE))

from run_experiment import build_model
from hornAlgorithm import HornAlgorithm, _build_global_attr, set2formula
import runFunctions
from link_id_converter import LINK_PAIRS
from eq_sampling import run_seeded_then_random_eq
from pairfocus_flow import (
    exact_flow_bits,
    is_valid_2bit_sections,
    random_flow4_except,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATASET_PATH      = Path(os.environ.get("PAIRFOCUS_DATASET", DATA_DIR / "pairfocus_static_inputs.pkl")).resolve()
SCALERS_DIR       = DATA_DIR
NEIGHBOURHOOD_CSV = Path(os.environ.get("PAIRFOCUS_NEIGHBOURHOOD_CSV", DATA_DIR / "link_pair_neighbourhood.csv")).resolve()
EXPERIMENT_DIR    = MODELS_DIR
MODEL_KEY         = _preparse_model_key(sys.argv)
CHECKPOINT        = EXPERIMENT_DIR / f"model_state_{MODEL_KEY.lower()}.pt"
USED_HPARAMS_FILE = EXPERIMENT_DIR / "used_hparams.json"
RESULTS_ROOT      = (PROJECT_ROOT / "results").resolve()

LOW_CAP_RAW      = 800.0    # veh/hr — cap = 10
HIGH_CAP_RAW     = 2000.0   # veh/hr — cap = 01
ANY_CAP_RAW      = 1500.0   # veh/hr — cap = 00 (neutral default, out-of-neigh)
CONGESTION_THRESHOLD = 5.0  # m/s  (≈ 18 km/h)

# Central Oslo pairs (radius < 0.55 from link_pairs.csv centroid)
CENTRAL_PAIRS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21]

# ---------------------------------------------------------------------------
# Neighbourhood table
# ---------------------------------------------------------------------------
_NEIGH: dict[int, dict[int, list[int]]] = {}
with open(NEIGHBOURHOOD_CSV) as f:
    for row in csv.DictReader(f):
        p = int(row["pair"])
        _NEIGH[p] = {
            1: [int(x) for x in row["hop1"].split(",") if x.strip()],
            2: [int(x) for x in row["hop2"].split(",") if x.strip()],
        }

def get_local_pairs(pair_num: int, hops: int) -> list[int]:
    included = {pair_num}
    for h in range(1, hops + 1):
        included.update(_NEIGH[pair_num].get(h, []))
    return sorted(included)

# ---------------------------------------------------------------------------
# Load dataset, scalers, model
# ---------------------------------------------------------------------------
with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)
with open(SCALERS_DIR / "exttest_capacity_scaler.pkl", "rb") as f:
    capacity_scaler = pickle.load(f)
with open(SCALERS_DIR / "exttest_speed_scaler.pkl", "rb") as f:
    speed_scaler = pickle.load(f)

static_features        = np.asarray(dataset["static_features_normalized"], dtype=np.float32)
edge_index_t           = torch.LongTensor(dataset["edge_index"])
edge_attr_t            = torch.FloatTensor(np.asarray(dataset["edge_attr"], dtype=np.float32))
static_features_tensor = torch.FloatTensor(static_features)
full_simple_hours      = [int(h) for h in dataset["simple_hours"]]

N_LINKS      = static_features.shape[0]
STATIC_DIM   = static_features.shape[1]
HOUR_DIM     = len(full_simple_hours)
HOUR_START   = STATIC_DIM
HOUR_END     = STATIC_DIM + HOUR_DIM
CAPACITY_IDX = HOUR_END
SAMPLE_DIM   = CAPACITY_IDX + 1
EDGE_DIM     = edge_attr_t.shape[1]

with open(USED_HPARAMS_FILE) as f:
    used_hparams = json.load(f)
cfg = used_hparams[MODEL_KEY]

output_lower_bound = (
    float(-speed_scaler.mean_[0] / speed_scaler.scale_[0])
    if hasattr(speed_scaler, "mean_") else None
)
model = build_model(
    MODEL_KEY, cfg,
    static_dim=STATIC_DIM, hour_dim=HOUR_DIM,
    sample_dim=SAMPLE_DIM, edge_dim=EDGE_DIM,
    device=torch.device("cpu"),
    output_lower_bound=output_lower_bound,
)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
model.eval()

def _scale_cap(cap_raw: float) -> float:
    return float(capacity_scaler.transform(np.array([[cap_raw]], dtype=np.float32))[0, 0])

LOW_CAP_SCALED  = _scale_cap(LOW_CAP_RAW)
HIGH_CAP_SCALED = _scale_cap(HIGH_CAP_RAW)
ANY_CAP_SCALED  = _scale_cap(ANY_CAP_RAW)

ORDERED_PAIRS = []
for e in LINK_PAIRS:
    ORDERED_PAIRS.append((e["label_N"], e["link_N"]))
    ORDERED_PAIRS.append((e["label_S"], e["link_S"]))

_link_pair_map: dict[int, int] = {}
for _e in LINK_PAIRS:
    _link_pair_map[_e["link_N"]] = _e["pair"]
    _link_pair_map[_e["link_S"]] = _e["pair"]
def _link_to_pair(link_idx: int) -> int:
    return _link_pair_map[link_idx]

print("Model loaded.", flush=True)


def _cap_bits_to_raw(low_bit: int, high_bit: int) -> float:
    """10 -> low, 01 -> high, 00 -> neutral/default.  11 is invalid."""
    if low_bit and high_bit:
        raise ValueError("invalid capacity state 11")
    if low_bit and not high_bit:
        return LOW_CAP_SCALED
    if high_bit and not low_bit:
        return HIGH_CAP_SCALED
    return ANY_CAP_SCALED


def _set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
class PairFocusInterp:
    """
    Attribute layout:
      Section 1 (input, 2 bits/local link):  caplow_{lbl}, caphigh_{lbl}
      Section 2 (output, 4 bits focus pair): conglow_N, conghigh_N, conglow_S, conghigh_S
    """
    def __init__(self, local_ordered_pairs, hour_N: int, hour_S: int):
        self.local_ordered_pairs = local_ordered_pairs
        self.n_local = len(local_ordered_pairs)
        self.lookupTable = {}
        for lbl, _ in local_ordered_pairs:
            self.lookupTable[f"caplow_{lbl}"]  = ([f"cap_{lbl}=low"],  f"cap_{lbl}~low")
            self.lookupTable[f"caphigh_{lbl}"] = ([f"cap_{lbl}=high"], f"cap_{lbl}~high")
        self.lookupTable["conglow_N"]  = ([f"cong_N_h{hour_N}=free"], f"cong_N_h{hour_N}~free")
        self.lookupTable["conghigh_N"] = ([f"cong_N_h{hour_N}=cong"], f"cong_N_h{hour_N}~cong")
        self.lookupTable["conglow_S"]  = ([f"cong_S_h{hour_S}=free"], f"cong_S_h{hour_S}~free")
        self.lookupTable["conghigh_S"] = ([f"cong_S_h{hour_S}=cong"], f"cong_S_h{hour_S}~cong")
        self.attributes = list(self.lookupTable.keys())
        self.lengths    = {k: len(v[0]) for k, v in self.lookupTable.items()}


# ---------------------------------------------------------------------------
# Horn algorithm
# ---------------------------------------------------------------------------
class PairFocusHornAlgorithm(HornAlgorithm):
    """
    Bit layout (2*n_local + 4):
      [0          : 2*n_local]  cap bits (low,high per link) — free input
      [2*n_local  : +4]         focus pair cong: lowN, highN, lowS, highS — derived
    """
    def __init__(self, interp, vocabulary, link_N, link_S, hour_N, hour_S,
                 congestion_threshold_raw=5.0, epsilon=0.2, delta=0.1,
                 n_eq_samples=1000,
                 n_positive_per_eq=50):
        self.n_local = interp.n_local
        self.N_CAP   = 2 * self.n_local
        super().__init__(
            interp, vocabulary, model,
            epsilon=epsilon, delta=delta,
            static_features=static_features,
            capacity_scaler=capacity_scaler,
            speed_scaler=speed_scaler,
            edge_index_tensor=edge_index_t,
            edge_attr_tensor=edge_attr_t,
            static_features_tensor=static_features_tensor,
            target_link=link_N,
            hour_start=HOUR_START, hour_end=HOUR_END,
            capacity_idx=CAPACITY_IDX,
            congestion_threshold_raw=congestion_threshold_raw,
            neutral_cap_raw=ANY_CAP_RAW,
        )
        self.link_N, self.link_S = link_N, link_S
        self.hour_N, self.hour_S = hour_N, hour_S
        self.sampleSize = n_eq_samples
        self.n_positive_per_eq = n_positive_per_eq
        self._memo: dict = {}
        self.hypSpace   = self.N_CAP

    def _build_x_all(self, local_cap_scaled, hour_val):
        x_all = np.zeros((N_LINKS, CAPACITY_IDX + 1), dtype=np.float32)
        x_all[:, :HOUR_START] = static_features
        hour_onehot = np.zeros(HOUR_DIM, dtype=np.float32)
        if hour_val in full_simple_hours:
            hour_onehot[full_simple_hours.index(hour_val)] = 1.0
        x_all[:, HOUR_START:HOUR_END] = hour_onehot
        x_all[:, CAPACITY_IDX] = ANY_CAP_SCALED
        for pos, (_, lnk) in enumerate(self.intepretor.local_ordered_pairs):
            x_all[lnk, CAPACITY_IDX] = local_cap_scaled[pos]
        return x_all

    def _cong_for(self, local_cap_scaled, hour_val, link_idx):
        x_all = self._build_x_all(local_cap_scaled, hour_val)
        x_t   = torch.as_tensor(x_all, dtype=torch.float32)
        idx_t = torch.tensor([link_idx], dtype=torch.long)
        g     = _build_global_attr(x_t, HOUR_START, HOUR_END, CAPACITY_IDX)
        x_n   = (x_t if getattr(self.model, "uses_dynamic_nodes", False)
                 else self.static_features_tensor)
        with torch.no_grad():
            y = self.model(x_n, self.edge_index_tensor, self.edge_attr_tensor,
                           g, idx_t, x_t[idx_t])
        spd = self.speed_scaler.inverse_transform(y.numpy().reshape(-1, 1)).reshape(-1)[0]
        return 1 if spd < self.congestion_threshold_raw else 0

    def _derive_cong4(self, cap_bits):
        key = tuple(cap_bits)
        if key in self._memo:
            return self._memo[key]
        if not is_valid_2bit_sections(cap_bits):
            raise ValueError("invalid pairfocus capacity assignment")
        cap_sc = [_cap_bits_to_raw(cap_bits[2*i], cap_bits[2*i+1]) for i in range(self.n_local)]
        cN = self._cong_for(cap_sc, self.hour_N, self.link_N)  # 1=congested
        cS = self._cong_for(cap_sc, self.hour_S, self.link_S)
        out = exact_flow_bits(cN) + exact_flow_bits(cS)
        self._memo[key] = out
        return out

    def _rand_cap(self):
        bits = []
        for _ in range(self.n_local):
            bits += random.choice([[0, 0], [1, 0], [0, 1]])
        return bits

    def probe(self, assignment):
        return self._derive_cong4(list(assignment[:self.N_CAP]))

    def generate_sample(self):
        cap = self._rand_cap()
        cong = self._derive_cong4(cap)
        if random.random() < 0.5:
            return (cap + cong, True)
        neg = random_flow4_except(cong)
        return (cap + neg, False)

    def _generate_positive_sample(self):
        cap = self._rand_cap()
        cong = self._derive_cong4(cap)
        return (cap + cong, True)

    def MQ(self, assignment):
        a = list(assignment)
        cap_bits = a[:self.N_CAP]
        flow_bits = a[self.N_CAP:]
        if not is_valid_2bit_sections(cap_bits + flow_bits):
            return False
        return flow_bits == self._derive_cong4(cap_bits)

    def get_hypothesis(self, S):
        H = set()
        out_secs = [[self.V[self.N_CAP + i]] for i in range(4)]
        for vec in S:
            L = [self.V[i] for i, v in enumerate(vec) if v == 1] + [true]
            ant = functools.reduce(lambda x, y: x & y, L)
            R = sum([sec for sec in out_secs if not (set(L) & set(sec))], []) + [sym_false]
            for con in R:
                H.add(ant >> con)
        return H

    def EQ(self, H):
        return run_seeded_then_random_eq(
            H, self.evalFormula, set2formula,
            self._generate_positive_sample, self.generate_sample,
            self.n_positive_per_eq, self.sampleSize,
        )


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------
def _make_experiment_dir(hops, threshold, orientation, pairs_to_run,
                         n_positive_per_eq, n_eq_samples, seed):
    ts = time.strftime("%Y%m%d_%H%M%S")
    thr = f"thr{threshold:.1f}".replace(".", "p")
    seed_part = f"_seed{seed}" if seed is not None else ""
    d = RESULTS_ROOT / f"horn_network_pairfocus_h{hops}_{thr}_{MODEL_KEY}_without_neigh{seed_part}_{ts}"
    for sub in ["output_data/hornRules", "output_data/metadata"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": ts, "variant": "pairfocus", "script": "run_network_pairfocus.py",
        "model": MODEL_KEY, "hops": hops, "orientation": orientation,
        "checkpoint": str(CHECKPOINT), "dataset": str(DATASET_PATH),
        "neighbourhood_csv": str(NEIGHBOURHOOD_CSV),
        "congestion_threshold_ms": threshold,
        "capacity_levels_vehhr": {"low": LOW_CAP_RAW, "high": HIGH_CAP_RAW, "any": ANY_CAP_RAW},
        "capacity_encoding": "2-bit/link: 10=low, 01=high, 00=any",
        "congestion_encoding": "4 bits focus pair: lowN,highN,lowS,highS at oriented hours",
        "n_positive_per_eq": n_positive_per_eq,
        "n_eq_samples": n_eq_samples,
        "seed": seed,
        "pairs": [e["pair"] for e in pairs_to_run],
    }
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return d


def run_pair(pair_entry, run_nr, exp_dir, hops, threshold, hour_N, hour_S,
             iterations, n_positive_per_eq, n_eq_samples, epsilon, delta, tag):
    p = pair_entry["pair"]
    link_N, link_S = pair_entry["link_N"], pair_entry["link_S"]
    label_N, label_S = pair_entry["label_N"], pair_entry["label_S"]
    local_set = set(get_local_pairs(p, hops))
    local_ord = [(l, k) for (l, k) in ORDERED_PAIRS if _link_to_pair(k) in local_set]

    interp = PairFocusInterp(local_ord, hour_N, hour_S)
    n_bits = 2 * interp.n_local + 4
    V = runFunctions.define_variables(n_bits)
    horn = PairFocusHornAlgorithm(interp, V, link_N, link_S, hour_N, hour_S,
                                  congestion_threshold_raw=threshold,
                                  epsilon=epsilon, delta=delta,
                                  n_eq_samples=n_eq_samples,
                                  n_positive_per_eq=n_positive_per_eq)
    metadata, hornRules, iters = horn.learn(iterations, background=set())
    thr = f"thr{threshold:.1f}".replace(".", "p")
    stem = f"pair{p:02d}_{label_N}_{label_S}_{tag}_h{hops}_{thr}_i{iters}_r{run_nr}"
    cwd = Path.cwd()
    try:
        os.chdir(exp_dir)
        runFunctions.storeMetadata(stem, metadata)
        runFunctions.storeHornRules(stem, hornRules, interp.lookupTable)
    finally:
        os.chdir(cwd)
    return iters, len(hornRules), n_bits, horn.sampleSize


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pair-focus 2-bit capacity Horn extraction")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--n-eq-samples", type=int, default=1000)
    ap.add_argument("--n-positive-per-eq", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--pairs", type=str, default="central")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--orientation", type=str, default="both", choices=["A", "B", "both"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", type=str.upper, default=MODEL_KEY, choices=sorted(VALID_MODEL_KEYS))
    args = ap.parse_args()

    if args.model != MODEL_KEY:
        raise ValueError("--model must be known before model loading; restart this script with the desired --model value")

    _set_seed(args.seed)

    if args.pairs == "central":
        sel = CENTRAL_PAIRS
    elif args.pairs == "all":
        sel = [e["pair"] for e in LINK_PAIRS]
    else:
        sel = [int(x) for x in args.pairs.split(",")]
    pair_by_num = {e["pair"]: e for e in LINK_PAIRS}
    pairs_to_run = [pair_by_num[pair] for pair in sel]
    orients = ["A", "B"] if args.orientation == "both" else [args.orientation]

    exp_dir = _make_experiment_dir(args.hops, args.threshold, args.orientation,
                                   pairs_to_run, args.n_positive_per_eq,
                                   args.n_eq_samples, args.seed)
    print(f"Experiment output : {exp_dir}")
    print(f"Pairs             : {len(pairs_to_run)} | orientations={orients} | hops={args.hops}")
    print(f"Seed              : {args.seed}")
    print()
    t0 = time.time()
    for pe in pairs_to_run:
        for o in orients:
            hN, hS = (8, 16) if o == "A" else (16, 8)
            for r in range(args.runs):
                ts = time.time()
                it, nr, nb, ss = run_pair(pe, r, exp_dir, args.hops, args.threshold,
                                          hN, hS, args.iterations, args.n_positive_per_eq,
                                          args.n_eq_samples,
                                          args.epsilon, args.delta, f"or{o}_N{hN}S{hS}")
                print(f"pair {pe['pair']:2d} {pe['label_N']}/{pe['label_S']} or{o} "
                      f"(N@{hN},S@{hS}) bits={nb} ss={ss} iters={it} rules={nr} "
                      f"{time.time()-ts:.1f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.1f}s.")
