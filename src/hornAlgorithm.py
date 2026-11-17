# Adapted from the HORN implementation of Blum, Koudijs, Ozaki and Touileb,
# "Learning Horn envelopes via queries from language models",
# Int. J. Approx. Reason. 171 (2024) 109026, https://doi.org/10.1016/j.ijar.2023.109026
# (implementing Angluin, Frazier and Pitt, Mach. Learn. 9 (1992) 147-164). See README.md.
# Code base: https://github.com/ZofieB/MT_UiB (S. Blum); its Horn.py derives from the
# implementation of Josang, Persia and Ozaki.
# Changes: replaced Hugging Face fill-mask pipeline with direct traffic model inference;
#          removed sentence/trigger-token logic; probe() now calls the PyTorch model directly

import functools
import random
import timeit
from math import prod
import numpy as np
import torch
from sympy import true, false, Pow, log
from intepretor import Intepretor
from tqdm import tqdm


def divideV(intepretor: Intepretor, V):
    splitIndexes = []
    i = 0
    for x in intepretor.lengths.items():
        splitIndexes.append((i, x[1] + i))
        i += x[1]
    splitIndexes.append((i, len(V)))
    available = [V[s:e] for s, e in splitIndexes]
    return available

def getRandomValue(length, allow_zero):
    vec = list(np.zeros(length, dtype=np.int8))
    i = random.randrange(length + 1) if allow_zero else random.randrange(length)
    if i < length:
        vec[i] = 1
    return vec

def set2formula(set):
    return functools.reduce(lambda x, y: x & y, set | {True})


def _build_global_attr(x_all_t, hour_start, hour_end, capacity_idx):
    """Inlined from run_experiment.build_global_attr."""
    hour_profile = x_all_t[:, hour_start:hour_end].mean(dim=0, keepdim=True)
    mean_capacity = x_all_t[:, capacity_idx:capacity_idx + 1].mean(dim=0, keepdim=True)
    return torch.cat([hour_profile, mean_capacity], dim=1)


class HornAlgorithm():
    def __init__(self, intepretor: Intepretor, vocabulary, model, epsilon=0.2, delta=0.1,
                 static_features=None, capacity_scaler=None, speed_scaler=None,
                 edge_index_tensor=None, edge_attr_tensor=None, static_features_tensor=None,
                 target_link=0, hour_start=5, hour_end=25, capacity_idx=25,
                 congestion_threshold_raw=5.0, neutral_cap_raw=1500.0):
        """
        intepretor             : Intepretor instance (defines feature dimensions)
        vocabulary             : V from runFunctions.define_variables(total_bits)
        model                  : loaded PyTorch model in eval mode
        epsilon, delta         : PAC learning parameters
        static_features        : numpy array (n_links × static_dim)
        capacity_scaler        : sklearn scaler for capacity values
        speed_scaler           : sklearn scaler to invert speed predictions
        edge_index_tensor      : torch.LongTensor (2 × n_edges)
        edge_attr_tensor       : torch.FloatTensor (n_edges × edge_dim)
        static_features_tensor : torch.FloatTensor (n_links × static_dim)
        target_link            : index of the link being probed (0-based)
        hour_start, hour_end   : slice indices for the hour one-hot block in x_all
        capacity_idx           : column index of the capacity feature in x_all
        congestion_threshold_raw : speed threshold in m/s (< threshold → congested)
        neutral_cap_raw        : capacity (veh/hr) used for all non-target links
        """
        self.intepretor = intepretor
        self.V = vocabulary
        self.model = model
        self.static_features = static_features
        self.capacity_scaler = capacity_scaler
        self.speed_scaler = speed_scaler
        self.edge_index_tensor = edge_index_tensor
        self.edge_attr_tensor = edge_attr_tensor
        self.static_features_tensor = static_features_tensor
        self.target_link = target_link
        self.hour_start = hour_start
        self.hour_end = hour_end
        self.capacity_idx = capacity_idx
        self.congestion_threshold_raw = congestion_threshold_raw
        self.neutral_cap_raw = neutral_cap_raw

        self.hypSpace = prod(self.intepretor.lengths.values()) * 2
        self.sampleSize = int((1 / epsilon) * log((Pow(2, self.hypSpace) / delta), 2))
        self.Vsections = divideV(intepretor, vocabulary)

        self.bad_nc = []

    def probe(self, bin):
        """
        Query the traffic model with a binary feature vector.

        Builds a full 70-link graph snapshot where every link is at the neutral
        state (zero hour, neutral_cap_raw capacity) except the target link, which
        is set to the hour and capacity decoded from `bin`.

        Returns [1, 0] if the target link is predicted congested (speed < threshold),
        or [0, 1] if free-flowing.
        """
        decoded = self.intepretor.binaryToModelInput(bin)

        n_links = self.static_features.shape[0]
        feature_dim = self.capacity_idx + 1  # static_dim + hour_dim + 1
        x_all = np.zeros((n_links, feature_dim), dtype=np.float32)

        # Static features for all links
        x_all[:, :self.hour_start] = self.static_features

        # Neutral capacity for all links
        neutral_cap_scaled = float(
            self.capacity_scaler.transform(np.array([[self.neutral_cap_raw]], dtype=np.float32))[0, 0]
        )
        x_all[:, self.capacity_idx] = neutral_cap_scaled

        # Override target link with probed hour and capacity
        x_all[self.target_link, self.hour_start:self.hour_end] = decoded['hour_onehot']
        x_all[self.target_link, self.capacity_idx] = float(
            self.capacity_scaler.transform(np.array([[decoded['cap_raw']]], dtype=np.float32))[0, 0]
        )

        x_all_t = torch.as_tensor(x_all, dtype=torch.float32)
        link_idx_t = torch.tensor([self.target_link], dtype=torch.long)

        global_attr = _build_global_attr(x_all_t, self.hour_start, self.hour_end, self.capacity_idx)

        x_nodes = x_all_t if getattr(self.model, 'uses_dynamic_nodes', False) else self.static_features_tensor

        with torch.no_grad():
            y_scaled = self.model(
                x_nodes,
                self.edge_index_tensor,
                self.edge_attr_tensor,
                global_attr,
                link_idx_t,
                x_all_t[link_idx_t],
            )

        speed_scaled = y_scaled.item()
        speed_raw = float(
            self.speed_scaler.inverse_transform(np.array([[speed_scaled]], dtype=np.float32))[0, 0]
        )

        if speed_raw < self.congestion_threshold_raw:
            return [1, 0]   # congested
        else:
            return [0, 1]   # free-flowing

    def generate_sample(self):
        vec = sum([getRandomValue(self.intepretor.lengths[att], True)
                   for att in self.intepretor.attributes[:-1]], [])
        n_classes = list(self.intepretor.lengths.values())[-1]
        labelVec = getRandomValue(n_classes, False)
        vec += labelVec
        predLabelVec = self.probe(vec)
        if predLabelVec == [0] * n_classes:
            self.bad_nc.append(vec)
        return (vec, labelVec == predLabelVec)

    def evalFormula(self, formula, vector):
        if formula in {true, false}:
            return formula
        d = {}
        for i in range(len(self.V)):
            d[self.V[i]] = vector[i] == 1
        return formula.subs(d)

    def EQ(self, H):
        h = set2formula(H)
        for sampleNr in range(1, self.sampleSize + 1):
            (generatedVec, label) = self.generate_sample()
            evaluatedTheory = self.evalFormula(h, generatedVec)
            if label == False and evaluatedTheory and (generatedVec not in self.bad_nc):
                return (generatedVec, sampleNr)
            if label == True and evaluatedTheory == False:
                return (generatedVec, sampleNr)
        return (True, 0)

    def MQ(self, assignment):
        n_classes = list(self.intepretor.lengths.values())[-1]
        vec, labelVec = assignment[:-n_classes], assignment[-n_classes:]
        return labelVec == self.probe(assignment)

    def get_hypothesis(self, S):
        H = set()
        sFiltered = list(filter(lambda x: x not in self.bad_nc, S))
        for vec in sFiltered:
            L = [self.V[i] for i, val in enumerate(vec) if val == 1] + [true]
            R = sum(list(filter(lambda x: (set(L) & set(x)) == set(), self.Vsections)), []) + [false]
            ant = functools.reduce(lambda x, y: x & y, L)
            for con in R:
                rule = (ant) >> con
                H.add(rule)
        return H

    def find_bad_nc(self, H, S):
        h = set2formula(H)
        for vec in S:
            if (vec not in self.bad_nc) and (self.evalFormula(h, vec) == True):
                self.bad_nc.append(vec)

    def learn(self, iterationCap, background=set()):
        metadata = []
        posCounterEx = []
        H = background
        S = []

        for iteration in tqdm(range(1, iterationCap + 1), desc="Eq iteration"):
            start = timeit.default_timer()
            (counterEx, sampleNr) = self.EQ(H)

            if counterEx == True:
                metadata.append([iteration, len(H), "TRUE", float(f"{timeit.default_timer()-start:.3f}")])
                return (metadata, H, iteration)

            pos_ex = False
            for clause in H.copy():
                if (clause not in background) and (self.evalFormula(clause, counterEx) == False):
                    H.remove(clause)
                    pos_ex = True

            if pos_ex:
                posCounterEx.append(counterEx)

            if pos_ex == False:
                for idx, s in enumerate(S):
                    cap_sc = [1 if (s[i] == 1 and counterEx[i] == 1) else 0 for i in range(len(self.V))]
                    true_sc = {i for i, val in enumerate(cap_sc) if val == 1}
                    true_s = {i for i, val in enumerate(s) if val == 1}
                    if cap_sc in self.bad_nc:
                        continue
                    if cap_sc in S:
                        self.bad_nc.append(cap_sc)
                        continue
                    if true_sc.issubset(true_s) and (true_s.issubset(true_sc) == False):
                        if self.MQ(cap_sc) == False:
                            S[idx] = cap_sc
                            break
                else:
                    S.append(counterEx)
                H = self.get_hypothesis(S).union(background)
                H = set(filter(lambda x: all(self.evalFormula(x, vec) for vec in posCounterEx), H.copy()))

            self.find_bad_nc(H, S)
            metadata.append([iteration, len(H), sampleNr, float(f"{timeit.default_timer()-start:.3f}")])

        return (metadata, H, iterationCap)


# ---------------------------------------------------------------------------
# SpatialHornAlgorithm — multi-output extension
# ---------------------------------------------------------------------------
class SpatialHornAlgorithm(HornAlgorithm):
    """
    Learns spatial co-congestion rules across multiple output links simultaneously.

    Vocabulary layout (all defined in the Intepretor's CSV files):
      [input section 0] ... [input section n_input_sections-1]
      [output section 0] ... [output section k-1]   ← congestion of each output link

    The probe runs the model once and returns the concatenation of each output
    link's congestion as a single flat list:
      [cong_link_A_bit0, cong_link_A_bit1, cong_link_B_bit0, cong_link_B_bit1, ...]

    generate_sample and MQ are overridden to treat the entire joint output as
    the label rather than just the last section.
    """

    def __init__(self, intepretor, vocabulary, model,
                 n_input_sections,
                 static_features, capacity_scaler, speed_scaler,
                 edge_index_tensor, edge_attr_tensor, static_features_tensor,
                 corridor_links, output_links,
                 interesting_hours, neutral_hour, full_simple_hours,
                 hour_start, hour_end, capacity_idx,
                 multilink_cap_bins, neutral_cap_raw,
                 congestion_threshold_raw=5.0,
                 epsilon=0.2, delta=0.1):
        """
        n_input_sections : how many Intepretor sections are inputs (hour + cap per link)
        output_links     : list of link indices whose congestion forms the joint output
        corridor_links   : list of link indices with capacity inputs (may differ from output_links)
        """
        super().__init__(
            intepretor, vocabulary, model,
            epsilon=epsilon, delta=delta,
            static_features=static_features,
            capacity_scaler=capacity_scaler,
            speed_scaler=speed_scaler,
            edge_index_tensor=edge_index_tensor,
            edge_attr_tensor=edge_attr_tensor,
            static_features_tensor=static_features_tensor,
            target_link=output_links[0],   # unused by this class's probe()
            hour_start=hour_start, hour_end=hour_end,
            capacity_idx=capacity_idx,
            congestion_threshold_raw=congestion_threshold_raw,
            neutral_cap_raw=neutral_cap_raw,
        )
        self.n_input_sections   = n_input_sections
        self.corridor_links     = corridor_links
        self.output_links       = output_links
        self.interesting_hours  = interesting_hours
        self.neutral_hour       = neutral_hour
        self.full_simple_hours  = full_simple_hours
        self.multilink_cap_bins = multilink_cap_bins

        attrs = list(intepretor.attributes)
        self._input_attrs  = attrs[:n_input_sections]
        self._output_attrs = attrs[n_input_sections:]
        self._n_output_bits = sum(intepretor.lengths[a] for a in self._output_attrs)

    # ------------------------------------------------------------------
    def probe(self, bin):
        """
        Decode hour and corridor capacities, run model once, return
        the concatenated [congested/free_flowing] bits for every output link.
        """
        bin = list(bin)

        # Decode hour (first input section)
        n_hours   = len(self.interesting_hours)
        hour_bits = bin[:n_hours]
        h_idx     = int(np.argmax(hour_bits)) if any(hour_bits) else -1
        hour_val  = self.interesting_hours[h_idx] if h_idx >= 0 else self.neutral_hour

        hour_onehot = np.zeros(len(self.full_simple_hours), dtype=np.float32)
        if hour_val in self.full_simple_hours:
            hour_onehot[self.full_simple_hours.index(hour_val)] = 1.0

        # Decode capacities for each corridor link
        n_cap_bins = len(self.multilink_cap_bins)
        offset     = n_hours
        cap_raw_per_link = {}
        for link_idx in self.corridor_links:
            cap_bits = bin[offset:offset + n_cap_bins]
            c_idx    = int(np.argmax(cap_bits)) if any(cap_bits) else -1
            cap_raw_per_link[link_idx] = (
                self.multilink_cap_bins[c_idx] if c_idx >= 0 else self.neutral_cap_raw
            )
            offset += n_cap_bins

        # Build x_all
        n_links     = self.static_features.shape[0]
        feature_dim = self.capacity_idx + 1
        x_all       = np.zeros((n_links, feature_dim), dtype=np.float32)
        x_all[:, :self.hour_start] = self.static_features

        neutral_scaled = float(
            self.capacity_scaler.transform(
                np.array([[self.neutral_cap_raw]], dtype=np.float32)
            )[0, 0]
        )
        x_all[:, self.capacity_idx]          = neutral_scaled
        x_all[:, self.hour_start:self.hour_end] = hour_onehot

        for link_idx, cap_raw in cap_raw_per_link.items():
            x_all[link_idx, self.capacity_idx] = float(
                self.capacity_scaler.transform(
                    np.array([[cap_raw]], dtype=np.float32)
                )[0, 0]
            )

        x_all_t       = torch.as_tensor(x_all, dtype=torch.float32)
        out_links_t   = torch.tensor(self.output_links, dtype=torch.long)
        global_attr   = _build_global_attr(
            x_all_t, self.hour_start, self.hour_end, self.capacity_idx
        )
        x_nodes = (
            x_all_t if getattr(self.model, 'uses_dynamic_nodes', False)
            else self.static_features_tensor
        )

        with torch.no_grad():
            y_scaled = self.model(
                x_nodes, self.edge_index_tensor, self.edge_attr_tensor,
                global_attr, out_links_t, x_all_t[out_links_t],
            )

        speeds_raw = self.speed_scaler.inverse_transform(
            y_scaled.detach().numpy().reshape(-1, 1)
        ).reshape(-1)

        result = []
        for spd in speeds_raw:
            result += [1, 0] if spd < self.congestion_threshold_raw else [0, 1]
        return result

    # ------------------------------------------------------------------
    def generate_sample(self):
        # Random inputs (all-zero = neutral allowed)
        vec_in = sum(
            [getRandomValue(self.intepretor.lengths[a], True) for a in self._input_attrs],
            []
        )
        # Random joint label — each output section gets exactly one bit set
        labelVec = sum(
            [getRandomValue(self.intepretor.lengths[a], False) for a in self._output_attrs],
            []
        )
        vec_full     = vec_in + labelVec
        predLabelVec = self.probe(vec_full)

        if predLabelVec == [0] * self._n_output_bits:
            self.bad_nc.append(vec_full)

        return (vec_full, labelVec == predLabelVec)

    # ------------------------------------------------------------------
    def MQ(self, assignment):
        vec, labelVec = assignment[:-self._n_output_bits], assignment[-self._n_output_bits:]
        return list(labelVec) == self.probe(assignment)

