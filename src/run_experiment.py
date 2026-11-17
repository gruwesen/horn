import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score


class EdgeBlock(nn.Module):
    def __init__(self, edge_dim, node_dim, global_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, edge_attr, x_s, x_t, global_attr):
        out = torch.cat([edge_attr, x_s, x_t, global_attr], dim=-1)
        return self.edge_mlp(out)


class NodeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_agg, global_attr):
        out = torch.cat([x, edge_agg, global_attr], dim=-1)
        return self.node_mlp(out)


class GlobalBlock(nn.Module):
    def __init__(self, global_dim, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, global_attr, node_agg, edge_agg):
        out = torch.cat([global_attr, node_agg, edge_agg], dim=-1)
        return self.global_mlp(out)


class GraphNetworkRegressorTunable(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, dropout, output_lower_bound=None):
        super().__init__()
        self.output_lower_bound = float(output_lower_bound) if output_lower_bound is not None else None
        self.uses_dynamic_nodes = True
        self.edge_block = EdgeBlock(edge_dim, node_dim, global_dim, hidden_dim)
        self.node_block = NodeBlock(node_dim, hidden_dim, global_dim, hidden_dim)
        self.global_block = GlobalBlock(global_dim, hidden_dim, hidden_dim, hidden_dim)

        # Head takes node embedding + global embedding; x_samples no longer needed
        # because capacity and hour are baked into x_nodes_dyn at training time.
        self.sample_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def scatter_mean(src, index, dim_size):
        out = torch.zeros(dim_size, src.size(1), device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        out.index_add_(0, index, src)
        ones = torch.ones(index.size(0), 1, device=src.device)
        count.index_add_(0, index, ones)
        return out / count.clamp(min=1.0)

    def compute_graph_embeddings(self, x, edge_index, edge_attr, global_attr):
        row, col = edge_index
        global_edge = global_attr.repeat(edge_attr.size(0), 1)
        edge_out = self.edge_block(edge_attr, x[row], x[col], global_edge)

        edge_agg = self.scatter_mean(edge_out, row, x.size(0))
        global_node = global_attr.repeat(x.size(0), 1)
        node_out = self.node_block(x, edge_agg, global_node)

        node_agg = node_out.mean(dim=0, keepdim=True)
        edge_agg_global = edge_out.mean(dim=0, keepdim=True)
        global_out = self.global_block(global_attr, node_agg, edge_agg_global)
        return edge_out, node_out, global_out

    def forward(self, x_nodes, edge_index, edge_attr, global_attr, link_indices, x_samples):
        _, node_out, global_out = self.compute_graph_embeddings(x_nodes, edge_index, edge_attr, global_attr)
        selected_node_emb = node_out[link_indices]
        selected_global = global_out.expand(link_indices.size(0), -1)
        out = self.sample_head(torch.cat([selected_node_emb, selected_global], dim=1))
        if self.output_lower_bound is not None:
            lb = out.new_tensor(self.output_lower_bound)
            out = lb + F.softplus(out - lb)
        return out


class GCNIIConvLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h, h0, adj_norm, alpha, beta):
        propagated = torch.sparse.mm(adj_norm, h)
        support = (1.0 - alpha) * propagated + alpha * h0
        transformed = self.lin(support)
        return (1.0 - beta) * support + beta * transformed


class GCNIIRegressor(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_layers=8, alpha=0.1, lamda=0.5, dropout=0.2, output_lower_bound=None):
        super().__init__()
        self.output_lower_bound = float(output_lower_bound) if output_lower_bound is not None else None
        self.uses_dynamic_nodes = True
        self.alpha = float(alpha)
        self.lamda = float(lamda)
        self.dropout = nn.Dropout(dropout)

        # node_dim = static_dim + hour_dim + 1 (capacity).
        # Hour is baked into x_nodes_dyn so every GCNII layer propagates
        # time-of-day-weighted capacity through the neighbourhood.
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNIIConvLayer(hidden_dim) for _ in range(n_layers)])

        # Head reads directly from node embeddings; no separate sample features.
        self.sample_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def build_adj_norm(edge_index, n_nodes, device):
        row, col = edge_index
        row_all = torch.cat([row, col], dim=0)
        col_all = torch.cat([col, row], dim=0)

        self_idx = torch.arange(n_nodes, device=device)
        row_all = torch.cat([row_all, self_idx], dim=0)
        col_all = torch.cat([col_all, self_idx], dim=0)

        indices = torch.stack([row_all, col_all], dim=0)
        values = torch.ones(indices.size(1), device=device)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes), device=device).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)

        r, c = adj.indices()
        v = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
        return torch.sparse_coo_tensor(adj.indices(), v, size=adj.size(), device=device).coalesce()

    def compute_node_embeddings(self, x_nodes, edge_index):
        n_nodes = x_nodes.size(0)
        # Build adj_norm once and cache — the graph topology never changes.
        if (not hasattr(self, '_adj_norm_cache')
                or self._adj_norm_cache.device != x_nodes.device
                or self._adj_norm_cache.size(0) != n_nodes):
            self._adj_norm_cache = self.build_adj_norm(edge_index, n_nodes=n_nodes, device=x_nodes.device)
        adj_norm = self._adj_norm_cache

        h0 = F.relu(self.input_proj(x_nodes))
        h = self.dropout(h0)

        for layer_idx, layer in enumerate(self.layers, start=1):
            beta = np.log(self.lamda / float(layer_idx) + 1.0)
            h = layer(h, h0, adj_norm, self.alpha, beta)
            h = F.relu(h)
            h = self.dropout(h)

        return h

    def forward(self, x_nodes, edge_index, edge_attr, global_attr, link_indices, x_samples):
        h = self.compute_node_embeddings(x_nodes, edge_index)
        out = self.sample_head(h[link_indices])
        if self.output_lower_bound is not None:
            lb = out.new_tensor(self.output_lower_bound)
            out = lb + F.softplus(out - lb)
        return out


class BasicMLPRegressor(nn.Module):
    def __init__(self, sample_dim, hidden_dim=128, dropout=0.2, output_lower_bound=None):
        super().__init__()
        self.output_lower_bound = float(output_lower_bound) if output_lower_bound is not None else None
        self.net = nn.Sequential(
            nn.Linear(sample_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_nodes, edge_index, edge_attr, global_attr, link_indices, x_samples):
        out = self.net(x_samples)
        if self.output_lower_bound is not None:
            lb = out.new_tensor(self.output_lower_bound)
            out = lb + F.softplus(out - lb)
        return out


class SimpleGNNRegressor(nn.Module):
    def __init__(self, node_dim, sample_dim, hidden_dim=128, n_layers=2, dropout=0.2, output_lower_bound=None):
        super().__init__()
        self.output_lower_bound = float(output_lower_bound) if output_lower_bound is not None else None
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(max(1, int(n_layers)))])
        self.dropout = nn.Dropout(dropout)

        self.sample_head = nn.Sequential(
            nn.Linear(hidden_dim + sample_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def build_adj_norm(edge_index, n_nodes, device):
        row, col = edge_index
        row_all = torch.cat([row, col], dim=0)
        col_all = torch.cat([col, row], dim=0)

        self_idx = torch.arange(n_nodes, device=device)
        row_all = torch.cat([row_all, self_idx], dim=0)
        col_all = torch.cat([col_all, self_idx], dim=0)

        indices = torch.stack([row_all, col_all], dim=0)
        values = torch.ones(indices.size(1), device=device)
        adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes), device=device).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)

        r, c = adj.indices()
        v = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
        return torch.sparse_coo_tensor(adj.indices(), v, size=adj.size(), device=device).coalesce()

    def compute_node_embeddings(self, x_nodes, edge_index):
        h = F.relu(self.input_proj(x_nodes))
        h = self.dropout(h)

        n_nodes = x_nodes.size(0)
        if (not hasattr(self, '_adj_norm_cache')
                or self._adj_norm_cache.device != x_nodes.device
                or self._adj_norm_cache.size(0) != n_nodes):
            self._adj_norm_cache = self.build_adj_norm(edge_index, n_nodes=n_nodes, device=x_nodes.device)
        adj_norm = self._adj_norm_cache
        for conv in self.convs:
            h = torch.sparse.mm(adj_norm, h)
            h = conv(h)
            h = F.relu(h)
            h = self.dropout(h)

        return h

    def forward(self, x_nodes, edge_index, edge_attr, global_attr, link_indices, x_samples):
        h = self.compute_node_embeddings(x_nodes, edge_index)
        out = self.sample_head(torch.cat([h[link_indices], x_samples], dim=1))
        if self.output_lower_bound is not None:
            lb = out.new_tensor(self.output_lower_bound)
            out = lb + F.softplus(out - lb)
        return out


class LinearRegressionRegressor(nn.Module):
    def __init__(self, sample_dim, output_lower_bound=None):
        super().__init__()
        self.output_lower_bound = float(output_lower_bound) if output_lower_bound is not None else None
        self.linear = nn.Linear(sample_dim, 1)

    def forward(self, x_nodes, edge_index, edge_attr, global_attr, link_indices, x_samples):
        out = self.linear(x_samples)
        if self.output_lower_bound is not None:
            lb = out.new_tensor(self.output_lower_bound)
            out = lb + F.softplus(out - lb)
        return out


def parse_args():
    here = Path(__file__).resolve().parent
    default_dataset = here.parent / "dataset_1.pkl"
    default_output = here.parent / "runs" / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"

    parser = argparse.ArgumentParser(description="Run full experiment with best hyperparameters")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument(
        "--input-variant",
        type=str,
        default="auto",
        choices=["auto", "default", "with_neigh", "without_neigh"],
        help="Which dataset tensor variant to use",
    )
    parser.add_argument("--hparams", type=Path, required=True, help="Path to best_hparams.json from search")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma list (e.g. GN,GCNII,MLP,SIMPLEGNN,LINREG). Default: hparams models + baseline models",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Train each model this many times with seeds seed, seed+1, ... and keep the best by val loss.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-model-state", action="store_true")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_data(dataset_path, device):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    return data


def resolve_hour_dim(data):
    simple_hours = data.get("simple_hours")
    if simple_hours is None:
        raise KeyError("Dataset missing 'simple_hours'; cannot resolve hour feature width safely")
    hour_dim = int(len(simple_hours))
    if hour_dim <= 0:
        raise ValueError(f"Invalid hour feature width from simple_hours: {hour_dim}")
    return hour_dim


def resolve_variant_keys(data, variant):
    if variant == "with_neigh":
        return "X_train_with_neigh", "X_val_with_neigh", "X_test_with_neigh"
    if variant == "without_neigh":
        return "X_train_without_neigh", "X_val_without_neigh", "X_test_without_neigh"
    if variant == "default":
        return "X_train", "X_val", "X_test"

    # auto
    if "X_train_with_neigh" in data and "X_val_with_neigh" in data and "X_test_with_neigh" in data:
        return "X_train_with_neigh", "X_val_with_neigh", "X_test_with_neigh"
    return "X_train", "X_val", "X_test"


def resolve_target_scaler(data, dataset_path):
    dataset_path = Path(dataset_path)

    def _load_sidecar_scaler(filename):
        sidecar = dataset_path.parent / filename
        if not sidecar.exists():
            return None
        with sidecar.open("rb") as f:
            return pickle.load(f)

    target_column = data.get("target_column", "flow_scaled")
    if target_column == "speed_scaled" and "speed_scaler" in data:
        return target_column, data["speed_scaler"]
    if target_column == "flow_scaled" and "flow_scaler" in data:
        return target_column, data["flow_scaler"]

    # Backwards/sidecar compatibility: newer dataset builders persist scalers
    # as sibling .pkl files rather than embedding them in the dataset object.
    if target_column == "speed_scaled":
        sidecar = _load_sidecar_scaler("speed_scaler.pkl")
        if sidecar is not None:
            return target_column, sidecar
    if target_column == "flow_scaled":
        sidecar = _load_sidecar_scaler("flow_scaler.pkl")
        if sidecar is not None:
            return target_column, sidecar

    if "speed_scaler" in data:
        return "speed_scaled", data["speed_scaler"]
    if "flow_scaler" in data:
        return "flow_scaled", data["flow_scaler"]

    speed_sidecar = _load_sidecar_scaler("speed_scaler.pkl")
    if speed_sidecar is not None:
        return "speed_scaled", speed_sidecar
    flow_sidecar = _load_sidecar_scaler("flow_scaler.pkl")
    if flow_sidecar is not None:
        return "flow_scaled", flow_sidecar

    raise KeyError("Could not resolve target scaler. Expected one of speed_scaler or flow_scaler")


def build_tensors(data, device, variant):
    x_train_key, x_val_key, x_test_key = resolve_variant_keys(data, variant)

    for key in [x_train_key, x_val_key, x_test_key, "y_train", "y_val", "y_test", "link_indices_train", "link_indices_val", "link_indices_test", "static_features_normalized", "edge_index"]:
        if key not in data:
            raise KeyError(f"Dataset missing required key: {key}")

    if "edge_attr" in data:
        edge_attr = np.asarray(data["edge_attr"], dtype=np.float32)
    else:
        # Backward compatibility for older datasets without edge attributes.
        edge_attr = np.ones((int(np.asarray(data["edge_index"]).shape[1]), 1), dtype=np.float32)

    tensors = {
        "X_train_t": torch.FloatTensor(data[x_train_key]).to(device),
        "y_train_t": torch.FloatTensor(data["y_train"].reshape(-1, 1)).to(device),
        "X_val_t": torch.FloatTensor(data[x_val_key]).to(device),
        "y_val_t": torch.FloatTensor(data["y_val"].reshape(-1, 1)).to(device),
        "X_test_t": torch.FloatTensor(data[x_test_key]).to(device),
        "y_test_t": torch.FloatTensor(data["y_test"].reshape(-1, 1)).to(device),
        "static_features_tensor": torch.FloatTensor(data["static_features_normalized"]).to(device),
        "edge_index_tensor": torch.LongTensor(data["edge_index"]).to(device),
        "edge_attr_tensor": torch.FloatTensor(edge_attr).to(device),
        "link_train_t": torch.LongTensor(data["link_indices_train"]).to(device),
        "link_val_t": torch.LongTensor(data["link_indices_val"]).to(device),
        "link_test_t": torch.LongTensor(data["link_indices_test"]).to(device),
    }
    return tensors, x_train_key, x_val_key, x_test_key


def build_global_attr(x_batch, hour_start, hour_end, capacity_idx):
    hour_profile = x_batch[:, hour_start:hour_end].mean(dim=0, keepdim=True)
    mean_capacity = x_batch[:, capacity_idx : capacity_idx + 1].mean(dim=0, keepdim=True)
    return torch.cat([hour_profile, mean_capacity], dim=1)


def build_model(model_key, cfg, static_dim, hour_dim, sample_dim, edge_dim, device, output_lower_bound=None):
    if model_key == "GN":
        # node_dim = sample_dim: the full per-link feature vector (static + hour + capacity,
        # plus neighbour features for the with_neigh variant) is passed as x_nodes so that
        # edge messages carry each link's dynamic state through the graph.
        return GraphNetworkRegressorTunable(
            node_dim=sample_dim,
            edge_dim=edge_dim,
            global_dim=hour_dim + 1,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            output_lower_bound=output_lower_bound,
        ).to(device)

    if model_key == "GCNII":
        # node_dim = sample_dim for the same reason: covers both variants.
        return GCNIIRegressor(
            node_dim=sample_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            alpha=float(cfg["alpha"]),
            lamda=float(cfg["lamda"]),
            dropout=float(cfg["dropout"]),
            output_lower_bound=output_lower_bound,
        ).to(device)

    if model_key == "MLP":
        return BasicMLPRegressor(
            sample_dim=sample_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            output_lower_bound=output_lower_bound,
        ).to(device)

    if model_key == "SIMPLEGNN":
        return SimpleGNNRegressor(
            node_dim=static_dim,
            sample_dim=sample_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg["dropout"]),
            output_lower_bound=output_lower_bound,
        ).to(device)

    if model_key == "LINREG":
        return LinearRegressionRegressor(sample_dim=sample_dim, output_lower_bound=output_lower_bound).to(device)

    raise ValueError(f"Unsupported model: {model_key}")


def default_cfg_for_model(model_key):
    defaults = {
        "MLP": {
            "hidden_dim": 128,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 20,
            "patience": 3,
        },
        "SIMPLEGNN": {
            "hidden_dim": 128,
            "n_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 20,
            "patience": 3,
        },
        "LINREG": {
            "lr": 5e-3,
            "weight_decay": 0.0,
            "batch_size": 1024,
            "epochs": 20,
            "patience": 3,
        },
    }
    return defaults.get(model_key, None)


def train_one_model(model_key, cfg, tensors, target_scaler, hour_start, hour_end, capacity_idx, device):
    X_train_t = tensors["X_train_t"]
    y_train_t = tensors["y_train_t"]
    X_val_t = tensors["X_val_t"]
    y_val_t = tensors["y_val_t"]
    X_test_t = tensors["X_test_t"]
    y_test_t = tensors["y_test_t"]
    static_features_tensor = tensors["static_features_tensor"]
    edge_index_tensor = tensors["edge_index_tensor"]
    edge_attr_tensor = tensors["edge_attr_tensor"]
    link_train_t = tensors["link_train_t"]
    link_val_t = tensors["link_val_t"]
    link_test_t = tensors["link_test_t"]

    output_lower_bound = float(-target_scaler.mean_[0] / target_scaler.scale_[0])
    model = build_model(
        model_key=model_key,
        cfg=cfg,
        static_dim=int(static_features_tensor.shape[1]),
        hour_dim=int(hour_end - hour_start),
        sample_dim=int(X_train_t.shape[1]),
        edge_dim=int(edge_attr_tensor.shape[1]),
        device=device,
        output_lower_bound=output_lower_bound,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    edge_attr = edge_attr_tensor

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    patience = int(cfg.get("patience", 3))

    history = []

    if getattr(model, "uses_dynamic_nodes", False):
        # ── Snapshot-based training for GN and GCNII ─────────────────────────
        # Each training step processes one full graph snapshot: all n_links links
        # at a single (scenario, hour) pair.  x_nodes_dyn carries [static | hour |
        # capacity] for every link, so message-passing can propagate capacity
        # changes across the neighbourhood.
        #
        # Dataset layout (verified):
        #   rows_per_scenario = n_links * n_hours  (= 70 * 20 = 1400)
        #   within each scenario block: link i rows = base + i * n_hours + h
        n_links   = static_features_tensor.shape[0]
        n_hours   = hour_end - hour_start
        rows_per_snap = n_links * n_hours
        n_scen_train = X_train_t.shape[0] // rows_per_snap
        n_scen_val   = X_val_t.shape[0]   // rows_per_snap
        n_scen_test  = X_test_t.shape[0]  // rows_per_snap
        # link_block[i] = i * n_hours; add hour offset h to get row for link i
        link_block = torch.arange(n_links, device=device) * n_hours
        all_links  = torch.arange(n_links, device=device)

        for epoch in range(1, int(cfg["epochs"]) + 1):
            model.train()
            perm = torch.randperm(n_scen_train, device=device)
            running_loss = 0.0
            n_batches = 0

            for s in perm:
                s = int(s)
                base = s * rows_per_snap
                for h in range(n_hours):
                    rows    = base + link_block + h
                    x_snap  = X_train_t[rows]      # (n_links, node_dim)
                    y_snap  = y_train_t[rows]      # (n_links, 1)
                    g = build_global_attr(x_snap, hour_start, hour_end, capacity_idx)

                    optimizer.zero_grad()
                    pred = model(x_snap, edge_index_tensor, edge_attr, g, all_links, x_snap)
                    loss = criterion(pred, y_snap)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    running_loss += float(loss.item())
                    n_batches += 1

            train_loss = running_loss / max(1, n_batches)

            model.eval()
            with torch.no_grad():
                val_acc   = 0.0
                val_steps = 0
                for s in range(n_scen_val):
                    base = s * rows_per_snap
                    for h in range(n_hours):
                        rows   = base + link_block + h
                        x_snap = X_val_t[rows]
                        y_snap = y_val_t[rows]
                        g = build_global_attr(x_snap, hour_start, hour_end, capacity_idx)
                        pred = model(x_snap, edge_index_tensor, edge_attr, g, all_links, x_snap)
                        val_acc   += float(criterion(pred, y_snap).item())
                        val_steps += 1
                val_loss = val_acc / max(1, val_steps)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1

            print(f"{model_key} epoch {epoch:02d}/{int(cfg['epochs'])} | train={train_loss:.6f} | val={val_loss:.6f}")

            if bad_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Collect test predictions into a pre-allocated buffer indexed by original
        # row position so the result aligns with y_test_t ordering.
        model.eval()
        with torch.no_grad():
            y_pred_buf = torch.zeros(X_test_t.shape[0], device=device)
            for s in range(n_scen_test):
                base = s * rows_per_snap
                for h in range(n_hours):
                    rows   = base + link_block + h
                    x_snap = X_test_t[rows]
                    g = build_global_attr(x_snap, hour_start, hour_end, capacity_idx)
                    pred = model(x_snap, edge_index_tensor, edge_attr, g, all_links, x_snap)
                    y_pred_buf[rows] = pred.squeeze(1)
            y_pred_scaled = y_pred_buf.cpu().numpy().reshape(-1, 1)

    else:
        # ── Row-based training for MLP, SIMPLEGNN, LINREG ────────────────────
        for epoch in range(1, int(cfg["epochs"]) + 1):
            model.train()
            perm = torch.randperm(X_train_t.size(0), device=device)
            running_loss = 0.0
            n_batches = 0

            for i in range(0, X_train_t.size(0), int(cfg["batch_size"])):
                b = perm[i : i + int(cfg["batch_size"])]
                xb = X_train_t[b]
                yb = y_train_t[b]
                lb = link_train_t[b]
                g = build_global_attr(xb, hour_start, hour_end, capacity_idx)

                optimizer.zero_grad()
                pred = model(static_features_tensor, edge_index_tensor, edge_attr, g, lb, xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += float(loss.item())
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)

            model.eval()
            with torch.no_grad():
                g_val = build_global_attr(X_val_t, hour_start, hour_end, capacity_idx)
                val_pred = model(static_features_tensor, edge_index_tensor, edge_attr, g_val, link_val_t, X_val_t)
                val_loss = float(criterion(val_pred, y_val_t).item())

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1

            print(f"{model_key} epoch {epoch:02d}/{int(cfg['epochs'])} | train={train_loss:.6f} | val={val_loss:.6f}")

            if bad_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            g_test = build_global_attr(X_test_t, hour_start, hour_end, capacity_idx)
            y_pred_scaled = model(
                static_features_tensor,
                edge_index_tensor,
                edge_attr,
                g_test,
                link_test_t,
                X_test_t,
            ).cpu().numpy().reshape(-1, 1)

    y_test_scaled = y_test_t.cpu().numpy().reshape(-1, 1)
    y_true = target_scaler.inverse_transform(y_test_scaled).reshape(-1)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "model_family": model_key,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_val_loss": best_val,
        "epochs_ran": len(history),
    }

    return model, pd.DataFrame(history), metrics, y_true, y_pred


def plot_curves(history_by_model, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_key, hist_df in history_by_model.items():
        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label=model_key)
        axes[1].plot(hist_df["epoch"], hist_df["val_loss"], label=model_key)

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].grid(alpha=0.2)

    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE loss")
    axes[1].grid(alpha=0.2)

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Hyperparameters file: {args.hparams}")
    print(f"Output dir: {args.output_dir}")

    with open(args.hparams, "r", encoding="utf-8") as f:
        best_hparams_by_model = json.load(f)

    if args.models.strip():
        model_list = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    else:
        model_list = [m.upper() for m in best_hparams_by_model.keys()]
        for extra_model in ["MLP", "SIMPLEGNN", "LINREG"]:
            if extra_model not in model_list:
                model_list.append(extra_model)

    data = load_data(args.dataset, device)
    tensors, x_train_key, x_val_key, x_test_key = build_tensors(data, device, variant=args.input_variant)
    target_column, target_scaler = resolve_target_scaler(data, args.dataset)

    static_dim = int(tensors["static_features_tensor"].shape[1])
    hour_dim = resolve_hour_dim(data)
    if hour_dim <= 0:
        raise ValueError(f"Invalid derived hour_dim: {hour_dim}")

    hour_start = static_dim
    hour_end = hour_start + hour_dim
    capacity_idx = hour_end

    metrics_rows = []
    history_by_model = {}

    for model_key in model_list:
        hp = best_hparams_by_model.get(model_key)
        if hp is not None:
            cfg = {
                "lr": float(hp["lr"]),
                "weight_decay": float(hp["weight_decay"]),
                "batch_size": int(round(float(hp["batch_size"]))),
                "epochs": int(round(float(hp["epochs"]))),
                "patience": int(round(float(hp.get("patience", 3)))),
            }

            if model_key in {"GN", "GCNII", "MLP", "SIMPLEGNN"}:
                cfg["hidden_dim"] = int(round(float(hp["hidden_dim"])))
                cfg["dropout"] = float(hp["dropout"])

            if model_key == "GCNII":
                cfg["n_layers"] = int(round(float(hp["n_layers"])))
                cfg["alpha"] = float(hp["alpha"])
                cfg["lamda"] = float(hp["lamda"])

            if model_key == "SIMPLEGNN":
                cfg["n_layers"] = int(round(float(hp.get("n_layers", 2))))
        else:
            cfg = default_cfg_for_model(model_key)
            if cfg is None:
                raise ValueError(f"Missing hyperparameters for model '{model_key}' in {args.hparams}")

        print(f"\n=== Running {model_key} ===")
        print(f"Config: {cfg}")

        seeds = [args.seed + i for i in range(args.n_seeds)]
        best_seed_result = None  # (model, history_df, metrics, y_true, y_pred, seed_val)

        for seed_val in seeds:
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)

            model_s, history_s, metrics_s, y_true_s, y_pred_s = train_one_model(
                model_key=model_key,
                cfg=cfg,
                tensors=tensors,
                target_scaler=target_scaler,
                hour_start=hour_start,
                hour_end=hour_end,
                capacity_idx=capacity_idx,
                device=device,
            )

            # Always save per-seed artifacts.
            history_s.to_csv(
                args.output_dir / f"history_{model_key.lower()}_seed{seed_val}.csv", index=False
            )
            np.savez(
                args.output_dir / f"predictions_{model_key.lower()}_seed{seed_val}.npz",
                model_family=model_key,
                model_name=f"MapGNN_{model_key}_BestHP_seed{seed_val}",
                y_true=y_true_s.astype(np.float32),
                y_pred=y_pred_s.astype(np.float32),
                link_indices_test=np.asarray(data["link_indices_test"]).astype(np.int64),
                target_column=target_column,
                input_variant=args.input_variant,
                seed=seed_val,
            )
            if args.save_model_state:
                torch.save(
                    model_s.state_dict(),
                    args.output_dir / f"model_state_{model_key.lower()}_seed{seed_val}.pt",
                )

            print(
                f"{model_key} seed {seed_val} | val_loss={metrics_s['best_val_loss']:.6f} "
                f"| R2={metrics_s['r2']:.4f} | RMSE={metrics_s['rmse']:.3f}"
            )

            if best_seed_result is None or metrics_s["best_val_loss"] < best_seed_result[2]["best_val_loss"]:
                best_seed_result = (model_s, history_s, metrics_s, y_true_s, y_pred_s, seed_val)

        model, history_df, metrics, y_true, y_pred, best_seed = best_seed_result
        print(f"{model_key} best seed: {best_seed} | val_loss={metrics['best_val_loss']:.6f}")

        # Save canonical outputs (used by downstream scripts) from the best seed.
        history_path = args.output_dir / f"history_{model_key.lower()}.csv"
        history_df.to_csv(history_path, index=False)
        history_by_model[model_key] = history_df

        pred_path = args.output_dir / f"predictions_{model_key.lower()}.npz"
        np.savez(
            pred_path,
            model_family=model_key,
            model_name=f"MapGNN_{model_key}_BestHP",
            y_true=y_true.astype(np.float32),
            y_pred=y_pred.astype(np.float32),
            link_indices_test=np.asarray(data["link_indices_test"]).astype(np.int64),
            target_column=target_column,
            input_variant=args.input_variant,
            seed=best_seed,
        )

        if args.save_model_state:
            torch.save(model.state_dict(), args.output_dir / f"model_state_{model_key.lower()}.pt")

        row = {**metrics, "best_seed": best_seed, "n_seeds": args.n_seeds,
               "prediction_file": pred_path.name, "history_file": history_path.name}
        metrics_rows.append(row)

        print(f"{model_key} test: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["r2", "rmse", "mae"], ascending=[False, True, True])
    metrics_csv = args.output_dir / "experiment_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    with open(args.output_dir / "used_hparams.json", "w", encoding="utf-8") as f:
        used_hparams = {}
        for m in model_list:
            used_hparams[m] = best_hparams_by_model.get(m, default_cfg_for_model(m))
        json.dump(used_hparams, f, indent=2)

    curve_path = args.output_dir / "training_curves_all_models.png"
    plot_curves(history_by_model, curve_path)

    run_meta = {
        "dataset": str(args.dataset.resolve()),
        "hparams_file": str(args.hparams.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "input_variant": args.input_variant,
        "resolved_x_train_key": x_train_key,
        "resolved_x_val_key": x_val_key,
        "resolved_x_test_key": x_test_key,
        "target_column": target_column,
        "models": model_list,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "seeds_used": list(range(args.seed, args.seed + args.n_seeds)),
        "device": str(device),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("\nSaved:")
    print(f"- {metrics_csv}")
    print(f"- {curve_path}")
    for model_key in model_list:
        print(f"- {args.output_dir / f'predictions_{model_key.lower()}.npz'}")
        print(f"- {args.output_dir / f'history_{model_key.lower()}.csv'}")


if __name__ == "__main__":
    main()
