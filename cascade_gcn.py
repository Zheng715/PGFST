import random
import sys
import os
import warnings

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from config import Config

warnings.filterwarnings("ignore")

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_deterministic(42)


class DeterministicGCNConfig:
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATES = [0.5, 0.2, 0.4]
    SIMILARITY_TYPE = "cosine"
    TOP_K_SPARSE = 24
    LOW_DIM_FOR_A = 256
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-05
    CLIP_GRAD_NORM = 1.3
    ALPHA = 0.01
    BETA = 4.0
    GAMMA = 1.0
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 20


class DeterministicGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, adj):
        x = self.dropout(x)
        support = self.linear(x)
        support = self.bn(support)
        output = torch.mm(adj, support)
        return F.relu(output)


class DeterministicGraphDecoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.LOW_DIM_FOR_A)
        )

        self.projection = nn.Sequential(
            nn.Linear(config.LOW_DIM_FOR_A, config.LOW_DIM_FOR_A),
            nn.ReLU(),
            nn.Linear(config.LOW_DIM_FOR_A, config.LOW_DIM_FOR_A)
        )

        self._initialize_weights_deterministic()

    def forward(self, x):
        z = self.decoder(x)
        z = self.projection(z)

        if self.config.SIMILARITY_TYPE == "cosine":
            z_norm = F.normalize(z, p=2, dim=1)
            sim_matrix = torch.mm(z_norm, z_norm.T)
            adj_A = (sim_matrix + 1) / 2
        else:
            dists = torch.cdist(z, z, p=2)
            sim_matrix = 1 - dists / (dists.max() + 1e-08)
            adj_A = sim_matrix

        k = self.config.TOP_K_SPARSE
        topk_vals, topk_idx = torch.topk(adj_A, k=k, dim=1)

        sparse_adj = torch.zeros_like(adj_A)
        sparse_adj.scatter_(1, topk_idx, topk_vals)
        sparse_adj = (sparse_adj + sparse_adj.T) / 2

        return sparse_adj

    def _initialize_weights_deterministic(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.8)
                nn.init.constant_(m.bias, 0)


class DeterministicCascadeGCN(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.config = config

        self.gcn1 = DeterministicGCNLayer(
            input_dim,
            config.HIDDEN_DIMS[0],
            config.DROPOUT_RATES[0]
        )

        self.gcn2 = DeterministicGCNLayer(
            config.HIDDEN_DIMS[0],
            config.HIDDEN_DIMS[1],
            config.DROPOUT_RATES[1]
        )

        self.gcn3 = DeterministicGCNLayer(
            config.HIDDEN_DIMS[1],
            config.HIDDEN_DIMS[2],
            config.DROPOUT_RATES[2]
        )

        self.reduce1 = nn.Linear(config.HIDDEN_DIMS[0], 128)
        self.reduce2 = nn.Linear(config.HIDDEN_DIMS[1], 128)

        self.fusion_norm = nn.BatchNorm1d(
            config.HIDDEN_DIMS[2] + 256
        )

        self.classifier = nn.Linear(
            config.HIDDEN_DIMS[2] + 256,
            output_dim
        )

        self.graph_decoder = DeterministicGraphDecoder(
            config.HIDDEN_DIMS[2] + 256,
            config
        )

        self._initialize_weights_deterministic()

    def forward(self, x, adj1, adj2, adj3):
        h1 = self.gcn1(x, adj1)
        h2 = self.gcn2(h1, adj2)
        h3 = self.gcn3(h2, adj3)

        h1_reduced = F.relu(self.reduce1(h1))
        h2_reduced = F.relu(self.reduce2(h2))

        fused = torch.cat(
            [h3, h1_reduced, h2_reduced],
            dim=1
        )

        fused = self.fusion_norm(fused)
        adj_A = self.graph_decoder(fused)

        return fused, adj_A

    def _initialize_weights_deterministic(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)


class DeterministicTrainer:
    def __init__(self, config, model_config=None):
        if model_config is None:
            model_config = DeterministicGCNConfig()

        self.config = config
        self.model_config = model_config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_data(self):
        fused_features = pd.read_csv(
            f"{self.config.OUTPUT_DIR}/fused_features.csv",
            header=None
        ).values

        X = torch.FloatTensor(fused_features).to(self.device)

        adj_gene = pd.read_csv(
            f"{self.config.OUTPUT_DIR}/adj_gene.csv",
            header=None
        ).values

        adj_spatial = pd.read_csv(
            f"{self.config.OUTPUT_DIR}/adj_spot.csv",
            header=None
        ).values

        adj_image = pd.read_csv(
            f"{self.config.OUTPUT_DIR}/adj_image.csv",
            header=None
        ).values

        A1 = torch.FloatTensor(adj_gene).to(self.device)
        A2 = torch.FloatTensor(adj_spatial).to(self.device)
        A3 = torch.FloatTensor(adj_image).to(self.device)

        metadata = pd.read_csv(
            f"{self.config.DATA_ROOT}/metadata.tsv",
            sep="\t"
        )

        true_labels = None
        valid_indices = None

        if "layer_guess_reordered" in metadata.columns:
            valid_mask = (
                (metadata["layer_guess_reordered"] != "NA")
                & metadata["layer_guess_reordered"].notna()
            )

            valid_indices = np.where(valid_mask)[0]
            le = LabelEncoder()

            true_labels = le.fit_transform(
                metadata.loc[
                    valid_mask,
                    "layer_guess_reordered"
                ].values
            )

        if valid_indices is not None and len(valid_indices) > 0:
            X = X[valid_indices]
            A1 = A1[valid_indices][:, valid_indices]
            A2 = A2[valid_indices][:, valid_indices]
            A3 = A3[valid_indices][:, valid_indices]

        N = X.shape[0]

        return X, A1, A2, A3, true_labels, N

    def _compute_deterministic_loss(self, adj_A, A1, A2, A3, epoch):
        progress = min(
            epoch / self.model_config.WARMUP_EPOCHS,
            1.0
        )

        decay_factor = 1.0 - progress * 0.5

        diff_loss1 = (
            F.mse_loss(adj_A, A1)
            * self.model_config.ALPHA
        )

        diff_loss2 = (
            F.mse_loss(adj_A, A2)
            * self.model_config.BETA
        )

        diff_loss3 = (
            F.mse_loss(adj_A, A3)
            * self.model_config.GAMMA
        )

        total_diff = (
            diff_loss1 + diff_loss2 + diff_loss3
        ) * decay_factor

        identity = torch.eye(
            adj_A.size(0),
            device=self.device
        )

        graph_reg = (
            F.mse_loss(
                torch.mm(adj_A, adj_A.T),
                identity
            ) * 0.1
        )

        total_loss = total_diff + graph_reg

        return total_loss, total_diff.item(), graph_reg.item()

    def KMeansPlusPlus(self, Z, K, tol=1e-06, seed=3):
        NCells, Dim = Z.shape
        SetSeed(seed)

        Centers = torch.zeros(K, Dim, device=Device)
        Centers[0] = Z[torch.randint(NCells, (1,))]

        for i in range(1, K):
            Dists = torch.cdist(Z, Centers[:i])
            MinDists = Dists.min(dim=1)[0]
            Probs = MinDists.pow(2)
            Probs = Probs / Probs.sum()

            Centers[i] = Z[torch.multinomial(Probs, 1)]
            del Dists, MinDists, Probs

        while True:
            Dist = torch.cdist(Z, Centers)
            labels = Dist.argmin(1)

            NewCenters = torch.stack(
                [
                    Z[labels == k].mean(0)
                    for k in range(K)
                ],
                dim=0
            )

            for k in range(K):
                if (labels == k).sum() == 0:
                    NewCenters[k] = Z[
                        torch.randint(NCells, (1,))
                    ]

            if torch.allclose(Centers, NewCenters, atol=tol):
                del Dist, NewCenters
                break

            Centers = NewCenters
            del Dist, NewCenters

        return labels

    def _compute_all_metrics(self, labels, true_labels):
        if true_labels is None:
            return {
                "ARI": np.nan,
                "NMI": np.nan
            }

        return {
            "ARI": adjusted_rand_score(true_labels, labels),
            "NMI": normalized_mutual_info_score(true_labels, labels)
        }

    def train_model(self, n_clusters=5, run_seed=42):
        set_deterministic(run_seed)

        X, A1, A2, A3, true_labels, N = self.load_data()
        input_dim = X.shape[1]

        model = DeterministicCascadeGCN(
            input_dim,
            n_clusters,
            self.model_config
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.model_config.LEARNING_RATE,
            weight_decay=self.model_config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=80,
            eta_min=1e-06
        )

        for epoch in tqdm(
            range(self.model_config.NUM_EPOCHS),
            desc="训练进度",
            colour="red",
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
            mininterval=0.0
        ):
            model.train()

            features, adj_A = model(X, A1, A2, A3)

            total_loss, diff_loss, reg_loss = (
                self._compute_deterministic_loss(
                    adj_A, A1, A2, A3, epoch
                )
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.model_config.CLIP_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()

        model.eval()

        with torch.no_grad():
            final_features_tensor, _ = model(X, A1, A2, A3)

            final_labels_tensor = self.KMeansPlusPlus(
                final_features_tensor,
                n_clusters,
                tol=1e-06,
                seed=3407
            )

        final_labels = final_labels_tensor.cpu().numpy()
        final_features = final_features_tensor.cpu().numpy().copy()

        final_metrics = self._compute_all_metrics(
            final_labels,
            true_labels
        )

        self._save_results(final_labels, final_features)

        return final_labels, final_features, final_metrics

    def _save_results(self, labels, features):
        output_file = (
            f"{self.config.OUTPUT_DIR}/"
            "deterministic_clustering_results.csv"
        )

        results_df = pd.DataFrame({
            "spot_index": range(len(labels)),
            "cluster_label": labels
        })

        results_df.to_csv(output_file, index=False)

        features_file = (
            f"{self.config.OUTPUT_DIR}/"
            "deterministic_clustering_results.npy"
        )

        np.save(features_file, features)


def SetSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
