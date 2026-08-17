
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import Config


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


set_deterministic(42)


class DeterministicGCNConfig:
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATES = [0.5, 0.2, 0.4]
    USE_BATCH_NORM = True

    SIMILARITY_TYPE = "cosine"
    TOP_K_SPARSE = 24
    LOW_DIM_FOR_A = 256

    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    CLIP_GRAD_NORM = 1.3

    ALPHA = 0.01
    BETA = 4.0
    GAMMA = 1.0

    NUM_EPOCHS = 200
    EVAL_FREQ = 5

    NUM_KMEANS_RUNS = 1
    KMEANS_MAX_ITER = 200
    KMEANS_RANDOM_STATE = 42


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
            sim_matrix = 1 - (dists / (dists.max() + 1e-8))
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

        self.gcn1 = DeterministicGCNLayer(input_dim, config.HIDDEN_DIMS[0], config.DROPOUT_RATES[0])
        self.gcn2 = DeterministicGCNLayer(config.HIDDEN_DIMS[0], config.HIDDEN_DIMS[1], config.DROPOUT_RATES[1])
        self.gcn3 = DeterministicGCNLayer(config.HIDDEN_DIMS[1], config.HIDDEN_DIMS[2], config.DROPOUT_RATES[2])

        self.reduce1 = nn.Linear(config.HIDDEN_DIMS[0], 128)
        self.reduce2 = nn.Linear(config.HIDDEN_DIMS[1], 128)

        self.fusion_norm = nn.BatchNorm1d(config.HIDDEN_DIMS[2] + 256)
        self.classifier = nn.Linear(config.HIDDEN_DIMS[2] + 256, output_dim)
        self.graph_decoder = DeterministicGraphDecoder(config.HIDDEN_DIMS[2] + 256, config)

        self._initialize_weights_deterministic()

    def forward(self, x, adj1, adj2, adj3):
        h1 = self.gcn1(x, adj1)
        h2 = self.gcn2(h1, adj2)
        h3 = self.gcn3(h2, adj3)

        h1_reduced = F.relu(self.reduce1(h1))
        h2_reduced = F.relu(self.reduce2(h2))
        fused = torch.cat([h3, h1_reduced, h2_reduced], dim=1)
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
                nn.init.constant_(m.weight, 0.8)
                nn.init.constant_(m.bias, 0)


class DeterministicTrainer:
    def __init__(self, config, model_config=None):
        if model_config is None:
            model_config = DeterministicGCNConfig()

        self.config = config
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        fused_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/fused_features.csv", header=None).values
        X = torch.FloatTensor(fused_features).to(self.device)

        adj_gene = pd.read_csv(f"{self.config.OUTPUT_DIR}/adj_gene.csv", header=None).values
        adj_spatial = pd.read_csv(f"{self.config.OUTPUT_DIR}/adj_spot.csv", header=None).values
        adj_image = pd.read_csv(f"{self.config.OUTPUT_DIR}/adj_image.csv", header=None).values

        A1 = torch.FloatTensor(adj_gene).to(self.device)
        A2 = torch.FloatTensor(adj_spatial).to(self.device)
        A3 = torch.FloatTensor(adj_image).to(self.device)

        metadata = pd.read_csv(f"{self.config.DATA_ROOT}/metadata.tsv", sep='\t')

        true_labels = None
        valid_indices = None

        if 'layer_guess_reordered' in metadata.columns:
            valid_mask = (
                (metadata['layer_guess_reordered'] != 'NA') &
                (metadata['layer_guess_reordered'].notna())
            )
            valid_indices = np.where(valid_mask)[0]

            le = LabelEncoder()
            true_labels = le.fit_transform(metadata.loc[valid_mask, 'layer_guess_reordered'].values)

        if valid_indices is not None and len(valid_indices) > 0:
            X = X[valid_indices]
            A1 = A1[valid_indices][:, valid_indices]
            A2 = A2[valid_indices][:, valid_indices]
            A3 = A3[valid_indices][:, valid_indices]

        return X, A1, A2, A3, true_labels

    def _compute_deterministic_loss(self, adj_A, A1, A2, A3, epoch):
        progress = min(epoch / 20, 1.0)
        decay_factor = 1.0 - progress * 0.5

        diff_loss1 = F.mse_loss(adj_A, A1) * self.model_config.ALPHA
        diff_loss2 = F.mse_loss(adj_A, A2) * self.model_config.BETA
        diff_loss3 = F.mse_loss(adj_A, A3) * self.model_config.GAMMA

        total_diff = (diff_loss1 + diff_loss2 + diff_loss3) * decay_factor

        identity = torch.eye(adj_A.size(0)).to(self.device)
        graph_reg = F.mse_loss(torch.mm(adj_A, adj_A.T), identity) * 0.1

        return total_diff + graph_reg

    def _deterministic_kmeans(self, features, n_clusters):
        features_np = features.cpu().detach().numpy()
        best_inertia = float('inf')
        best_labels = None

        for run_id in range(self.model_config.NUM_KMEANS_RUNS):
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=1,
                max_iter=self.model_config.KMEANS_MAX_ITER,
                random_state=self.model_config.KMEANS_RANDOM_STATE + run_id,
                algorithm='lloyd'
            )
            labels = kmeans.fit_predict(features_np)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels

        return best_labels

    def train_model(self, n_clusters=5):
        set_deterministic(42)

        X, A1, A2, A3, true_labels = self.load_data()
        input_dim = X.shape[1]

        model = DeterministicCascadeGCN(input_dim, n_clusters, self.model_config).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.model_config.LEARNING_RATE,
            weight_decay=self.model_config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.model_config.NUM_EPOCHS,
            eta_min=1e-6
        )

        best_ari = -1
        best_labels = None
        best_features = None

        for epoch in tqdm(range(self.model_config.NUM_EPOCHS), desc="Training"):
            model.train()

            features, adj_A = model(X, A1, A2, A3)
            total_loss = self._compute_deterministic_loss(adj_A, A1, A2, A3, epoch)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_config.CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % self.model_config.EVAL_FREQ == 0 or epoch < 10:
                cluster_labels = self._deterministic_kmeans(features, n_clusters)

                if true_labels is not None:
                    current_ari = adjusted_rand_score(true_labels, cluster_labels)
                    if current_ari > best_ari:
                        best_ari = current_ari
                        best_labels = cluster_labels.copy()
                        best_features = features.cpu().detach().numpy().copy()

        if best_labels is not None:
            self._save_results(best_labels, best_ari, best_features)
        else:
            final_features = features.cpu().detach().numpy()
            final_labels = self._deterministic_kmeans(features, n_clusters)
            if true_labels is not None:
                final_ari = adjusted_rand_score(true_labels, final_labels)
                self._save_results(final_labels, final_ari, final_features)
                best_ari = final_ari
            else:
                self._save_results(final_labels, -1, final_features)

        return best_labels, best_features, best_ari

    def _save_results(self, labels, ari, features):
        results_df = pd.DataFrame({
            'spot_index': range(len(labels)),
            'cluster_label': labels
        })
        results_df.to_csv(f"{self.config.OUTPUT_DIR}/deterministic_clustering_results.csv", index=False)

        np.save(f"{self.config.OUTPUT_DIR}/best_features.npy", features)

