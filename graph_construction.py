
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize




class GraphConstructor:
    def __init__(self, config):
        self.config = config
        self.n_neighbors_gene = 10
        self.n_neighbors_spatial = 30
        self.n_neighbors_image = 10
        self.n_samples = None

    def load_features(self):
        gene_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/gene_features.csv", header=None).values
        spatial_features_2d = pd.read_csv(f"{self.config.OUTPUT_DIR}/raw_spatial_2d.csv", header=None).values
        image_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/image_features.csv", header=None).values

        self.n_samples = {
            'gene': gene_features.shape[0],
            'spatial': spatial_features_2d.shape[0],
            'image': image_features.shape[0]
        }
        return gene_features, spatial_features_2d, image_features

    def compute_gene_adjacency(self, gene_features):
        adj_gene = kneighbors_graph(
            gene_features,
            n_neighbors=self.n_neighbors_gene,
            metric='cosine',
            mode='connectivity',
            include_self=False
        )
        return adj_gene

    def compute_spatial_adjacency(self, spatial_features_6d):
        adj_spot = kneighbors_graph(
            spatial_features_6d,
            n_neighbors=self.n_neighbors_spatial,
            metric='euclidean',
            mode='connectivity',
            include_self=False
        )
        adj_spot = (adj_spot + adj_spot.T) / 2
        adj_spot = adj_spot + sp.eye(adj_spot.shape[0])
        return adj_spot

    def compute_image_adjacency(self, image_features):
        adj_image = kneighbors_graph(
            image_features,
            n_neighbors=self.n_neighbors_image,
            metric='euclidean',
            mode='connectivity',
            include_self=False
        )
        return adj_image

    def normalize_adjacency(self, adj_gene, adj_spot, adj_image):
        A1_normalized = normalize(adj_gene, norm='l1', axis=1)
        A2_normalized = normalize(adj_spot, norm='l1', axis=1)
        A3_normalized = normalize(adj_image, norm='l1', axis=1)
        return A1_normalized, A2_normalized, A3_normalized

    def save_adjacency_matrices(self, A1, A2, A3):
        pd.DataFrame(A1.toarray()).to_csv(f"{self.config.OUTPUT_DIR}/adj_gene.csv", index=False, header=False)
        pd.DataFrame(A2.toarray()).to_csv(f"{self.config.OUTPUT_DIR}/adj_spot.csv", index=False, header=False)
        pd.DataFrame(A3.toarray()).to_csv(f"{self.config.OUTPUT_DIR}/adj_image.csv", index=False, header=False)

        sp.save_npz(f"{self.config.OUTPUT_DIR}/adj_gene.npz", A1)
        sp.save_npz(f"{self.config.OUTPUT_DIR}/adj_spot.npz", A2)
        sp.save_npz(f"{self.config.OUTPUT_DIR}/adj_image.npz", A3)

    def construct_all_graphs(self):
        gene_features, spatial_features_6d, image_features = self.load_features()

        adj_gene = self.compute_gene_adjacency(gene_features)
        adj_spot = self.compute_spatial_adjacency(spatial_features_6d)
        adj_image = self.compute_image_adjacency(image_features)

        A1, A2, A3 = self.normalize_adjacency(adj_gene, adj_spot, adj_image)
        self.save_adjacency_matrices(A1, A2, A3)

        return A1, A2, A3

