
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config


class ImprovedFeatureFusion:
    def __init__(self, config):
        self.config = config
        self.gene_weight = 0.50
        self.spatial_weight = 0.40
        self.image_weight = 0.10

    def load_features(self):
        gene_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/gene_features.csv", header=None).values
        spatial_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/spatial_features.csv", header=None).values
        image_features = pd.read_csv(f"{self.config.OUTPUT_DIR}/image_features.csv", header=None).values
        return gene_features, spatial_features, image_features

    def balanced_feature_fusion(self):
        gene_features, spatial_features, image_features = self.load_features()

        scaler_gene = StandardScaler()
        gene_norm = scaler_gene.fit_transform(gene_features) * self.gene_weight

        scaler_spatial = StandardScaler()
        spatial_norm = scaler_spatial.fit_transform(spatial_features) * self.spatial_weight

        scaler_image = StandardScaler()
        image_norm = scaler_image.fit_transform(image_features) * self.image_weight

        return np.hstack((gene_norm, spatial_norm, image_norm))

    def save_fused_features(self, fused_features):
        output_file = f"{self.config.OUTPUT_DIR}/fused_features.csv"
        pd.DataFrame(fused_features).to_csv(output_file, index=False, header=False)
        return output_file