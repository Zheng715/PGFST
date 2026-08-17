
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


from cnn_image_extractor import extract_cnn_image_features


class GeneExpressionPreprocessor:
    def __init__(self, config):
        self.config = config

    def load_expression_data(self):
        adata = sc.read_10x_h5(self.config.EXPRESSION_FILE)
        return adata

    def preprocess_expression_data(self, adata, spatial_coords_2d):
        df = pd.DataFrame(adata.X.T.toarray(),
                          index=adata.var_names,
                          columns=adata.obs_names)

        expression_ratio = (df > 0).mean(axis=1)
        df_filtered = df[expression_ratio >= self.config.MIN_EXPRESSION_RATIO]

        gene_variance = df_filtered.var(axis=1)
        top_genes = gene_variance.sort_values(ascending=False).head(self.config.TOP_VARIABLE_GENES).index
        df_top = df_filtered.loc[top_genes]

        df_filled = df_top.fillna(0)
        df_normalized = df_filled.div(df_filled.sum(axis=0), axis=1) * 1e4

        gene_data_augmented = self._spatial_augmentation(
            df_normalized.values, spatial_coords_2d
        )
        df_augmented = pd.DataFrame(
            gene_data_augmented,
            index=df_normalized.index,
            columns=df_normalized.columns
        )

        pca_result = self._perform_pca(df_augmented, self.config.PCA_COMPONENTS_GENE)
        return df_augmented, pca_result

    def _spatial_augmentation(self, data, spatial_coords):
        knn = NearestNeighbors(n_neighbors=self.config.NEIGHBOR_K + 1, metric='euclidean')
        knn.fit(spatial_coords)
        neigh_indices = knn.kneighbors(spatial_coords, return_distance=False)
        neigh_indices = neigh_indices[:, 1:]
        adjacent_avg = np.mean(np.take(data, neigh_indices, axis=1), axis=2)
        return data + self.config.ADJ_WT * adjacent_avg

    def _perform_pca(self, data, n_components):
        data_transposed = data.T
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data_transposed)
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(standardized_data)
        return pca_result


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.gene_preprocessor = GeneExpressionPreprocessor(config)
        self.processed_data = {}

    def preprocess_all_data(self):
        self._preprocess_gene_expression()
        self._save_results()
        return self.processed_data

    def _preprocess_gene_expression(self):
        adata = self.gene_preprocessor.load_expression_data()
        spatial_df, spatial_coords_2d, image_coords = self._load_spatial_data()

        gene_expression_df, gene_pca = self.gene_preprocessor.preprocess_expression_data(
            adata, spatial_coords_2d
        )

        spatial_coords_64d = self._spatial_encoding_64d(spatial_coords_2d)

        # 新增保存原始二维坐标
        self.processed_data['raw_spatial_2d'] = spatial_coords_2d
        self.processed_data['gene_expression'] = gene_expression_df
        self.processed_data['gene_pca'] = gene_pca
        self.processed_data['spatial_coords_64d'] = spatial_coords_64d
        self.processed_data['image_coords'] = image_coords
        self.processed_data['spatial_df'] = spatial_df

    def _load_spatial_data(self):
        spatial_df = pd.read_csv(self.config.METADATA_FILE, sep='\t')
        spatial_coords_2d = spatial_df[['row', 'col']].values

        if 'imagerow' in spatial_df.columns and 'imagecol' in spatial_df.columns:
            image_coords = spatial_df[['imagerow', 'imagecol']].values
        else:
            image_coords = spatial_df[['imagenow', 'imagecol']].values if 'imagenow' in spatial_df.columns else None

        if image_coords is None:
            raise ValueError("无法找到图像坐标列")

        spot_id_columns = ['barcode', 'id', 'spot_id', 'Barcode']
        spot_id_col = None
        for col in spot_id_columns:
            if col in spatial_df.columns:
                spot_id_col = col
                break
        if spot_id_col:
            spatial_df['spot_id'] = spatial_df[spot_id_col].values
        else:
            spatial_df['spot_id'] = spatial_df.index.astype(str)

        return spatial_df, spatial_coords_2d, image_coords

    def _spatial_encoding_64d(self, spatial_coords_2d):
        coords = np.asarray(spatial_coords_2d, dtype=np.float32)
        x_row = coords[:, 0:1]
        x_col = coords[:, 1:2]

        max_freq = np.pi * (2 ** 15)
        freqs = np.exp(np.linspace(0, np.log(max_freq), 31))

        row_features = [x_row]
        for freq in freqs:
            row_features.append(np.sin(x_row * freq))
            row_features.append(np.cos(x_row * freq))
        row_all = np.concatenate(row_features, axis=1)

        col_features = [x_col]
        for freq in freqs:
            col_features.append(np.sin(x_col * freq))
            col_features.append(np.cos(x_col * freq))
        col_all = np.concatenate(col_features, axis=1)

        row_freq_only = row_all[:, 1:]
        row_variances = np.var(row_freq_only, axis=0)
        row_top_idx = np.argsort(row_variances)[::-1][:31] + 1

        col_freq_only = col_all[:, 1:]
        col_variances = np.var(col_freq_only, axis=0)
        col_top_idx = np.argsort(col_variances)[::-1][:31] + 1

        row_selected = np.concatenate([x_row, row_all[:, row_top_idx]], axis=1)
        col_selected = np.concatenate([x_col, col_all[:, col_top_idx]], axis=1)

        return np.concatenate([row_selected, col_selected], axis=1)

    def _save_results(self):

        pd.DataFrame(self.processed_data['gene_pca']).to_csv(
            f"{self.config.OUTPUT_DIR}/gene_features.csv",
            index=False, header=False
        )


        pd.DataFrame(self.processed_data['spatial_coords_64d']).to_csv(
            f"{self.config.OUTPUT_DIR}/spatial_features.csv",
            index=False, header=False
        )

        pd.DataFrame(
            self.processed_data['raw_spatial_2d']
        ).to_csv(
            f"{self.config.OUTPUT_DIR}/raw_spatial_2d.csv",
            index=False, header=False
        )


        image_features = extract_cnn_image_features(self.config)
        self.processed_data['image_features'] = image_features
        pd.DataFrame(image_features).to_csv(
            f"{self.config.OUTPUT_DIR}/image_features.csv",
            index=False, header=False
        )