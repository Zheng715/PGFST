
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder




class ClusteringVisualization:
    def __init__(self, config):
        self.config = config
        self.fixed_colors_7 = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF8C00', '#20B2AA']
        self.fixed_colors_5 = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']

    def plot_spatial_scatter(self, spatial_coords, cluster_labels,
                             title="Spatial Cluster Scatter", save_path=None):
        label_encoder = LabelEncoder()
        cluster_labels_numeric = label_encoder.fit_transform(cluster_labels)

        plt.figure(figsize=(6, 7))

        n_clusters = len(np.unique(cluster_labels_numeric))
        if n_clusters <= 5:
            custom_cmap = plt.cm.colors.ListedColormap(self.fixed_colors_5[:n_clusters])
        else:
            custom_cmap = plt.cm.colors.ListedColormap(self.fixed_colors_7[:n_clusters])

        scatter = plt.scatter(spatial_coords[:, 1], spatial_coords[:, 0],
                              c=cluster_labels_numeric, cmap=custom_cmap, s=12, alpha=1.0)
        plt.gca().invert_yaxis()
        handles, labels = scatter.legend_elements(prop='colors', num=len(np.unique(cluster_labels_numeric)))
        plt.legend(handles, [f'{int(l)}' for l in np.unique(cluster_labels_numeric)],
                   title='Cluster', loc='upper left',
                   bbox_to_anchor=(1.02, 1), frameon=True, handletextpad=0.2, labelspacing=0.2)
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.tight_layout()
        plt.show()

    def load_data(self):

        try:
            cluster_results = pd.read_csv(f"{self.config.OUTPUT_DIR}/deterministic_clustering_results.csv")
            cluster_labels = cluster_results['cluster_label'].values
        except Exception as e:
            print(f"无法加载确定性聚类结果: {e}")
            return None, None

        try:
            metadata = pd.read_csv(f"{self.config.DATA_ROOT}/metadata.tsv", sep='\t')
            valid_mask = ~((metadata['layer_guess_reordered'] == 'NA') | metadata['layer_guess_reordered'].isna())
            filtered_metadata = metadata[valid_mask]
            spatial_coords = filtered_metadata[['row', 'col']].values

            if len(cluster_labels) == len(metadata):
                valid_indices = np.arange(len(metadata))[valid_mask]
                valid_cluster_labels = cluster_labels[valid_indices]
            else:
                valid_cluster_labels = cluster_labels

            return valid_cluster_labels, spatial_coords

        except Exception as e:
            print(f"加载数据失败: {e}")
            return None, None

    def run_complete_analysis(self, features=None, n_clusters=5):
        cluster_labels, spatial_coords = self.load_data()
        if cluster_labels is None:
            return

        save_path = f"{self.config.OUTPUT_DIR}/spatial_cluster_scatter.png"
        self.plot_spatial_scatter(spatial_coords, cluster_labels,
                                  title="Optimal Spatial Clusters Scatter",
                                  save_path=save_path)