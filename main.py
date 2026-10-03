from config import Config
from data_preprocessing import DataPreprocessor
from feature_fusion import ImprovedFeatureFusion
from graph_construction import GraphConstructor
from cascade_gcn import DeterministicTrainer
from clustering_visualization import ClusteringVisualization


def main():
    config = Config()

    print("\n[Stage 1/5] Data Preprocessing...")
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess_all_data()
    print("✓ Data preprocessing completed")

    print("\n[Stage 2/5] Feature Fusion...")
    fusion = ImprovedFeatureFusion(config)
    fused_features = fusion.balanced_feature_fusion()
    fusion.save_fused_features(fused_features)
    print("✓ Feature fusion completed")

    print("\n[Stage 3/5] Constructing Adjacency Matrices...")
    graph_constructor = GraphConstructor(config)
    A1, A2, A3 = graph_constructor.construct_all_graphs()
    print("✓ Original three modality graphs constructed")

    print("\n[Stage 4/5] Cascade GCN Training...")
    trainer = DeterministicTrainer(config)
    cluster_labels, features, metrics = trainer.train_model(n_clusters=5)
    ari_score = metrics["ARI"]
    print(f"✓ GCN training completed, ARI: {ari_score:.4f}")

    print("\n[Stage 5/5] Clustering Visualization...")
    visualizer = ClusteringVisualization(config)
    visualizer.run_complete_analysis(n_clusters=5)

    print(f"Final ARI Score: {ari_score:.4f}")


if __name__ == "__main__":
    main()