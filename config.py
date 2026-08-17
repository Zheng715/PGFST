import os


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, "Dataset")

    EXPRESSION_FILE = os.path.join(DATA_ROOT, "filtered_feature_bc_matrix.h5")
    METADATA_FILE = os.path.join(DATA_ROOT, "metadata.tsv")
    IMAGE_FILE = os.path.join(DATA_ROOT, "spatial/tissue_hires_image.png")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "output")

    MIN_EXPRESSION_RATIO = 0.01
    TOP_VARIABLE_GENES = 1000
    PCA_COMPONENTS_GENE = 500
    ADJ_WT = 0.3
    NEIGHBOR_K = 15

    SPATIAL_ENCODING_DIM = 128
    CROP_SIZE = (100, 100)
    PCA_COMPONENTS_IMAGE = 200
    IMAGE_FEATURE_DIM = 200
    CNN_MODEL_NAME = 'resnet18'


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.DATA_ROOT, exist_ok=True)