# cnn_image_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

from config import Config


class CNNImageFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 512

    def load_cnn_model(self):
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model.to(self.device)

    def image_crop(self, image, coordinates, crop_size=(50, 50)):
        x, y = coordinates
        left = max(int(x - crop_size[0] // 2), 0)
        upper = max(int(y - crop_size[1] // 2), 0)
        right = min(int(x + crop_size[0] // 2), image.width)
        lower = min(int(y + crop_size[1] // 2), image.height)
        if right <= left or lower <= upper:
            return None
        return image.crop((left, upper, right, lower))

    def extract_features_from_image_patch(self, image_patch, model, transform):
        if image_patch is None or image_patch.size[0] == 0 or image_patch.size[1] == 0:
            return np.zeros(self.feature_dim)
        try:
            image_tensor = transform(image_patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = model(image_tensor)
            features = features.squeeze().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
            return features
        except Exception:
            return np.zeros(self.feature_dim)

    def extract_cnn_features(self, image_path, image_coords):
        tissue_image = Image.open(image_path).convert('RGB')
        cnn_model = self.load_cnn_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        features = []
        for coord in image_coords:
            image_patch = self.image_crop(tissue_image, coord, self.config.CROP_SIZE)
            if image_patch is not None:
                feature = self.extract_features_from_image_patch(image_patch, cnn_model, transform)
                features.append(feature)
            else:
                features.append(np.zeros(self.feature_dim))

        features = np.array(features)
        return self._reduce_dimension(features)

    def _reduce_dimension(self, features):
        if np.all(features == 0) or np.sum(~np.all(features == 0, axis=1)) == 0:
            return np.zeros((features.shape[0], self.config.PCA_COMPONENTS_IMAGE))

        non_zero_mask = ~np.all(features == 0, axis=1)
        non_zero_features = features[non_zero_mask]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(non_zero_features)

        n_components = min(self.config.PCA_COMPONENTS_IMAGE, features_scaled.shape[1], features_scaled.shape[0])
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features_scaled)

        final_features = np.zeros((features.shape[0], n_components))
        final_features[non_zero_mask] = features_reduced
        return final_features


def extract_cnn_image_features(config):
    spatial_df = pd.read_csv(config.METADATA_FILE, sep='\t')
    image_coords = spatial_df[['imagerow', 'imagecol']].values

    if not os.path.exists(config.IMAGE_FILE):
        possible_paths = [
            config.IMAGE_FILE,
            os.path.join(config.DATA_ROOT, "tissue_hires_image.png"),
            os.path.join(config.DATA_ROOT, "spatial", "tissue_hires_image.png"),
            os.path.join(config.DATA_ROOT, "tissue_lowres_image.png"),
            os.path.join(config.DATA_ROOT, "spatial", "tissue_lowres_image.png"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config.IMAGE_FILE = path
                break

    extractor = CNNImageFeatureExtractor(config)
    image_features = extractor.extract_cnn_features(config.IMAGE_FILE, image_coords)

    pd.DataFrame(image_features).to_csv(
        f"{config.OUTPUT_DIR}/image_features.csv",
        index=False, header=False
    )
    return image_features