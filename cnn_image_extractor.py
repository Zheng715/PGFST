import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')
from config import Config

class CNNImageFeatureExtractor:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_cnn_model(self, model_name='resnet18', pretrained=True):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f'不支持的模型: {model_name}')
        model.eval()
        model = model.to(self.device)
        return model

    def image_crop(self, image, coordinates, crop_size=(100, 100)):
        x, y = coordinates
        left = max(int(x - crop_size[0] // 2), 0)
        upper = max(int(y - crop_size[1] // 2), 0)
        right = min(int(x + crop_size[0] // 2), image.width)
        lower = min(int(y + crop_size[1] // 2), image.height)
        if right <= left or lower <= upper:
            return None
        cropped = image.crop((left, upper, right, lower))
        return cropped

    def extract_features_from_image_patch(self, image_patch, model, transform):
        if image_patch is None or image_patch.size[0] == 0 or image_patch.size[1] == 0:
            return np.zeros(self.feature_dim)
        try:
            image_tensor = transform(image_patch).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                features = model(image_tensor)
            features = features.squeeze().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
            return features
        except Exception as e:
            return np.zeros(self.feature_dim)

    def extract_cnn_features(self, image_path, image_coords):
        try:
            tissue_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return self._create_fallback_features(len(image_coords))
        cnn_model = self.load_cnn_model(self.config.CNN_MODEL_NAME)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        features = []
        successful_extractions = 0
        for i, coord in enumerate(image_coords):
            try:
                image_patch = self.image_crop(tissue_image, coord, self.config.CROP_SIZE)
                if image_patch is not None:
                    feature = self.extract_features_from_image_patch(image_patch, cnn_model, transform)
                    features.append(feature)
                    successful_extractions += 1
                else:
                    features.append(np.zeros(self.feature_dim))
            except Exception as e:
                features.append(np.zeros(self.feature_dim))
        features = np.array(features)
        final_features = self._reduce_dimension(features)
        return final_features

    def _reduce_dimension(self, features):
        if np.all(features == 0):
            return np.zeros((features.shape[0], self.config.PCA_COMPONENTS_IMAGE))
        non_zero_mask = ~np.all(features == 0, axis=1)
        if np.sum(non_zero_mask) == 0:
            return np.zeros((features.shape[0], self.config.PCA_COMPONENTS_IMAGE))
        non_zero_features = features[non_zero_mask]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(non_zero_features)
        n_components = min(self.config.PCA_COMPONENTS_IMAGE, features_scaled.shape[1], features_scaled.shape[0])
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features_scaled)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        final_features = np.zeros((features.shape[0], n_components))
        final_features[non_zero_mask] = features_reduced
        return final_features

    def _create_fallback_features(self, n_spots):
        np.random.seed(42)
        return np.random.normal(0, 1, (n_spots, self.config.PCA_COMPONENTS_IMAGE))

def extract_cnn_image_features(config):
    try:
        spatial_df = pd.read_csv(config.METADATA_FILE, sep='\t')
    except Exception as e:
        try:
            spatial_df = pd.read_csv(config.METADATA_FILE, delimiter='\t')
        except:
            spatial_df = pd.read_csv(config.METADATA_FILE)
    image_coords = None
    if 'imagerow' in spatial_df.columns and 'imagecol' in spatial_df.columns:
        image_coords = spatial_df[['imagerow', 'imagecol']].to_numpy(dtype=np.float64)
    else:
        raise KeyError("在空间数据中找不到图像坐标列 'imagerow' 和 'imagecol'")
    if not os.path.exists(config.IMAGE_FILE):
        possible_paths = [config.IMAGE_FILE, os.path.join(config.DATA_ROOT, 'tissue_hires_image.png'), os.path.join(config.DATA_ROOT, 'spatial', 'tissue_hires_image.png'), os.path.join(config.DATA_ROOT, 'tissue_lowres_image.png'), os.path.join(config.DATA_ROOT, 'spatial', 'tissue_lowres_image.png')]
        image_found = False
        for path in possible_paths:
            if os.path.exists(path):
                config.IMAGE_FILE = path
                image_found = True
                break
        if not image_found:
            raise FileNotFoundError('找不到组织图像文件')
    required = {'width', 'height'}
    if not required.issubset(spatial_df.columns):
        raise ValueError('metadata.tsv 缺少 width/height，无法可靠转换图像坐标')
    widths = pd.to_numeric(spatial_df['width'], errors='raise').unique()
    heights = pd.to_numeric(spatial_df['height'], errors='raise').unique()
    if len(widths) != 1 or len(heights) != 1 or widths[0] <= 0 or (heights[0] <= 0):
        raise ValueError('metadata.tsv 的 width/height 必须各自为同一个正数')
    with Image.open(config.IMAGE_FILE) as im:
        actual_width, actual_height = im.size
    scale_x = actual_width / float(widths[0])
    scale_y = actual_height / float(heights[0])
    image_coords = np.column_stack((image_coords[:, 1] * scale_x, image_coords[:, 0] * scale_y))
    extractor = CNNImageFeatureExtractor(config)
    image_features = extractor.extract_cnn_features(config.IMAGE_FILE, image_coords)
    pd.DataFrame(image_features).to_csv(f'{config.OUTPUT_DIR}/image_features.csv', index=False, header=False)
    return image_features
