from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import sparse_encode

from defects_detector.core.base import BaseFeatureExtractor


def safe_norm(x, epsilon=1e-12, axis=None):
    """Calculate norm with numerical stability"""
    return torch.sqrt(torch.sum(x ** 2, dim=axis) + epsilon)


class PointEncoder(nn.Module):
    """Point cloud encoder with batch normalization"""

    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, feature_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        return x


class NeuralImplicitField(nn.Module):
    """Neural implicit field decoder for SDF prediction"""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(3, 512)
        self.fc3 = nn.Linear(512*2, 512)

        # Initialize fc3 with specific weights
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.normal_(self.fc3.weight, 0.0, np.sqrt(2) / np.sqrt(512))

        # Create 7 hidden layers with weight normalization
        for i in range(7):
            fc4 = nn.Linear(512, 512)
            torch.nn.init.constant_(fc4.bias, 0.0)
            torch.nn.init.normal_(fc4.weight, 0.0, np.sqrt(2) / np.sqrt(512))
            fc4 = nn.utils.weight_norm(fc4)
            setattr(self, "fc4" + str(i), fc4)

        # Output layer with special initialization for SDF
        self.fc5 = nn.Linear(512, 1)
        torch.nn.init.constant_(self.fc5.bias, -0.5)
        torch.nn.init.normal_(self.fc5.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(512), std=0.000001)

    def forward(self, features, points):
        """
        Args:
            features: Shape features [B, N, feature_dim]
            points: Query points [B, N, 3]
        Returns:
            SDF values [B, N, 1]
        """
        # Process shape features
        feature_encoding = F.relu(self.fc1(features))

        # Process point coordinates
        point_encoding = F.relu(self.fc2(points))

        # Concatenate and process through MLP
        net = torch.cat([point_encoding, feature_encoding], dim=2)
        net = F.relu(self.fc3(net))

        # Process through hidden layers
        for i in range(7):
            fc4 = getattr(self, "fc4" + str(i))
            net = F.relu(fc4(net))

        # Output SDF values
        sdf = self.fc5(net)
        return sdf

    def get_gradient(self, features, points):
        """Calculate gradients of SDF with respect to points"""
        points.requires_grad_(True)
        sdf = self.forward(features, points)

        # Calculate gradients
        gradient = torch.autograd.grad(
            sdf,
            points,
            torch.ones_like(sdf, requires_grad=False, device=sdf.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Normalize gradient
        normal_length = torch.unsqueeze(safe_norm(gradient, axis=-1), -1)
        grad_norm = gradient / normal_length

        # Get surface points by moving along the gradient
        surface_points = points - sdf * grad_norm
        return surface_points


class SDFModel(nn.Module):
    """Complete model combining encoder and neural implicit field decoder"""

    def __init__(self, point_num=1024, feature_dim=128):
        super().__init__()
        self.encoder = PointEncoder(in_channels=3, feature_dim=feature_dim)
        self.NIF = NeuralImplicitField(feature_dim=feature_dim)
        self.point_num = point_num

    def forward(self, points, query_points):
        """
        Forward pass to get surface points

        Args:
            points: Input point cloud [B, N, 3]
            query_points: Query points [B, M, 3]

        Returns:
            Surface points [B, M, 3]
        """
        # Transpose for 1D convolution [B, 3, N]
        points_t = torch.permute(points, (0, 2, 1))

        # Extract features
        features = self.encoder(points_t)

        # Replicate features for each query point
        point_features = torch.tile(
            torch.unsqueeze(features, 1),
            [1, self.point_num, 1]
        )

        # Get surface points
        surface_points = self.NIF.get_gradient(point_features, query_points)
        return surface_points

    def get_feature(self, points):
        """Extract features from point cloud"""
        points_t = torch.permute(points, (0, 2, 1))
        return self.encoder(points_t)

    def get_sdf(self, features, query_points):
        """Get SDF values for query points given features"""
        return self.NIF(features, query_points)

    def freeze_model(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad_(False)


class SDFFeatureExtractor(BaseFeatureExtractor):
    """SDF feature extractor for point clouds"""

    def __init__(self, model: SDFModel, image_size, feature_size, batch_size=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sdf_model = model.to(self.device).eval()

        self.image_size = image_size
        self.feature_size = feature_size
        self.batch_size = batch_size

        self.resize = torch.nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.average = torch.nn.AvgPool2d(3, stride=1)

    def extract_features(self, points_all, points_idx) -> torch.Tensor:
        """Извлекает признаки из входных данных"""

        def process_patch(patch):
            points = points_all[patch].reshape(-1, self.sdf_model.point_num, 3)
            return self.sdf_model.get_feature(points.to(self.device))

        return torch.cat([process_patch(patch) for patch in range(len(points_all))], dim=0)


    def get_score_map(self, feature: torch.Tensor, points_all, points_idx):
        """
        Создает карту оценок аномалий на основе SDF-признаков

        Args:
            feature: Извлеченные признаки формы [N, feature_dim]
            points_all: Список облаков точек для оценки
            points_idx: Список индексов точек, соответствующих пикселям изображения

        Returns:
            Карта оценок размером [image_size, image_size]
        """
        s_map = np.full((self.image_size * self.image_size), 0, dtype=float)
        for patch in range(len(points_all)):

            if points_all[patch].shape[1] != self.sdf_model.point_num:
                print('Error!', points_all[patch].shape)
                continue

            points = points_all[patch].reshape(-1, self.sdf_model.point_num, 3)
            indices = points_idx[patch].reshape(-1, self.sdf_model.point_num)

            point_feature = torch.tile(torch.unsqueeze(feature[patch], 0), [1, self.sdf_model.point_num, 1])
            index = indices[0].reshape(self.sdf_model.point_num)
            point_target = points[0, :].reshape(self.batch_size, self.sdf_model.point_num, 3)
            point_target = point_target.to(self.device)
            point_feature = point_feature.to(self.device)
            sdf_c = self.sdf_model.get_sdf(point_feature, point_target)

            sdf_c = np.abs(sdf_c.detach().cpu().numpy().reshape(-1))
            index = index.cpu().numpy()

            tmp_map = np.full((self.image_size * self.image_size), 0, dtype=float)
            for L in range(sdf_c.shape[0]):
                tmp_map[index[L]] = sdf_c[L]
                if (s_map[index[L]] == 0) or (s_map[index[L]] > sdf_c[L]):
                    s_map[index[L]] = sdf_c[L]

        s_map = s_map.reshape(1, 1, self.image_size, self.image_size)
        return torch.tensor(s_map)

    def compute_anomaly_map(self, *args: Any):
        ...