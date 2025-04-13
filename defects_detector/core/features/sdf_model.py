import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
            layer = nn.Linear(512, 512)
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(512))
            layer = nn.utils.weight_norm(layer)
            setattr(self, f"fc{i+4}", layer)

        # Output layer with special initialization for SDF
        self.fc_out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.fc_out.bias, -0.5)
        torch.nn.init.normal_(
            self.fc_out.weight,
            mean=2*np.sqrt(np.pi) / np.sqrt(512),
            std=0.000001
        )

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
            layer = getattr(self, f"fc{i+4}")
            net = F.relu(layer(net))

        # Output SDF values
        sdf = self.fc_out(net)
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
        self.decoder = NeuralImplicitField(feature_dim=feature_dim)
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
        surface_points = self.decoder.get_gradient(point_features, query_points)
        return surface_points

    def get_feature(self, points):
        """Extract features from point cloud"""
        points_t = torch.permute(points, (0, 2, 1))
        return self.encoder(points_t)

    def get_sdf(self, features, query_points):
        """Get SDF values for query points given features"""
        return self.decoder(features, query_points)

    def freeze_model(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad_(False)