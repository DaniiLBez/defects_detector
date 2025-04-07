import glob
import tifffile as tiff
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from defects_detector.preprocessing.patches.base import BaseDataLoader, BasePatchCutter
from defects_detector.utils.pointnet_util import project_to_org, farthest_point_sample, index_points, square_distance, query_ball_point


class MVTec3DLoader(BaseDataLoader):
    """Data loader for MVTec3D-AD dataset"""

    def load_data(self, path: str) -> Dict[str, Any]:
        """Load MVTec3D-AD data"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        return {"points": tiff.imread(path)}


    def get_point_cloud(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract point cloud from MVTec3D-AD data"""
        points = input["points"]
        target_height = kwargs.get('target_height', 224)
        target_width = kwargs.get('target_width', 224)
        torch_organized_pc = torch.tensor(points).permute(2, 0, 1).unsqueeze(dim=0)
        resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc,
                                                                     size=(target_height, target_width),
                                                                     mode='nearest')
        organized_pc_np = resized_organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_np.reshape(organized_pc_np.shape[0] * organized_pc_np.shape[1], organized_pc_np.shape[2])
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        return {
            "point_cloud": unorganized_pc[nonzero_indices, :],
            "indices": nonzero_indices,
        }

    def normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize MVTec3D-AD point cloud"""
        pointcloud_s = points.astype(np.float32)
        # print('### pointcloud sparse:', pointcloud_s.shape[0])

        pointcloud_s_t = pointcloud_s - np.array(
            [np.min(pointcloud_s[:, 0]), np.min(pointcloud_s[:, 1]), np.min(pointcloud_s[:, 2])])
        pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:, 0]) - np.min(pointcloud_s[:, 0]),
                                                     np.max(pointcloud_s[:, 0]) - np.min(pointcloud_s[:, 0]),
                                                     np.max(pointcloud_s[:, 0]) - np.min(pointcloud_s[:, 0])]))
        return pointcloud_s_t

    def get_dataset_files(self, input_dir: str, split: str) -> List[Dict[str, Any]]:
        """
        Get dataset files based on directory structure and split

        Args:
            input_dir: Root directory of dataset
            split: Dataset split ('train', 'test', 'validation', 'pretrain')

        Returns:
            List of dictionaries containing file information
        """
        file_info_list = []

        if split not in ['test', 'train', 'validation', 'pretrain']:
            return [
                {
                    "file_path": file_path,
                    "class_name": os.path.basename(input_dir),
                    "object_id": os.path.splitext(os.path.basename(file_path))[0],
                    "split": split
                }
                for file_path in glob.glob(os.path.join(input_dir, "xyz", "*.tiff"))
            ]

        # For MVTec3D-AD dataset structure
        class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        for class_name in class_dirs:
            if split == 'train':
                # Only good samples in train
                dir_path = glob.glob(os.path.join(input_dir, class_name, "train", "good", "xyz", "*.tiff"))
                defect_type = "good"
            elif split == 'pretrain':
                # Random subset for pretraining
                import random
                dir_path = glob.glob(os.path.join(input_dir, class_name, "train", "good", "xyz", "*.tiff"))
                dir_path = random.sample(dir_path, min(20, len(dir_path)))  # Take up to 20 samples
                defect_type = "good"
            else:
                # All types in test/validation
                defect_types_path = os.path.join(input_dir, class_name, split)
                if os.path.exists(defect_types_path):
                    for defect_type in os.listdir(defect_types_path):
                        tiff_path = glob.glob(os.path.join(input_dir, class_name, split, defect_type, "xyz", "*.tiff"))
                        for path in tiff_path:
                            file_info = {
                                "file_path": path,
                                "class_name": class_name,
                                "defect_type": defect_type,
                                "object_id": os.path.splitext(os.path.basename(path))[0],
                                "split": split
                            }
                            file_info_list.append(file_info)
                continue  # Skip the loop below since we already added file_info entries

            print(f"Found {len(dir_path)} files for {class_name} ({split})")

            for path in dir_path:
                file_info = {
                    "file_path": path,
                    "class_name": class_name,
                    "defect_type": defect_type,
                    "object_id": os.path.splitext(os.path.basename(path))[0],
                    "split": split
                }
                file_info_list.append(file_info)

        return file_info_list

    def get_save_path(self, file_info: Dict[str, Any], base_save_path: str) -> str:
        """
        Generate appropriate save path for processed files

        Args:
            file_info: Dictionary with file information
            base_save_path: Base directory for saving files

        Returns:
            Full save path for the file
        """
        if file_info["split"] == 'pretrain':
            save_dir = os.path.join(base_save_path, 'PRETRAIN_DATA', file_info["class_name"])
        elif file_info["split"] == 'custom' and file_info.get("defect_type") is None:
            save_dir = os.path.join(base_save_path, "npz")
        else:
            save_dir = os.path.join(base_save_path, file_info["class_name"],
                                    file_info["split"], file_info["defect_type"], "npz")

        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{file_info['object_id']}.npz")

        return save_file


class MVTec3DPatchCutter(BasePatchCutter):
    """Patch cutter for MVTec3D-AD point clouds"""

    def __init__(self, group_size: int = 500, group_mul: float = 10.0):
        """
        Initialize MVTec3D Patch Cutter

        Args:
            group_size: Number of points in each patch
            group_mul: Multiplier for calculating number of groups
        """
        self.group_size = group_size
        self.group_mul = group_mul

    def sample_and_group(self, points: Tensor, npoint: int, nsample: int,
                         indices: np.ndarray, radius=0.5, knn=True) -> Tuple[Tensor, np.ndarray]:
        """Sample points and group into patches for MVTec3D-AD"""
        """
            Input:
                xyz: input points position data, [B, N, 3]
                points: input points data, [B, N, D]
                npoint: number of knn groups.
                nsample: number of sample points in the knn group.
                radius:
            Return:
                grouped_xyz: sampled points position data, [B, npoint, nsample, 3]
                fps_idx: the index of center points 
                org_idx_all: sampled pc corresponding to the original pc index
            """
        fps_idx = farthest_point_sample(points, npoint)  # [B, npoint]
        torch.cuda.empty_cache()
        new_xyz = index_points(points, fps_idx)
        torch.cuda.empty_cache()
        if knn:
            dists = square_distance(new_xyz, points)  # B x npoint x N
            idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
        else:
            idx = query_ball_point(radius, nsample, points, new_xyz)
        torch.cuda.empty_cache()
        org_idx_all = project_to_org(idx, indices)
        torch.cuda.empty_cache()
        grouped_xyz = index_points(points, idx)  # [B, npoint, nsample, C]
        torch.cuda.empty_cache()
        return grouped_xyz, org_idx_all

    def cut_patches(self, points: np.ndarray, indices: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Cut patches from MVTec3D-AD point cloud"""
        # Calculate number of groups based on point count and group size
        num_group = round(points.shape[0] * self.group_mul / self.group_size)

        # Prepare points for processing
        points_tensor = np.expand_dims(points, axis=0)

        # FPS + KNN Patch Cutting
        grouped_xyz, org_idx_all = self.sample_and_group(
            points=torch.tensor(points_tensor),
            npoint=num_group,
            nsample=self.group_size,
            indices=indices
        )

        return {
            "patches": np.squeeze(grouped_xyz.cpu().data.numpy()),
            "indices": np.squeeze(org_idx_all)
        }

    def sample_query_points(self, target_points: np.ndarray, query_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample query points for training from MVTec3D-AD data"""
        sample = []
        sample_near = []
        gt_kd_tree = cKDTree(target_points)

        # Split according to the original code: 62.5% near points, 37.5% far points
        num_near = int(np.ceil(query_num * 0.625))
        num_far = query_num - num_near

        # Sample near points
        for _ in range(num_near):
            ptree = cKDTree(target_points)
            sigmas = []
            for p in np.array_split(target_points, 100, axis=0):
                d = ptree.query(p, 51)
                sigmas.append(d[0][:, -1])

            sigmas = np.concatenate(sigmas)
            sampled_points = target_points + 0.5 * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0,
                                                                                                 size=target_points.shape)
            sample.append(sampled_points)

            _, vertex_ids = gt_kd_tree.query(sampled_points, p=2, k=1)
            sample_near.append(target_points[vertex_ids].reshape(-1, 3))

        # Sample far points
        for _ in range(num_far):
            ptree = cKDTree(target_points)
            sigmas = []
            for p in np.array_split(target_points, 100, axis=0):
                d = ptree.query(p, 51)
                sigmas.append(d[0][:, -1])

            sigmas = np.concatenate(sigmas)
            sampled_points = target_points + 1.0 * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0,
                                                                                                 size=target_points.shape)
            sample.append(sampled_points)

            _, vertex_ids = gt_kd_tree.query(sampled_points, p=2, k=1)
            sample_near.append(target_points[vertex_ids].reshape(-1, 3))

        return np.vstack(sample), np.vstack(sample_near)