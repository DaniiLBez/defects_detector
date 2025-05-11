import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
from typing import Union, Tuple, Dict, Any
import tqdm

from defects_detector.preprocessing.plane_remover.base import PreprocessingServiceBase

class MVTec3DPreprocessingService(PreprocessingServiceBase):
    """MVTec 3D-AD specific implementation of preprocessing service"""

    def preprocess_all(self, dataset_path: Union[str, Path], verbose: bool = True) -> int:
        """Process all tiff files in the dataset recursively"""
        paths = list(Path(dataset_path).rglob('*.tiff'))
        if verbose:
            print(f"Found {len(paths)} tiff files in {dataset_path}")

        processed_files = 0
        for path in tqdm.tqdm(paths, disable=not verbose):
            self.preprocess_file(path)
            processed_files += 1
            if verbose and processed_files % 50 == 0:
                print(f"Processed {processed_files} tiff files...")

        return processed_files

    def preprocess_file(self, tiff_path: Union[str, Path]) -> None:
        """Preprocess a single MVTec 3D-AD point cloud file"""
        # Load data
        data = self._load_data(tiff_path)

        # Process data
        processed_data = self._process_data(data, tiff_path)

        # Save processed data
        self._save_data(processed_data, tiff_path)

    def _load_data(self, tiff_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load MVTec 3D-AD data files"""
        organized_pc = self._read_tiff_organized_pc(tiff_path)
        rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
        gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
        organized_rgb = np.array(Image.open(rgb_path))

        data = {
            'point_cloud': organized_pc,
            'rgb': organized_rgb,
            'gt_exists': os.path.isfile(gt_path)
        }

        if data['gt_exists']:
            data['gt'] = np.array(Image.open(gt_path))

        return data

    def _process_data(self, data: Dict[str, Any], file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process MVTec 3D-AD data"""
        # Remove plane
        planeless_pc, planeless_rgb = self._remove_plane(data['point_cloud'], data['rgb'])

        # Pad with zeros
        padded_pc = self._pad_cropped_pc(planeless_pc, single_channel=False)
        padded_rgb = self._pad_cropped_pc(planeless_rgb, single_channel=False)

        # Connected component cleaning
        cleaned_pc, cleaned_rgb = self._connected_components_cleaning(padded_pc, padded_rgb, file_path)

        result = {
            'point_cloud': cleaned_pc,
            'rgb': cleaned_rgb
        }

        if data['gt_exists']:
            result['gt'] = self._pad_cropped_pc(data['gt'], single_channel=True)

        return result

    def _save_data(self, data: Dict[str, Any], tiff_path: Union[str, Path]) -> None:
        """Save processed MVTec 3D-AD data"""
        rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
        gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")

        tiff.imwrite(tiff_path, data['point_cloud'])
        Image.fromarray(data['rgb']).save(rgb_path)

        if 'gt' in data:
            Image.fromarray(data['gt']).save(gt_path)

    @staticmethod
    def _read_tiff_organized_pc(tiff_path: Union[str, Path]) -> np.ndarray:
        """Read organized point cloud from tiff file"""
        return tiff.imread(tiff_path)

    @staticmethod
    def _organized_pc_to_unorganized_pc(organized_pc: np.ndarray) -> np.ndarray:
        """Convert organized PC to unorganized PC"""
        return organized_pc.reshape(-1, organized_pc.shape[2])

    @staticmethod
    def _get_edges_of_pc(organized_pc: np.ndarray) -> np.ndarray:
        """Get edge points of the organized point cloud"""
        unorganized_edges_pc = organized_pc[0:10, :, :].reshape(
            organized_pc[0:10, :, :].shape[0] * organized_pc[0:10, :, :].shape[1], organized_pc[0:10, :, :].shape[2])
        unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[-10:, :, :].reshape(
            organized_pc[-10:, :, :].shape[0] * organized_pc[-10:, :, :].shape[1], organized_pc[-10:, :, :].shape[2])],
                                              axis=0)
        unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, 0:10, :].reshape(
            organized_pc[:, 0:10, :].shape[0] * organized_pc[:, 0:10, :].shape[1], organized_pc[:, 0:10, :].shape[2])],
                                              axis=0)
        unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, -10:, :].reshape(
            organized_pc[:, -10:, :].shape[0] * organized_pc[:, -10:, :].shape[1], organized_pc[:, -10:, :].shape[2])],
                                              axis=0)
        unorganized_edges_pc = unorganized_edges_pc[np.nonzero(np.all(unorganized_edges_pc != 0, axis=1))[0], :]
        return unorganized_edges_pc

    @staticmethod
    def _get_plane_eq(unorganized_pc: np.ndarray, ransac_n_pts: int = 50) -> np.ndarray:
        """Get plane equation from point cloud"""
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
        plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts,
                                                    num_iterations=1000)
        return plane_model

    def _remove_plane(self, organized_pc_clean: np.ndarray, organized_rgb: np.ndarray,
                      distance_threshold: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """Remove plane from point cloud"""
        unorganized_pc = self._organized_pc_to_unorganized_pc(organized_pc_clean)
        unorganized_rgb = self._organized_pc_to_unorganized_pc(organized_rgb)
        clean_planeless_unorganized_pc = unorganized_pc.copy()
        planeless_unorganized_rgb = unorganized_rgb.copy()

        plane_model = self._get_plane_eq(self._get_edges_of_pc(organized_pc_clean))
        distances = np.abs(np.dot(np.array(plane_model), np.hstack(
            (clean_planeless_unorganized_pc, np.ones((clean_planeless_unorganized_pc.shape[0], 1)))).T))
        plane_indices = np.argwhere(distances < distance_threshold)

        planeless_unorganized_rgb[plane_indices] = 0
        clean_planeless_unorganized_pc[plane_indices] = 0
        clean_planeless_organized_pc = clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape)
        planeless_organized_rgb = planeless_unorganized_rgb.reshape(organized_rgb.shape)

        return clean_planeless_organized_pc, planeless_organized_rgb

    @staticmethod
    def _pad_cropped_pc(cropped_pc: np.ndarray, single_channel: bool = False) -> np.ndarray:
        """Pad point cloud to make it square"""
        orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
        round_orig_h = int(math.ceil(orig_h / 100.0)) * 100
        round_orig_w = int(math.ceil(orig_w / 100.0)) * 100
        large_side = max(round_orig_h, round_orig_w)

        a = (large_side - orig_h) // 2
        aa = large_side - a - orig_h

        b = (large_side - orig_w) // 2
        bb = large_side - b - orig_w

        if single_channel:
            return np.pad(cropped_pc, pad_width=((a, aa), (b, bb)), mode='constant')
        else:
            return np.pad(cropped_pc, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

    def _connected_components_cleaning(self, organized_pc: np.ndarray, organized_rgb: np.ndarray,
                                       image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Clean point cloud using connected components"""
        unorganized_pc = self._organized_pc_to_unorganized_pc(organized_pc)
        unorganized_rgb = self._organized_pc_to_unorganized_pc(organized_rgb)

        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))

        unique_cluster_ids, cluster_size = np.unique(labels, return_counts=True)
        max_label = labels.max()
        if max_label > 0:
            print(f"Point cloud file {image_path} has {max_label + 1} clusters")
            print(f"Cluster ids: {unique_cluster_ids}. Cluster size {cluster_size}")

        largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]
        outlier_indices_nonzero_array = np.argwhere(labels != largest_cluster_id)
        outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
        unorganized_pc[outlier_indices_original_pc_array] = 0
        unorganized_rgb[outlier_indices_original_pc_array] = 0

        organized_clustered_pc = unorganized_pc.reshape(organized_pc.shape)
        organized_clustered_rgb = unorganized_rgb.reshape(organized_rgb.shape)

        return organized_clustered_pc, organized_clustered_rgb
