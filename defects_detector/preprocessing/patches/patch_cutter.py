import numpy as np
import os
from typing import Dict, List, Any
from defects_detector.preprocessing.patches.base import BaseDataLoader, BasePatchCutter

class PatchCutterService:
    """Main service for cutting patches from 3D data"""

    def __init__(
            self,
            data_loader: BaseDataLoader,
            patch_cutter: BasePatchCutter,
            save_path: str = "./patches"
    ):
        self.data_loader = data_loader
        self.patch_cutter = patch_cutter
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def process_file(self, input_file: str, save_path: str, split: str) -> str:
        """Process a single file and save patches"""
        # Load and preprocess data
        data = self.data_loader.load_data(input_file)
        point_cloud_data = self.data_loader.get_point_cloud(data)
        points = point_cloud_data["point_cloud"]
        indices = point_cloud_data["indices"]

        # Normalize points
        points = self.data_loader.normalize(points)

        # Cut patches
        patches = self.patch_cutter.cut_patches(points, indices)

        # Save patches
        self.save_patches(patches, save_path, split)

        return save_path

    def process_directory(self, input_dir: str, split: str) -> List[str]:
        """Process all files in directory by class"""
        processed_files = []

        # Get dataset files using the data loader's method
        file_info_list = self.data_loader.get_dataset_files(input_dir, split)

        for file_info in file_info_list:
            # Get save path using the data loader's method
            save_path = self.data_loader.get_save_path(file_info, self.save_path)

            # Process file
            processed_file = self.process_file(file_info["file_path"], save_path, split)
            processed_files.append(processed_file)

        return processed_files

    def save_patches(self, patches: Dict[str, np.ndarray], save_path: str, split: str) -> None:
        """Save patches to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if split == 'pretrain':
            # For pretraining, we need to process patches for SDF model
            grouped_xyz = patches["patches"]

            origin_all = []
            sample_all = []
            sample_near_all = []

            for patch in range(grouped_xyz.shape[0]):
                samples, sample_near = self.patch_cutter.sample_query_points(
                    grouped_xyz[patch],
                    query_num=500  # Default value, could be made configurable
                )

                origin_all.append(grouped_xyz[patch])
                sample_all.append(samples)
                sample_near_all.append(sample_near)

            np.savez(save_path, origins_all=origin_all, samples_all=sample_all, points_all=sample_near_all)
        else:
            np.savez(save_path, points_gt=patches["patches"], points_idx=patches["indices"])