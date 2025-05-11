from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any

from torch import Tensor


class BaseDataLoader(ABC):
    """Abstract base class for loading 3D data from different formats"""

    @abstractmethod
    def load_data(self, path: str) -> Dict[str, Any]:
        """Load 3D data from specified path"""
        pass

    @abstractmethod
    def get_point_cloud(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract point cloud from loaded data"""
        pass

    @abstractmethod
    def normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud"""
        pass

    @abstractmethod
    def get_dataset_files(self, input_dir: str, split: str) -> List[Dict[str, Any]]:
        """Get dataset files based on directory structure and split"""
        pass

    @abstractmethod
    def get_save_path(self, file_info: Dict[str, Any], base_save_path: str) -> str:
        """Generate appropriate save path for processed files"""
        pass

class BasePatchCutter(ABC):
    """Abstract base class for cutting patches from point clouds"""

    @abstractmethod
    def sample_and_group(self, points: Tensor, npoint: int, nsample: int,
                         indices: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        """Sample points and group into patches"""
        pass

    @abstractmethod
    def cut_patches(self, points: np.ndarray, indices: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Cut patches from point cloud"""
        pass

    @abstractmethod
    def sample_query_points(self, target_points: np.ndarray, query_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample query points for training data"""
        pass