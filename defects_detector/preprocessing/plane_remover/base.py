from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

class PreprocessingServiceBase(ABC):
    """Base service for preprocessing point clouds"""

    @abstractmethod
    def preprocess_all(self, dataset_path: Union[str, Path], verbose: bool = True) -> int:
        """Process all point cloud files in the dataset recursively"""
        pass

    @abstractmethod
    def preprocess_file(self, file_path: Union[str, Path]) -> None:
        """Preprocess a single point cloud file"""
        pass