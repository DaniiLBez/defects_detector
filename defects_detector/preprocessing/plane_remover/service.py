from pathlib import Path
from typing import Union

from defects_detector.preprocessing.plane_remover.base import PreprocessingServiceBase


class PreprocessingService:
    """Factory class for preprocessing services"""

    def __init__(self, preprocessor: PreprocessingServiceBase):
        """Default constructor creates MVTec3D service for backward compatibility"""
        self._service = preprocessor

    def preprocess_all(self, dataset_path: Union[str, Path], verbose: bool = True) -> int:
        """Delegate to underlying service implementation"""
        return self._service.preprocess_all(dataset_path, verbose)

    def preprocess_file(self, file_path: Union[str, Path]) -> None:
        """Delegate to underlying service implementation"""
        self._service.preprocess_file(file_path)