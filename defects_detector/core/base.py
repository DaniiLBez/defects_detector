from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class DefectDetectionBase(ABC):
    """Base interface for defect detection services"""

    @abstractmethod
    def fit(self, data_loader: Any) -> None:
        """Extract features from training data"""
        pass

    @abstractmethod
    def align(self, data_loader: Any) -> Tuple[float, float]:
        """Compute weight and bias for alignment"""
        pass

    @abstractmethod
    def predict(self) -> Dict[str, Any]:
        """Predict defects in a single sample"""
        pass


class BaseFeatureExtractor(ABC):
    """Базовый абстрактный класс для извлечения признаков"""

    @abstractmethod
    def extract_features(self, *args: Any):
        """Извлекает признаки из входных данных"""
        pass

    @abstractmethod
    def compute_anomaly_map(self, *args: Any):
        """Вычисляет карту аномалий на основе сравнения признаков"""
        pass


class DataLoaderBase(ABC):
    """Базовый абстрактный класс для загрузки данных"""

    @abstractmethod
    def load_data(self, *args: Any):
        """Загружает данные из источника"""
        pass