from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, List, Tuple, Optional


class BaseFeatureExtractor(ABC):
    """Базовый абстрактный класс для извлечения признаков"""

    @abstractmethod
    def extract_features(self, data: Any) -> torch.Tensor:
        """Извлекает признаки из входных данных"""
        pass

    @abstractmethod
    def compute_anomaly_map(self, features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
        """Вычисляет карту аномалий на основе сравнения признаков"""
        pass


class MemoryBank:
    """Класс для хранения и поиска эталонных признаков"""

    def __init__(self):
        self.rgb_features = []
        self.sdf_features = []
        self.indices = []
        self.rgb_features_tensor = None
        self.sdf_features_tensor = None
        self.indices_tensor = None
        self.is_finalized = False

    def add_features(self, features: torch.Tensor, indices: Optional[torch.Tensor] = None) -> None:
        """Добавляет признаки в банк памяти"""
        pass

    def find_nearest_features(self, query_features: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Находит ближайшие признаки в банке памяти к заданным признакам запроса"""
        pass

    def finalize(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, name: Optional[str] = None) -> "MemoryBank":
        pass


class FeatureExtractor:
    """Экстрактор признаков, объединяющий RGB и SDF подходы"""

    def __init__(self, feature_dim: int, n_components: int,
                 method: str = 'RGB_SDF', image_size: int = 224,
                 point_num: int = 1024):
        """
        Инициализирует экстрактор признаков

        Args:
            feature_dim: Размерность признакового пространства
            n_components: Количество компонент для разреженного кодирования
            method: Метод извлечения признаков ('RGB', 'SDF', 'RGB_SDF')
            image_size: Размер входного изображения
            point_num: Количество точек в облаке точек
        """
        pass

    def extract_rgb_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Извлекает признаки из RGB изображения

        Args:
            image: Входное изображение

        Returns:
            RGB признаки
        """
        pass

    def extract_sdf_features(self, points: torch.Tensor, points_idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Извлекает признаки из облака точек

        Args:
            points: Облако точек
            points_idx: Индексы точек

        Returns:
            SDF признаки и их индексы
        """
        pass

    def compute_anomaly_map(self, features: torch.Tensor, reference_features: torch.Tensor,
                           method: str = 'sparse_dict') -> Tuple[torch.Tensor, float]:
        """
        Вычисляет карту аномалий на основе сравнения признаков

        Args:
            features: Признаки тестового образца
            reference_features: Эталонные признаки
            method: Метод сравнения ('sparse_dict', 'knn')

        Returns:
            Карта аномалий и глобальная оценка аномальности
        """
        pass