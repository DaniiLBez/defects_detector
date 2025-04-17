from abc import ABC, abstractmethod

import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional

from defects_detector.core.features.rgb import RGBFeatureExtractor
from defects_detector.core.features.sdf import SDFFeatureExtractor
from defects_detector.utils.utils import KNNGaussianBlur, get_relative_rgb_f_indices


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

    def add_features(self, rgb_features, sdf_features, indices) -> None:
        """Добавляет признаки в банк памяти"""
        self.rgb_features.append(rgb_features.to("cpu"))
        self.sdf_features.append(sdf_features.to("cpu"))
        self.indices.append(indices)


    def find_nearest_features(self, query_features: torch.Tensor, k: int = 10, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Находит ближайшие признаки в банке памяти к заданным признакам запроса"""
        patch_lib = self.sdf_features_tensor
        dist = torch.cdist(query_features, patch_lib)
        _, knn_idx = torch.topk(dist, k=k + 1, largest=False)
        return knn_idx[:, 1:] if kwargs.get("mode", "testing") == 'alignment' else knn_idx[:, :-1]

    def finalize(self) -> None:
        """
        Завершает добавление признаков и объединяет все признаки в тензоры для ускорения поиска.

        Raises:
            RuntimeError: Если банк памяти уже финализирован
        """
        if self.is_finalized:
            raise RuntimeError("Банк памяти уже финализирован")

        # Проверка наличия признаков
        if not self.rgb_features or not self.sdf_features or not self.indices:
            raise ValueError("Нельзя финализировать пустой банк памяти")

        # Объединяем признаки в тензоры
        self.rgb_features_tensor = torch.cat(self.rgb_features, dim=0)
        self.sdf_features_tensor = torch.cat(self.sdf_features, dim=0)
        self.indices_tensor = torch.cat(self.indices, dim=0)

        self.is_finalized = True

    def save(self, directory: str) -> None:
        """
        Сохраняет банк памяти в указанную директорию.

        Args:
            directory: Путь к директории для сохранения

        Raises:
            RuntimeError: Если банк памяти не финализирован перед сохранением
        """
        import os

        if not self.is_finalized:
            raise RuntimeError("Банк памяти должен быть финализирован перед сохранением")

        # Создаем директорию, если она не существует
        os.makedirs(directory, exist_ok=True)

        # Сохраняем тензоры
        torch.save(self.rgb_features_tensor, os.path.join(directory, "rgb_features.pt"))
        torch.save(self.sdf_features_tensor, os.path.join(directory, "sdf_features.pt"))
        torch.save(self.indices_tensor, os.path.join(directory, "indices.pt"))

    @classmethod
    def load(cls, path: str, name: Optional[str] = None) -> "MemoryBank":
        """
        Загружает банк памяти из указанного пути.

        Args:
            path: Путь к файлу банка памяти
            name: Опциональное имя банка памяти

        Returns:
            Экземпляр класса MemoryBank с загруженными признаками

        Raises:
            FileNotFoundError: Если файлы не найдены по указанному пути
        """
        import os

        # Выбираем директорию для загрузки
        directory = path
        if name is not None:
            directory = os.path.join(path, name)

        # Проверяем существование файлов
        rgb_path = os.path.join(directory, "rgb_features.pt")
        sdf_path = os.path.join(directory, "sdf_features.pt")
        idx_path = os.path.join(directory, "indices.pt")

        if not (os.path.exists(rgb_path) and os.path.exists(sdf_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(f"Файлы банка памяти не найдены в {directory}")

        # Создаем новый экземпляр класса
        bank = cls()

        # Загружаем тензоры
        bank.rgb_features_tensor = torch.load(rgb_path)
        bank.sdf_features_tensor = torch.load(sdf_path)
        bank.indices_tensor = torch.load(idx_path)

        # Списки оставляем пустыми, так как мы загрузили уже объединенные тензоры
        bank.is_finalized = True

        return bank


class FeatureExtractor:
    """Экстрактор признаков, объединяющий RGB и SDF подходы"""

    def __init__(self, rgb: RGBFeatureExtractor, sdf: SDFFeatureExtractor, bank: MemoryBank):
        """
                Инициализация экстрактора признаков.

                Args:
                    rgb: Экземпляр RGBFeatureExtractor для работы с RGB признаками.
                    sdf: Экземпляр SDFFeatureExtractor для работы с SDF признаками.
                    bank: Экземпляр MemoryBank для хранения эталонных признаков.
                """
        self.rgb_pixel_preds, self.sdf_pixel_preds = [], []
        self.origin_f_map = []
        self.rgb, self.sdf, self.bank = rgb, sdf, bank

        self.bias, self.weight = 0, 0
        self.blur = KNNGaussianBlur(4)

    def add_features_to_memory(self, sample, data_id):
        (image, points, points_idx) = sample
        rgb_features = self.rgb.extract_features(image)
        sdf_features = self.sdf.extract_features(points, points_idx)

        def compute_indices(patch):
            indices = points_idx[patch].reshape(self.sdf.sdf_model.points_num)
            # compute the correspoding location of rgb features
            return get_relative_rgb_f_indices(indices, self.rgb.image_size, self.rgb.feature_size)

        rgb_indices = [data_id * self.rgb.feature_size**2 + compute_indices(patch) for patch in range(len(points))]
        self.bank.add_features(rgb_features, sdf_features, rgb_indices)

    def foreground_subsampling(self):
        """
        Выполняет выборку переднего плана из эталонных признаков.
        """
        sdf_lib, rgb_lib = self.bank.sdf_features_tensor, self.bank.rgb_features_tensor
        indices = torch.unique(self.bank.indices_tensor)

        self.bank.rgb_features_tensor = rgb_lib[indices]
        self.origin_f_map[indices] = torch.arange(indices.shape[0], dtype=torch.long)

    @staticmethod
    def compute_distribution_params(pixel_preds):
        """
        Вычисляет параметры распределения значений предсказаний.

        Args:
            pixel_preds: Список предсказаний на уровне пикселей

        Returns:
            tuple: Нижняя и верхняя границы распределения (среднее ± 3*std)
        """
        pixel_map = np.array(pixel_preds)
        non_zero_indices = np.nonzero(pixel_map)

        if len(non_zero_indices[0]) == 0:
            return 0.0, 1.0

        non_zero_values = pixel_map[non_zero_indices]

        mean_value = np.mean(non_zero_values)
        std_value = np.std(non_zero_values)

        lower_bound = mean_value - 3 * std_value
        upper_bound = mean_value + 3 * std_value

        return lower_bound, upper_bound

    def cal_alignment(self):
        """
        Вычисляет параметры для выравнивания распределений SDF и RGB признаков.
        """
        # Вычисление границ распределений
        sdf_lower, sdf_upper = self.compute_distribution_params(self.sdf_pixel_preds)
        rgb_lower, rgb_upper = self.compute_distribution_params(self.rgb_pixel_preds)

        # Предотвращение деления на ноль
        rgb_range = rgb_upper - rgb_lower
        if abs(rgb_range) < 1e-10:
            self.weight = 1.0
        else:
            self.weight = (sdf_upper - sdf_lower) / rgb_range

        self.bias = sdf_lower - self.weight * rgb_lower

