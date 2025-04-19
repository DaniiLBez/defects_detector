import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from tqdm import tqdm

from defects_detector.core.base import DefectDetectionBase
from defects_detector.core.features.features_extractor import FeatureExtractor, MemoryBank
from defects_detector.core.features.rgb import RGBFeatureExtractor, RGBModel
from defects_detector.core.features.sdf import SDFFeatureExtractor, SDFModel


class ShapeGuidedDetector(DefectDetectionBase):
    """Реализация обнаружения дефектов с использованием метода ShapeGuided"""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация детектора дефектов.

        Args:
            config: Словарь с параметрами конфигурации
                image_size: Размер изображения
                feature_size: Размер карты признаков 
                point_num: Количество точек в облаке
                rgb_backbone: Название архитектуры для RGB модели
                sdf_checkpoint_path: Путь к весам SDF модели
                output_dir: Директория для сохранения результатов
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Параметры
        self.image_size = config.get("image_size", 224)
        self.feature_size = config.get("feature_size", 28)
        self.point_num = config.get("point_num", 500)

        # Создание директории для вывода, если она указана
        self.output_dir = config.get("output_dir", None)
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Инициализация моделей и экстракторов признаков
        self._initialize_models()

        # Банк памяти для хранения эталонных признаков
        self.memory_bank = MemoryBank()

        # Выравнивание между RGB и SDF признаками
        self.weight = 1.0
        self.bias = 0.0

    def _initialize_models(self) -> None:
        """Инициализация моделей RGB и SDF"""
        # RGB модель
        rgb_backbone = self.config.get("rgb_backbone", "wide_resnet50_2")
        self.rgb_model = RGBModel(
            device=self.device,
            backbone_name=rgb_backbone,
            out_indices=(1, 2)
        )

        # SDF модель
        sdf_checkpoint_path = self.config.get("sdf_checkpoint_path", "")
        if not sdf_checkpoint_path:
            raise ValueError("Необходимо указать путь к весам SDF модели")

        self.sdf_model = SDFModel(point_num=self.point_num)
        checkpoint = torch.load(sdf_checkpoint_path, map_location=self.device)
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])
        self.sdf_model.freeze_model()

        # Экстракторы признаков
        self.rgb_extractor = RGBFeatureExtractor(
            rgb_model=self.rgb_model,
            image_size=self.image_size,
            feature_size=self.feature_size
        )

        self.sdf_extractor = SDFFeatureExtractor(
            model=self.sdf_model,
            image_size=self.image_size,
            feature_size=self.feature_size,
            batch_size=1
        )

        # Объединенный экстрактор признаков
        self.feature_extractor = FeatureExtractor(
            rgb=self.rgb_extractor,
            sdf=self.sdf_extractor,
            bank=self.memory_bank
        )

    def fit(self, data_loader: Any) -> None:
        """
        Извлечение признаков из обучающих данных и их сохранение в банк памяти.

        Args:
            data_loader: Загрузчик обучающих данных
        """
        with torch.no_grad():
            for train_id, (sample, _) in enumerate(tqdm(data_loader, desc='Извлечение признаков из обучающих данных')):
                # Добавление признаков в банк памяти
                self.feature_extractor.add_features_to_memory(sample, train_id)

        # Финализация банка памяти для последующих быстрых поисков
        self.memory_bank.finalize()

        # Подвыборка переднего плана для улучшения эффективности
        self.feature_extractor.foreground_subsampling()

        # Сохранение банка памяти, если указана директория
        if self.output_dir:
            self.memory_bank.save(os.path.join(self.output_dir, "memory_bank"))

    def align(self, data_loader: Any) -> Tuple[float, float]:
        """
        Вычисление веса и смещения для выравнивания RGB и SDF признаков.

        Args:
            data_loader: Загрузчик данных для выравнивания

        Returns:
            Кортеж (вес, смещение) для выравнивания признаков
        """
        print('Вычисление параметров выравнивания признаков')
        with torch.no_grad():
            for align_id, (sample, _) in enumerate(data_loader):
                if align_id < 25:  # используем ограниченное количество образцов для выравнивания
                    self.feature_extractor.predict_align_data(sample)
                else:
                    break

        # Вычисление параметров выравнивания
        self.feature_extractor.cal_alignment()
        self.weight = self.feature_extractor.weight
        self.bias = self.feature_extractor.bias

        # Сохранение параметров, если указана директория
        if self.output_dir:
            with open(os.path.join(self.output_dir, "alignment.txt"), "a") as f:
                f.write(f"Weight: {self.weight}, Bias: {self.bias}\n")

        return self.weight, self.bias

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание дефектов в одном образце.

        Args:
            data: Словарь с данными для предсказания
                image: RGB изображение (тензор или путь к файлу)
                depth: Карта глубины (тензор или путь к файлу)

        Returns:
            Словарь с результатами предсказания
                anomaly_map: Карта аномалий
                anomaly_score: Общая оценка аномальности
                sdf_score: Оценка аномальности по SDF
                rgb_score: Оценка аномальности по RGB
                overlay: Наложенная визуализация (если визуализация включена)
        """
        # Подготовка данных в зависимости от их типа
        sample = self._prepare_sample(data)

        # Предсказание с использованием экстрактора признаков
        with torch.no_grad():
            self.feature_extractor.predict(sample)

        # Извлечение результатов
        anomaly_map = torch.tensor(self.feature_extractor.pixel_preds[-self.image_size * self.image_size:])
        anomaly_map = anomaly_map.reshape(1, 1, self.image_size, self.image_size)

        rgb_score = self.feature_extractor.rgb_image_preds[-1]
        sdf_score = self.feature_extractor.sdf_image_preds[-1]
        anomaly_score = self.feature_extractor.image_preds[-1]

        # Создание результата
        result = {
            'anomaly_map': anomaly_map,
            'anomaly_score': float(anomaly_score),
            'rgb_score': float(rgb_score),
            'sdf_score': float(sdf_score),
        }

        # Добавление визуализации, если требуется
        if self.config.get("visualize", False):
            overlay = self._create_visualization(data.get("image"), anomaly_map)
            result['overlay'] = overlay

        return result

    # def _prepare_sample(self, data: Dict[str, Any]) -> Tuple:
    #     """
    #     Подготовка образца для предсказания из различных типов входных данных.
    #
    #     Args:
    #         data: Словарь с входными данными
    #
    #     Returns:
    #         Кортеж (изображение, точки, индексы точек)
    #     """
    #     # Получение изображения
    #     image = data.get("image")
    #     if isinstance(image, str):
    #         # Загрузка изображения из файла
    #         from PIL import Image
    #         import numpy as np
    #         img = Image.open(image).convert('RGB')
    #         image = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    #         image = image.unsqueeze(0)
    #
    #     # Получение карты глубины и конвертация в облако точек
    #     depth_map = data.get("depth")
    #     if isinstance(depth_map, str):
    #         # Загрузка карты глубины из файла
    #         import cv2
    #         depth_map = cv2.imread(depth_map, cv2.IMREAD_UNCHANGED)
    #         if depth_map.ndim == 3:
    #             depth_map = depth_map[:, :, 0]
    #
    #     # Преобразование карты глубины в облако точек
    #     points_all, points_idx = depth_to_point_cloud(depth_map, self.point_num)
    #
    #     return image, points_all, points_idx

    # def _create_visualization(self, image, anomaly_map):
    #     """
    #     Создание визуализации с наложением карты аномалий на изображение.
    #
    #     Args:
    #         image: Исходное изображение
    #         anomaly_map: Карта аномалий
    #
    #     Returns:
    #         Изображение с наложенной картой аномалий
    #     """
    #     import cv2
    #     import numpy as np
    #
    #     # Преобразование тензоров в numpy массивы при необходимости
    #     if isinstance(image, torch.Tensor):
    #         image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    #         if image.shape[2] == 3:
    #             image = (image * 255).astype(np.uint8)
    #
    #     if isinstance(anomaly_map, torch.Tensor):
    #         anomaly_map = anomaly_map.squeeze().cpu().numpy()
    #
    #     # Нормализация карты аномалий для визуализации
    #     anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    #     anomaly_map_colored = cv2.applyColorMap((anomaly_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #
    #     # Наложение карты аномалий на изображение
    #     overlay = cv2.addWeighted(image, 0.7, anomaly_map_colored, 0.3, 0)
    #
    #     return overlay

    def save(self, path: str) -> None:
        """
        Сохранение модели и параметров детектора.

        Args:
            path: Путь для сохранения
        """
        os.makedirs(path, exist_ok=True)

        # Сохранение банка памяти
        self.memory_bank.save(os.path.join(path, "memory_bank"))

        # Сохранение параметров выравнивания
        alignment = {'weight': self.weight, 'bias': self.bias}
        torch.save(alignment, os.path.join(path, "alignment.pt"))

        # Сохранение конфигурации
        config = {
            'image_size': self.image_size,
            'feature_size': self.feature_size,
            'point_num': self.point_num
        }
        torch.save(config, os.path.join(path, "config.pt"))

    @classmethod
    def load(cls, path: str) -> "ShapeGuidedDetector":
        """
        Загрузка детектора из сохраненной модели.

        Args:
            path: Путь к сохраненной модели

        Returns:
            Экземпляр ShapeGuidedDetector
        """
        # Загрузка конфигурации
        config = torch.load(os.path.join(path, "config.pt"))

        # Добавление пути к весам SDF модели из конфигурации при загрузке
        config['sdf_checkpoint_path'] = os.path.join(path, "sdf_model.pt")

        # Создание экземпляра детектора
        detector = cls(config)

        # Загрузка банка памяти
        detector.memory_bank = MemoryBank.load(path, "memory_bank")

        # Загрузка параметров выравнивания
        alignment = torch.load(os.path.join(path, "alignment.pt"))
        detector.weight = alignment['weight']
        detector.bias = alignment['bias']
        detector.feature_extractor.weight = detector.weight
        detector.feature_extractor.bias = detector.bias

        return detector