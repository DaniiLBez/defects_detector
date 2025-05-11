import os
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from defects_detector.core.base import DefectDetectionBase
from defects_detector.core.data import MVTecDataLoader
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
        self.weight, self.bias = 1.0, 0.0
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

        memory_bank_path = self.config.get("memory_bank", None)
        # Банк памяти для хранения эталонных признаков
        self.memory_bank = MemoryBank()
        if memory_bank_path:
            # Загрузка предварительно сохраненного банка памяти
            self.memory_bank = self.memory_bank.load(memory_bank_path)

        # Параметры загрузчика данных
        dataset = MVTecDataLoader(self.image_size, config.get("image_path"), config.get("grid_path"))
        self.data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

        # Инициализация моделей и экстракторов признаков
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Инициализация моделей RGB и SDF"""
        # RGB модель
        self.rgb_model = RGBModel(device=self.device)

        # SDF модель
        sdf_checkpoint_path = self.config.get("sdf_checkpoint_path", "")
        if not sdf_checkpoint_path:
            raise ValueError("Необходимо указать путь к весам SDF модели")

        self.sdf_model = SDFModel(point_num=self.point_num)
        checkpoint = torch.load(sdf_checkpoint_path, map_location=self.device)
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])

        # Экстракторы признаков
        rgb_extractor = RGBFeatureExtractor(
            rgb_model=self.rgb_model,
            image_size=self.image_size,
            feature_size=self.feature_size
        )

        sdf_extractor = SDFFeatureExtractor(
            model=self.sdf_model,
            image_size=self.image_size,
            feature_size=self.feature_size,
            batch_size=1
        )

        # Объединенный экстрактор признаков
        self.feature_extractor = FeatureExtractor(
            rgb=rgb_extractor,
            sdf=sdf_extractor,
            bank=self.memory_bank
        )

    def fit(self, data_loader: Any) -> None:
        """
        Извлечение признаков из обучающих данных и их сохранение в банк памяти.

        Args:
            data_loader: Загрузчик обучающих данных
        """
        if self.memory_bank.is_finalized:
            # Подвыборка переднего плана для улучшения эффективности
            self.feature_extractor.foreground_subsampling()
            return

        with torch.no_grad():
            for train_id, (sample, _) in enumerate(tqdm(data_loader, desc='Извлечение признаков из обучающих данных')):
                # Добавление признаков в банк памяти
                self.feature_extractor.add_features_to_memory(sample, train_id)

        # Финализация банка памяти для последующих быстрых поисков
        self.memory_bank.finalize()
        # Сохранение банка памяти, если указана директория
        if self.output_dir and self.config.get("save_memory_bank", False):
            self.memory_bank.save(os.path.join(self.output_dir, "memory_bank"))

        # Подвыборка переднего плана для улучшения эффективности
        self.feature_extractor.foreground_subsampling()

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
            for align_id, (sample, _) in enumerate(tqdm(data_loader, desc="Извлечение признаков для выравнивания", total=25)):
                if align_id >= 25:  # используем ограниченное количество образцов для выравнивания
                    break
                self.feature_extractor.predict_align_data(sample)

        # Вычисление параметров выравнивания
        self.feature_extractor.cal_alignment()
        self.weight = self.feature_extractor.weight
        self.bias = self.feature_extractor.bias

        # Сохранение параметров, если указана директория
        if self.output_dir:
            with open(os.path.join(self.output_dir, "alignment.txt"), "a") as f:
                f.write(f"Weight: {self.weight}, Bias: {self.bias}\n")

        return self.weight, self.bias

    def predict(self):
        """
        Предсказание дефектов в загруженных образцах.
        """
        for i, sample in enumerate(tqdm(self.data_loader, desc="Предсказание")):
            # Предсказание с использованием экстрактора признаков
            with torch.no_grad():
                self.feature_extractor.predict(sample)

