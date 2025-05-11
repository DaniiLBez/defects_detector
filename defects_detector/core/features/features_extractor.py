import numpy as np
import torch
from typing import Optional

from sklearn.decomposition import sparse_encode

from defects_detector.core.features.rgb import RGBFeatureExtractor
from defects_detector.core.features.sdf import SDFFeatureExtractor
from defects_detector.utils.utils import KNNGaussianBlur, get_relative_rgb_f_indices


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
        self.indices.extend(indices)


    def find_nearest_features(self, query_features: torch.Tensor, k: int = 10, **kwargs):
        """Находит ближайшие признаки в банке памяти к заданным признакам запроса"""
        patch_lib = self.sdf_features_tensor
        dist = torch.cdist(query_features.cpu(), patch_lib)
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

        # Сохраняем исходные списки признаков
        torch.save(self.rgb_features, os.path.join(directory, "rgb_features_list.pt"))
        torch.save(self.sdf_features, os.path.join(directory, "sdf_features_list.pt"))
        torch.save(self.indices, os.path.join(directory, "indices_list.pt"))

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

        if not all(map(os.path.exists, [rgb_path, sdf_path, idx_path])):
            raise FileNotFoundError(f"Файлы банка памяти не найдены в {directory}")

        # Создаем новый экземпляр класса
        bank = cls()

        # Загружаем тензоры
        bank.rgb_features_tensor = torch.load(rgb_path)
        bank.sdf_features_tensor = torch.load(sdf_path)
        bank.indices_tensor = torch.load(idx_path)

        # Загружаем исходные списки, если они есть
        rgb_list_path = os.path.join(directory, "rgb_features_list.pt")
        sdf_list_path = os.path.join(directory, "sdf_features_list.pt")
        idx_list_path = os.path.join(directory, "indices_list.pt")

        if all(map(os.path.exists, [rgb_list_path, sdf_list_path, idx_list_path])):
            bank.rgb_features = torch.load(rgb_list_path)
            bank.sdf_features = torch.load(sdf_list_path)
            bank.indices = torch.load(idx_list_path)

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

        self.image_list = []
        self.image_preds, self.pixel_preds = [], []
        self.sdf_image_preds, self.rgb_image_preds = [], []
        self.rgb_pixel_preds, self.sdf_pixel_preds = [], []
        self.origin_f_map = []
        self.rgb, self.sdf, self.bank = rgb, sdf, bank

        self.bias, self.weight = 0, 0
        self.blur = KNNGaussianBlur(4)

    def compute_indices(self, points_idx, patch):
        indices = points_idx[patch].reshape(self.sdf.sdf_model.point_num)
        # compute the correspoding location of rgb features
        return get_relative_rgb_f_indices(indices, self.rgb.image_size, self.rgb.feature_size)

    def add_features_to_memory(self, sample, data_id):
        (image, points, points_idx) = sample
        rgb_features = self.rgb.extract_features(image)
        sdf_features = self.sdf.extract_features(points, points_idx)

        rgb_indices = [data_id * self.rgb.feature_size**2 + self.compute_indices(points_idx, patch) for patch in range(len(points))]
        self.bank.add_features(rgb_features, sdf_features, rgb_indices)

    def foreground_subsampling(self):
        """
        Выполняет выборку переднего плана из эталонных признаков.
        """
        sdf_lib, rgb_lib = self.bank.sdf_features_tensor, self.bank.rgb_features_tensor
        indices = torch.unique(self.bank.indices_tensor)

        self.bank.rgb_features_tensor = rgb_lib[indices]
        self.origin_f_map = np.full(shape=(rgb_lib.shape[0]), fill_value=-1)
        self.origin_f_map[indices] = torch.arange(indices.shape[0], dtype=torch.long)
        self.origin_f_map = torch.Tensor(self.origin_f_map).long()

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

    def find_knn_feature(self, feature, mode='testing'):
        dict_features = []
        knn_idx = self.bank.find_nearest_features(feature, k=10, mode=mode)

        feature = feature.to('cpu')
        for patch in range(knn_idx.shape[0]):
            knn_features = self.bank.sdf_features_tensor[knn_idx[patch]]
            code_matrix = sparse_encode(X=feature[patch].view(1,-1), dictionary=knn_features, algorithm='omp', n_nonzero_coefs=3, alpha=1e-10)
            code_matrix = torch.Tensor(code_matrix)
            sparse_feature = torch.matmul(code_matrix, knn_features) # Sparse representation test rgb feature using the training rgb features stored in the memory.
            dict_features.append(sparse_feature)

        dict_features = torch.cat(dict_features, 0)
        pdist = torch.nn.PairwiseDistance(p=2, eps=1e-12)
        min_val = pdist(feature, dict_features)
        return dict_features, knn_idx, torch.max(min_val)

    def cal_alignment(self):
        """
        Вычисляет параметры для выравнивания распределений SDF и RGB признаков.
        """
        # Вычисление границ распределений
        sdf_lower, sdf_upper = self.compute_distribution_params(self.sdf_pixel_preds)
        rgb_lower, rgb_upper = self.compute_distribution_params(self.rgb_pixel_preds)

        print(f'SDF lower: {sdf_lower}, SDF upper: {sdf_upper}')
        print(f'RGB lower: {rgb_lower}, RGB upper: {rgb_upper}')

        # Предотвращение деления на ноль
        rgb_range = rgb_upper - rgb_lower
        if abs(rgb_range) < 1e-10:
            self.weight = 1.0
        else:
            self.weight = (sdf_upper - sdf_lower) / rgb_range

        self.bias = sdf_lower - self.weight * rgb_lower

    def predict_align_data(self, sample):
        """
        Предсказывает данные для выравнивания.

        Args:
            sample: Входные данные для предсказания
        """
        image, points_all, points_idx = sample

        with torch.no_grad():
            # Извлечение SDF признаков
            sdf_features = self.sdf.extract_features(points_all, points_idx)

            # Поиск ближайших признаков в режиме выравнивания
            dict_features, knn_indices, sdf_score = self.find_knn_feature(sdf_features, mode='alignment')
            sdf_map = self.sdf.get_score_map(dict_features, points_all, points_idx)

            # Извлечение RGB признаков
            rgb_features = self.rgb.extract_features(image)
            rgb_lib_indices = torch.unique(knn_indices.flatten()).tolist()
            rgb_reference_features = self.bank.rgb_features_tensor[torch.unique(
                torch.cat([self.origin_f_map[self.bank.indices[idx]] for idx in rgb_lib_indices], dim=0)
            )]
            rgb_indices = [self.compute_indices(points_idx, patch) for patch in range(len(points_all))]

            rgb_map, rgb_score = self.rgb.compute_anomaly_map(
                rgb_features,
                rgb_reference_features,
                torch.unique(torch.cat(rgb_indices, dim=0)),
                mode='alignment',
            )

            rgb_map, sdf_map = self.blur(rgb_map), self.blur(sdf_map)

            # image_level
            self.sdf_image_preds.append(sdf_score.numpy())
            self.rgb_image_preds.append(rgb_score.numpy())
            # pixel_level
            self.rgb_pixel_preds.extend(rgb_map.flatten().numpy())
            self.sdf_pixel_preds.extend(sdf_map.flatten().numpy())

    def predict(self, sample):
        """
        Предсказывает аномалии на основе входных данных.
        Args:
            sample: Входные данные для предсказания
        """

        image, points_all, points_idx = sample

        self.image_list.append(image.squeeze(0).cpu().numpy())
        with torch.no_grad():
            # Извлечение SDF признаков
            sdf_features = self.sdf.extract_features(points_all, points_idx)

            # Поиск ближайших признаков в режиме выравнивания
            dict_features, knn_indices, sdf_score = self.find_knn_feature(sdf_features, mode='alignment')
            sdf_map = self.sdf.get_score_map(dict_features, points_all, points_idx)

            # Извлечение RGB признаков
            rgb_features = self.rgb.extract_features(image)
            rgb_lib_indices = torch.unique(knn_indices.flatten()).tolist()
            rgb_reference_features = self.bank.rgb_features_tensor[torch.unique(
                torch.cat([self.origin_f_map[self.bank.indices[idx]] for idx in rgb_lib_indices], dim=0)
            )]
            rgb_indices = [self.compute_indices(points_idx, patch) for patch in range(len(points_all))]

            rgb_map, rgb_score = self.rgb.compute_anomaly_map(
                rgb_features,
                rgb_reference_features,
                torch.unique(torch.cat(rgb_indices, dim=0)),
            )

            image_score = sdf_score * rgb_score
            new_rgb_map = rgb_map * self.weight + self.bias
            new_rgb_map = torch.clip(new_rgb_map, min=0, max=new_rgb_map.max())
            pixel_map = torch.maximum(new_rgb_map, sdf_map)

            sdf_map, rgb_map, pixel_map = map(self.blur, (sdf_map, rgb_map, pixel_map))

            ##### Record Image Level Score #####
            self.sdf_image_preds.append(sdf_score.numpy())
            self.rgb_image_preds.append(rgb_score.numpy())
            self.image_preds.append(image_score.numpy())

            ##### Record Pixel Level Score #####
            self.sdf_pixel_preds.extend(sdf_map.flatten().numpy())
            self.rgb_pixel_preds.extend(rgb_map.flatten().numpy())
            self.pixel_preds.extend(pixel_map.flatten().numpy())


