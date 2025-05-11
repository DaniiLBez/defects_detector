import numpy as np
import torch
import timm
from sklearn.decomposition import sparse_encode

from defects_detector.core.base import BaseFeatureExtractor


class RGBModel(torch.nn.Module):
    """
    RGB модель для извлечения признаков из изображений с использованием предобученной сети.
    """
    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(1, 2), checkpoint_path='',
                 pool_last=False):
        """
        Инициализация RGB модели.

        Args:
            device: Устройство для вычислений (cuda/cpu)
            backbone_name: Название базовой модели
            out_indices: Индексы слоев, с которых нужно получать промежуточные признаки
            checkpoint_path: Путь к предобученным весам (если есть)
            pool_last: Применять ли пулинг к последнему слою признаков
        """
        super().__init__()
        # Настройка параметров для извлечения признаков
        kwargs = {'features_only': bool(out_indices)}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        # Создание базовой модели с помощью timm
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=True,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
        self._device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    @property
    def device(self):
        """Возвращает текущее устройство модели"""
        return self._device

    def forward(self, x):
        """
        Прямой проход через модель.

        Args:
            x: Входное изображение

        Returns:
            Список карт признаков различных уровней
        """
        x = x.to(self.device)

        # Прямой проход через базовую модель
        features = self.backbone(x)

        # Пулинг последнего слоя, если включен
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        return features

    def _freeze_layer(self, layer_name):
        """Вспомогательный метод для заморозки параметров конкретного слоя"""
        if hasattr(self.backbone, layer_name):
            for param in getattr(self.backbone, layer_name).parameters():
                param.requires_grad = False

    def freeze_parameters(self, layers, freeze_bn=False):
        """
        Заморозка параметров модели.

        Args:
            layers: Список слоев, которые НЕ нужно замораживать
            freeze_bn: Нужно ли замораживать слои батч-нормализации
        """
        layers = [str(layer) for layer in layers]

        # Заморозка основных блоков
        if '1' not in layers:
            self._freeze_layer('conv1')
            self._freeze_layer('bn1')
            self._freeze_layer('layer1')

        if '2' not in layers:
            self._freeze_layer('layer2')

        if '3' not in layers:
            self._freeze_layer('layer3')

        if '4' not in layers:
            self._freeze_layer('layer4')

        if '-1' not in layers:
            self._freeze_layer('fc')

        # Заморозка слоев батч-нормализации, если требуется
        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, (torch.nn.modules.BatchNorm1d,
                                      torch.nn.modules.BatchNorm2d,
                                      torch.nn.modules.BatchNorm3d)):
                    module.eval()


class RGBFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, rgb_model: RGBModel, image_size: int = 224, feature_size: int = 28):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_model = rgb_model.to(self.device)
        self.rgb_model.freeze_parameters(layers=[], freeze_bn=True)

        self.image_size = image_size
        self.feature_size = feature_size

        self.resize = torch.nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.average = torch.nn.AvgPool2d(3, stride=1)


    def extract_features(self, data):
        """
        Извлекает признаки из входных данных.

        Args:
            data: Входные данные (изображение)

        Returns:
            Извлеченные признаки
        """
        with torch.no_grad():
            features = self.rgb_model(data)

        resized_maps = [self.resize(self.average(fmap)) for fmap in features]
        rgb_patch = torch.cat(resized_maps, 1)
        return rgb_patch.reshape(rgb_patch.shape[1], -1).T.to("cpu")

    def compute_anomaly_map(self,
                            features: torch.Tensor,
                            reference_features: torch.Tensor,
                            foreground_indices: torch.Tensor,
                            **kwargs):
        """
        Вычисляет карту аномалий на основе сравнения признаков.

        Args:
            features: Извлеченные признаки
            reference_features: Эталонные признаки
            foreground_indices: Индексы переднего плана [P], где P - количество пикселей переднего плана

        Returns:
            Карта аномалий
        """
        anomaly_map = torch.zeros(self.feature_size * self.feature_size)
        pdist = torch.nn.PairwiseDistance(p=2, eps=1e-12)

        features = features[foreground_indices]

        distance = torch.cdist(features, reference_features)
        knn_val, knn_idx = torch.topk(distance, k=5+1, largest=False)

        (knn_idx, min_value) = (knn_idx[:, 1:], knn_val[:, 1]) if kwargs.get("mode", "testing") == 'alignment' \
                                                                    else (knn_idx[:,:-1], knn_val[:, 0])

        reference_features_cpu, features_cpu = reference_features.cpu(), features.cpu()
        reconstructed_features = []
        for patch in range(knn_idx.shape[0]):
            # Словарь из ближайших эталонных признаков
            dictionary = reference_features_cpu[knn_idx[patch]]

            # Решаем задачу оптимизации для разреженного кода
            code = sparse_encode(
                X=features_cpu[patch].view(1,-1),
                dictionary=dictionary,
                algorithm='omp', n_nonzero_coefs=3, alpha=1e-10
            )

            # Реконструируем признак с помощью найденного кода
            reconstructed = torch.from_numpy(
                np.dot(code, dictionary)
            ).float()

            reconstructed_features.append(reconstructed)

        reconstructed_features = torch.cat(reconstructed_features, 0)
        anomaly_map[foreground_indices] = pdist(features_cpu, reconstructed_features)

        # Преобразуем карту аномалий к нужному формату и размеру
        anomaly_map = anomaly_map.view(1, 1, self.feature_size, self.feature_size)
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

        # Вычисляем общую оценку аномальности как максимальное значение карты
        anomaly_score = torch.max(anomaly_map)

        return anomaly_map, anomaly_score