import torch
import timm

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