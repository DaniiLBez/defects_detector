from abc import ABC

import numpy as np
import cv2
from skimage import morphology, measure
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional

from defects_detector.position_estimate.data import DataLoader

def area_scan_division_(camera_params: Dict[str, float]) -> Dict[str, float]:
    """
    Стратегия для преобразования параметров камеры в формат, используемый в CameraPositioningService.
    Args:
        camera_params: исходные параметры камеры
    Returns:
        camera_params: параметры камеры в формате, используемом в CameraPositioningService
    """
    if any([
        camera_params.get("focus", None),
        camera_params.get("sx", None),
        camera_params.get("sy", None),
        camera_params.get("cx", None),
        camera_params.get("cy", None),
    ]) is None:
        raise ValueError("Camera parameters must contain focus, sx, sy, cx, and cy values.")

    if not camera_params.get('fx', None):
        camera_params.setdefault("fx", camera_params.get("focus", 1.0) * camera_params.get("sx", 1.0))
    if not camera_params.get('fy', None):
        camera_params.setdefault("fy", camera_params.get("focus", 1.0) * camera_params.get("sy", 1.0))
    return camera_params


CAMERA_STRATEGIES = {
    "area_scan_division": area_scan_division_,
}

class CameraPositioningService:
    """
    Сервис для определения оптимальных позиций камеры для съемки дефектов.
    """

    def __init__(self, camera_params: Dict[str, float], data: DataLoader, cluster_eps: float = 0.05, min_samples: int = 5):
        """
        Инициализация сервиса.

        Args:
            camera_params: внутренние параметры камеры
            data: Карты глубины и карты дефектов
        """
        self.camera_params = CAMERA_STRATEGIES[camera_params.get("camera_type", "area_scan_division")](camera_params)
        self.data = data


    @staticmethod
    def extract_defects(score_map, threshold=None, min_area=5):
        """
        Выделение областей с дефектами из score_map

        Args:
            score_map: карта аномалий
            threshold: пороговое значение (если None, используется метод Отсу)
            min_area: минимальная площадь дефекта для фильтрации шума

        Returns:
            binary_map: бинаризованная карта дефектов
            regions: список обнаруженных областей с их характеристиками
        """
        # Нормализация карты аномалий
        norm_score_map = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Автоматическое определение порога, если не задан
        if threshold is None:
            threshold, _ = cv2.threshold(norm_score_map, 0, 255, cv2.THRESH_OTSU)
            threshold = threshold / 255.0

        # Бинаризация карты аномалий
        binary_map = (score_map > threshold).astype(np.uint8)

        # Морфологическая обработка для удаления шума
        kernel = morphology.disk(2)
        binary_map = morphology.opening(binary_map, kernel)

        # Поиск связных компонент
        labeled_map = measure.label(binary_map)
        regions = measure.regionprops(labeled_map)

        # Фильтрация маленьких областей
        filtered_regions = [region for region in regions if region.area >= min_area]

        return binary_map, filtered_regions

    def map_2d_to_3d(self, defects: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        Преобразует 2D координаты дефектов в 3D координаты.

        Args:
            defects: список дефектов с координатами center
            depth_map: карта глубины (2D массив)

        Returns:
            список дефектов с добавленными 3D координатами
        """
        fx, fy = self.camera_params['fx'], self.camera_params['fy']
        cx, cy = self.camera_params['cx'], self.camera_params['cy']

        for defect in defects:
            x, y = defect['center']
            x, y = int(x), int(y)

            # Получение глубины в точке дефекта
            z = depth_map[y, x]

            # Преобразование в 3D координаты
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            z_3d = z

            defect['position_3d'] = (x_3d, y_3d, z_3d)

        return defects

    def cluster_defects(self, defects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Кластеризует дефекты на основе их 3D координат.

        Args:
            defects: список дефектов с 3D координатами

        Returns:
            дефекты с метками кластеров и количество кластеров
        """
        if not defects:
            return defects, 0

        # Извлечение 3D координат
        positions = np.array([defect['position_3d'] for defect in defects])

        # Кластеризация
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples).fit(positions)
        labels = clustering.labels_

        # Добавление меток кластеров к дефектам
        for i, defect in enumerate(defects):
            defect['cluster'] = int(labels[i])

        num_clusters = np.max(labels) + 1 if len(labels) > 0 and np.max(labels) >= 0 else 0
        return defects, int(num_clusters)

    def compute_surface_normals(self, defects: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        Вычисляет нормали поверхности для каждого дефекта.

        Args:
            defects: список дефектов с 2D координатами
            depth_map: карта глубины

        Returns:
            дефекты с добавленными нормалями
        """
        # Вычисление градиентов карты глубины
        sobelx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

        for defect in defects:
            x, y = defect['center']
            x, y = int(x), int(y)

            # Получение градиентов в точке дефекта
            dx = sobelx[y, x]
            dy = sobely[y, x]

            # Нормаль поверхности (-dx, -dy, 1), нормализованная
            normal = np.array([-dx, -dy, 1.0])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            defect['normal'] = normal

        return defects

    def optimal_camera_positions(self, defects: List[Dict[str, Any]],
                              num_clusters: int,
                              current_camera_position: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Определяет оптимальные положения камеры для съемки кластеров дефектов.

        Args:
            defects: список дефектов с метками кластеров и нормалями
            num_clusters: количество кластеров
            current_camera_position: текущее положение камеры

        Returns:
            список оптимальных положений камеры для каждого кластера
        """
        camera_positions = []

        for cluster_id in range(num_clusters):
            cluster_defects = [d for d in defects if d['cluster'] == cluster_id]

            if not cluster_defects:
                continue

            # Центр кластера
            positions = np.array([d['position_3d'] for d in cluster_defects])
            center = np.mean(positions, axis=0)

            # Средняя нормаль кластера
            normals = np.array([d['normal'] for d in cluster_defects])
            avg_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal = avg_normal / norm

            # Вычисление оптимальной позиции камеры
            # Камера располагается в направлении нормали от центра кластера
            camera_pos = center + avg_normal * self.default_distance

            camera_positions.append({
                'position': camera_pos,
                'look_at': center,
                'cluster_id': cluster_id,
                'defects_count': len(cluster_defects)
            })

        return camera_positions

    def get_optimal_camera_positions(self, defects: List[Dict[str, Any]],
                                  depth_map: np.ndarray,
                                  current_position: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Основной метод для определения оптимальных позиций камеры.

        Args:
            defects: список дефектов с координатами center
            depth_map: карта глубины
            current_position: текущее положение камеры (опционально)

        Returns:
            список оптимальных позиций камеры
        """
        # 1. Преобразование в 3D
        defects_3d = self.map_2d_to_3d(defects, depth_map)

        # 2. Кластеризация
        clustered_defects, num_clusters = self.cluster_defects(defects_3d)

        # 3. Вычисление нормалей
        defects_with_normals = self.compute_surface_normals(clustered_defects, depth_map)

        # 4. Определение оптимальных позиций
        camera_positions = self.optimal_camera_positions(
            defects_with_normals, num_clusters, current_position
        )

        return camera_positions