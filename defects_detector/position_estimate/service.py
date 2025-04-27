import numpy as np
import cv2
from numpy import floating
from skimage import morphology, measure
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional

class CameraPositioningService:
    """
    Сервис для определения оптимальных позиций камеры для съемки дефектов.
    """

    def __init__(self,
                 default_distance: float = 0.5,
                 cluster_eps: float = 0.005,
                 min_samples: int = 5,
                 threshold: Optional[float] = None):
        """
        Инициализация сервиса.

        Args:
            default_distance: расстояние от камеры до объекта по умолчанию (м)
            cluster_eps: максимальное расстояние между точками в кластере
            min_samples: минимальное количество точек для формирования кластера
            threshold: пороговое значение для выделения дефектов
        """

        self.default_distance = default_distance
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.threshold = threshold

    def get_defects_from_regions(self, regions):
        """
        Преобразует регионы в формат дефектов с центрами

        Args:
            regions: список регионов из measure.regionprops

        Returns:
            список дефектов с центрами
        """
        defects = []
        for region in regions:
            y, x = region.centroid
            defects.append({
                'center': (x, y),
                'area': region.area,
                'bbox': region.bbox
            })
        return defects

    def calculate_optimal_distance(self, defects: List[Dict[str, Any]], depth_map: np.ndarray) -> float | floating[Any]:
        """
        Вычисляет оптимальное расстояние до объекта на основе карты глубины

        Args:
            defects: список дефектов
            depth_map: карта глубины

        Returns:
            оптимальное расстояние до объекта
        """
        depths = []
        for defect in defects:
            x, y = defect['center']
            x, y = int(x), int(y)

            x1 = max(0, x - 5)
            y1 = max(0, y - 5)
            x2 = min(depth_map.shape[1] - 1, x + 5)
            y2 = min(depth_map.shape[0] - 1, y + 5)

            area_depth = np.mean(depth_map[y1:y2, x1:x2])
            depths.append(area_depth)

        if not depths:
            return self.default_distance

        median_depth = np.median(depths)

        return median_depth

    def optimal_camera_positions(self, defects: List[Dict[str, Any]],
                                 num_clusters: int,
                                 depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        Определяет оптимальные положения камеры для съемки кластеров дефектов.

        Args:
            defects: список дефектов с метками кластеров и нормалями
            num_clusters: количество кластеров
            depth_map: карта глубины для вычисления оптимального расстояния

        Returns:
            список оптимальных положений камеры для каждого кластера
        """
        camera_positions = []

        for cluster_id in range(num_clusters):
            cluster_defects = [d for d in defects if d['cluster'] == cluster_id]

            if not cluster_defects:
                continue

            positions = np.array([d['position_3d'] for d in cluster_defects])
            center = np.mean(positions, axis=0)

            # Средняя нормаль кластера
            normals = np.array([d['normal'] for d in cluster_defects])
            avg_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal = avg_normal / norm

            optimal_distance = self.calculate_optimal_distance(cluster_defects, depth_map[:, :, 2])

            # Вычисление оптимальной позиции камеры
            camera_pos = center + avg_normal * optimal_distance

            def get_position(value) -> Dict[str, float]:
                return {
                    'x': value[0],
                    'y': value[1],
                    'z': value[2]
                }

            camera_positions.append({
                'camera_position': get_position(camera_pos),
                'defect_cluster_center': get_position(center),
                'cluster_id': cluster_id,
                'defects_count': len(cluster_defects),
                'distance': optimal_distance
            })

        return camera_positions

    def get_optimal_camera_positions(self, defects: List[Dict[str, Any]],
                                     depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        Основной метод для определения оптимальных позиций камеры.

        Args:
            defects: список дефектов с координатами center
            depth_map: карта глубины

        Returns:
            список оптимальных позиций камеры
        """
        # 1. Преобразование в 3D
        defects_3d = self.map_2d_to_3d(defects, depth_map)

        # 2. Кластеризация
        clustered_defects, num_clusters = self.cluster_defects(defects_3d)

        # 3. Вычисление нормалей
        defects_with_normals = self.compute_surface_normals(clustered_defects, depth_map)

        # 4. Определение оптимальных позиций с учетом карты глубины
        camera_positions = self.optimal_camera_positions(
            defects_with_normals, num_clusters, depth_map
        )

        return camera_positions

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
        # Автоматическое определение порога, если не задан
        if threshold is None:
            threshold = np.reshape(score_map, -1).mean() + 3 * np.reshape(score_map, -1).std()

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

        for defect in defects:
            x, y = defect['center']
            x, y = int(x), int(y)

            defect['position_3d'] = depth_map[y, x]

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
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=1).fit(positions)
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
        depth_map_2d = depth_map[:, :, 2]
        sobelx = cv2.Sobel(depth_map_2d, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_map_2d, cv2.CV_64F, 0, 1, ksize=3)

        for defect in defects:
            x, y = defect['center']
            x, y = int(x), int(y)

            dx = sobelx[y, x]
            dy = sobely[y, x]

            # Нормаль поверхности (-dx, -dy, -1), нормализованная
            normal = np.array([-dx, -dy, -1.0])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            defect['normal'] = normal

        return defects
