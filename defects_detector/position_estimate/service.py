import numpy as np
import cv2
from numpy import floating
from skimage import morphology, measure
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional

class CameraPositioningService:
    def __init__(self,
                 default_distance: float = 0.5,
                 cluster_eps: float = 0.005,
                 min_samples: int = 5,
                 threshold: Optional[float] = None):
        self.default_distance = default_distance
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.threshold = threshold

    def get_defects_from_regions(self, regions):
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
        return np.median(depths)

    def optimal_camera_positions(self, defects: List[Dict[str, Any]],
                                 num_clusters: int,
                                 depth_map: np.ndarray) -> List[Dict[str, Any]]:
        camera_positions = []
        for cluster_id in range(num_clusters):
            cluster_defects = [d for d in defects if d['cluster'] == cluster_id]
            if not cluster_defects:
                continue

            positions = np.array([d['position_3d'] for d in cluster_defects])
            center = np.mean(positions, axis=0)

            normals = np.array([d['normal'] for d in cluster_defects])
            avg_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal = avg_normal / norm

            optimal_distance = self.calculate_optimal_distance(cluster_defects, depth_map[:, :, 2])
            camera_pos = center + avg_normal * optimal_distance

            def get_position(value) -> Dict[str, float]:
                return {'x': value[0], 'y': value[1], 'z': value[2]}

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
        defects_3d = self.map_2d_to_3d(defects, depth_map)
        clustered_defects, num_clusters = self.cluster_defects(defects_3d)
        defects_with_normals = self.compute_surface_normals(clustered_defects, depth_map)
        return self.optimal_camera_positions(defects_with_normals, num_clusters, depth_map)

    @staticmethod
    def extract_defects(score_map, threshold=None, min_area=5):
        if threshold is None:
            threshold = np.reshape(score_map, -1).mean() + 3 * np.reshape(score_map, -1).std()
        binary_map = (score_map > threshold).astype(np.uint8)
        kernel = morphology.disk(2)
        binary_map = morphology.opening(binary_map, kernel)
        labeled_map = measure.label(binary_map)
        regions = measure.regionprops(labeled_map)
        return binary_map, [r for r in regions if r.area >= min_area]

    def map_2d_to_3d(self, defects: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]:
        for defect in defects:
            x, y = map(int, defect['center'])
            defect['position_3d'] = depth_map[y, x]
        return defects

    def cluster_defects(self, defects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        if not defects:
            return defects, 0
        positions = np.array([d['position_3d'] for d in defects])
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=1).fit(positions)
        for i, d in enumerate(defects):
            d['cluster'] = int(clustering.labels_[i])
        return defects, int(np.max(clustering.labels_) + 1 if len(clustering.labels_) > 0 else 0)

    def compute_surface_normals(self, defects: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """Вычисление нормалей только по Z-каналу с использованием оператора Собеля"""
        depth_map_z = depth_map[:, :, 2]
        sobelx = cv2.Sobel(depth_map_z, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_map_z, cv2.CV_64F, 0, 1, ksize=3)
        for defect in defects:
            x, y = map(int, defect['center'])
            dx = sobelx[y, x]
            dy = sobely[y, x]
            normal = np.array([-dx, -dy, -1.0])
            norm = np.linalg.norm(normal)
            defect['normal'] = normal / norm if norm > 0 else normal
        return defects
