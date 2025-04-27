import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from defects_detector.position_estimate.data import DataLoader
from defects_detector.position_estimate.service import CameraPositioningService


def parse_args():
    parser = argparse.ArgumentParser(description='Сервис оптимальных позиций камеры для съемки дефектов')

    # Обязательные параметры
    parser.add_argument('--dataset', type=str, required=True,
                        help='Путь к директории с картами глубины')
    parser.add_argument('--scores', type=str, required=True,
                        help='Путь к файлу с картами аномалий (.npz)')

    # Параметры алгоритма
    parser.add_argument('--distance', type=float, default=0.5,
                        help='Расстояние от камеры до объекта по умолчанию (м)')
    parser.add_argument('--cluster-eps', type=float, default=0.01,
                        help='Максимальное расстояние между центрами дефектов в кластере')
    parser.add_argument('--min-samples', type=int, default=1,
                        help='Минимальное количество точек для формирования кластера')
    parser.add_argument('--threshold', type=float,
                        help='Пороговое значение для выделения дефектов')

    # Параметры вывода
    parser.add_argument('--output', type=str, default='camera_positions.json',
                        help='Файл для вывода результатов')
    parser.add_argument('--sample-idx', type=int,
                        help='Индекс конкретного образца для обработки')

    return parser.parse_args()


class NumpyEncoder(json.JSONEncoder):
    """Класс для кодирования numpy типов в JSON"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    args = parse_args()

    # Инициализация загрузчика данных
    data_loader = DataLoader(args.dataset, args.scores)
    data_loader.load_data()

    # Инициализация сервиса
    service = CameraPositioningService(
        default_distance=args.distance,
        cluster_eps=args.cluster_eps,
        min_samples=args.min_samples,
        threshold=args.threshold
    )

    # Обработка данных
    if args.sample_idx is not None:
        depth_map, score_map = data_loader[args.sample_idx]
        _, regions = service.extract_defects(score_map, threshold=args.threshold)
        defects = service.get_defects_from_regions(regions)
        results = service.get_optimal_camera_positions(defects, depth_map)
        print(f"Обработан образец {args.sample_idx}, найдено позиций: {len(results)}")
    else:
        results = []
        for i, sample in enumerate(tqdm(data_loader, desc="Вычисление положений камеры")):
            depth_map, score_map = sample
            _, regions = service.extract_defects(score_map, threshold=args.threshold)
            defects = service.get_defects_from_regions(regions)
            sample_results = service.get_optimal_camera_positions(defects, depth_map)
            results.append(sample_results)

    # Сохранение результатов
    with open(args.output, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

    print(f"Результаты сохранены в {args.output}")


if __name__ == "__main__":
    main()