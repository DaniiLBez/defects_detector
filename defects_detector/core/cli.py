import argparse
import os
from typing import Dict, Any

import numpy as np
import yaml
from torch.utils.data import DataLoader

from defects_detector.core.data import MVTec3DTrain
from defects_detector.core.shape_guided_detector import ShapeGuidedDetector
from defects_detector.utils.utils import export_test_image


def parse_args():
    parser = argparse.ArgumentParser(description="ShapeGuidedDetector для обнаружения дефектов")
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигурационному файлу")

    # Параметры, которые можно переопределить
    parser.add_argument("--image_size", type=int, help="Размер изображения")
    parser.add_argument("--feature_size", type=int, help="Размер карты признаков")
    parser.add_argument("--point_num", type=int, help="Количество точек в облаке")
    parser.add_argument("--rgb_backbone", type=str, help="Название архитектуры для RGB модели")
    parser.add_argument("--sdf_checkpoint_path", type=str, help="Путь к весам SDF модели")
    parser.add_argument("--output_dir", type=str, help="Директория для сохранения результатов")
    parser.add_argument("--image_path", type=str, help="Путь к изображениям")
    parser.add_argument("--grid_path", type=str, help="Путь к сеткам")
    parser.add_argument("--class_name", type=str, help="Имя класса для обработки", required=True)
    parser.add_argument("--visualize", action="store_true", help="Включить визуализацию результатов")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fit", action="store_true", help="Сформировать банк памяти")
    group.add_argument("--memory_bank", type=str, help="Путь к предварительно сохраненному банку памяти")

    parser.add_argument("--save_memory_bank", action="store_true", help="Сохранить банк памяти после формирования")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    # Обновляем конфигурацию параметрами из командной строки, если они указаны
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != "config":
            config[arg_name] = arg_value
    return config


def main():

    args = parse_args()

    # Загрузка конфигурации из файла
    config = load_config(args.config)

    # Обновление конфигурации с параметрами командной строки
    config = update_config_with_args(config, args)

    # Создание директории для вывода, если она указана и не существует
    output_dir = config.get("output_dir")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Инициализация детектора
    detector = ShapeGuidedDetector(config)

    dataset = MVTec3DTrain(
        config["class_name"],
        img_size=config["image_size"],
        datasets_path=config["dataset_path"],
        grid_path=config["dataset_path"]
    )
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)

    # Создание загрузчика данных для обучения
    detector.fit(data_loader)
    print(f"Обучение завершено, данные сохранены в: {config['output_dir']}")

    weight, bias = detector.align(data_loader)
    print(f"Выравнивание завершено, вес: {weight:.6f}, смещение: {bias:.6f}")

    detector.predict()

    score_map = np.asarray(detector.feature_extractor.pixel_preds).reshape(-1, detector.image_size, detector.image_size)

    if config.get("visualize", False):
        mask = np.asarray(score_map)
        export_test_image(detector.feature_extractor.image_list, mask, os.path.join(config["output_dir"], "visualization"))

    if not os.path.exists(os.path.join(config["output_dir"], "result")):
        os.makedirs(os.path.join(config["output_dir"], "result"))

    np.savez(os.path.join(config["output_dir"], "result", "score_maps.npz"), maps=score_map)

if __name__ == "__main__":
    main()