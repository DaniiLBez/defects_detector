import argparse
import os
import subprocess
from typing import Dict, Any

import cv2
import yaml
from torch.utils.data import DataLoader

from defects_detector.core.data import MVTec3DTrain
from defects_detector.core.shape_guided_detector import ShapeGuidedDetector


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
    group.add_argument("--fit", action="store_true", help="Обучить модель")
    group.add_argument("--memory_bank", type=str, help="Путь к предварительно сохраненному банку памяти")

    parser.add_argument("--save_memory_bank", action="store_true", help="Сохранить банк памяти после обучения")

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

    result = detector.predict()

    if config.get("visualize", False):
        # Создаем директорию для визуализаций, если её нет
        viz_dir = os.path.join(config['output_dir'], "visualizations") if config['output_dir'] else "visualizations"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)


        for k, v in result.items():
            # Формируем имя файла для сохранения
            if isinstance(k, str):
                image_name = os.path.basename(k).split('.')[0]
            else:
                image_name = f"sample_{len(result)}"

            # Сохраняем изображение
            output_path = os.path.join(viz_dir, f"{image_name}_overlay.png")
            cv2.imwrite(output_path, v["overlay"])

if __name__ == "__main__":
    main()