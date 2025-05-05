import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from typing import Dict, List
from skimage import measure, morphology


def interactive_camera_validation(dataset_path, scores_path, results_path, sample_idx=0,
                                  html_output=None, stride=3, threshold=None):
    """
    Интерактивная визуализация для проверки позиций камеры с использованием Plotly
    с подсветкой дефектных областей

    Args:
        dataset_path: путь к директории с картами глубины
        scores_path: путь к файлу с картами аномалий (.npz)
        results_path: путь к файлу с результатами определения позиций камеры
        sample_idx: индекс образца для визуализации
        html_output: путь для сохранения HTML-файла с визуализацией
        stride: шаг прореживания точек для облегчения отображения
        threshold: пороговое значение для выделения дефектов (если None, вычисляется автоматически)
    """
    # Загружаем результаты
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Загружаем данные для проверки
    from defects_detector.position_estimate.data import DataLoader
    loader = DataLoader(dataset_path, scores_path)
    loader.load_data()

    # Получаем карты для указанного образца
    depth_map, score_map = loader[sample_idx]

    # Автоматическое определение порога, если не задан
    if threshold is None:
        threshold = np.reshape(score_map, -1).mean() + 3 * np.reshape(score_map, -1).std()

    # Бинаризация карты аномалий
    binary_map = (score_map > threshold).astype(np.uint8)

    # Морфологическая обработка для удаления шума
    kernel = morphology.disk(2)
    binary_map = morphology.opening(binary_map, kernel)

    # Создаем фигуру Plotly
    fig = go.Figure()

    # Контейнеры для нормальных и дефектных точек
    normal_points = []
    defect_points = []

    # Прореживаем точки для наглядности и разделяем на нормальные и дефектные
    for y in range(0, depth_map.shape[0], stride):
        for x in range(0, depth_map.shape[1], stride):
            point = depth_map[y, x]
            # Проверяем, является ли точка дефектом
            if binary_map[y, x] > 0:
                defect_points.append(point)
            else:
                normal_points.append(point)

    normal_points = np.array(normal_points)
    defect_points = np.array(defect_points)

    # Добавляем нормальные точки на график
    if len(normal_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=normal_points[:, 0], y=normal_points[:, 1], z=normal_points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='orange',  # Светло-серый для нормальных точек
                opacity=0.7
            ),
            name='Нормальные точки'
        ))

    # Добавляем дефектные точки на график
    if len(defect_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=defect_points[:, 0], y=defect_points[:, 1], z=defect_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,  # Чуть больше размер для выделения
                color='red',  # Красный цвет для дефектов
                opacity=1.0
            ),
            name='Дефектные точки'
        ))

    # Обрабатываем позиции камеры
    sample_results = results[sample_idx] if isinstance(results[0], list) else results

    # Контейнеры для информации
    info_text = []
    camera_x, camera_y, camera_z = [], [], []
    defect_x, defect_y, defect_z = [], [], []
    lines_x, lines_y, lines_z = [], [], []

    for pos in sample_results:
        # Позиция камеры
        if isinstance(pos['camera_position'], dict):
            cam_pos = np.array([pos['camera_position']['x'],
                                pos['camera_position']['y'],
                                pos['camera_position']['z']])
        else:
            cam_pos = np.array(pos['camera_position'])

        # Центр дефекта
        if isinstance(pos['defect_cluster_center'], dict):
            defect_center = np.array([pos['defect_cluster_center']['x'],
                                      pos['defect_cluster_center']['y'],
                                      pos['defect_cluster_center']['z']])
        else:
            defect_center = np.array(pos['defect_cluster_center'])

        # Сохраняем координаты для графиков
        camera_x.append(cam_pos[0])
        camera_y.append(cam_pos[1])
        camera_z.append(cam_pos[2])

        defect_x.append(defect_center[0])
        defect_y.append(defect_center[1])
        defect_z.append(defect_center[2])

        # Добавляем линии (соединяем каждую пару точек отдельной линией)
        lines_x.extend([cam_pos[0], defect_center[0], None])
        lines_y.extend([cam_pos[1], defect_center[1], None])
        lines_z.extend([cam_pos[2], defect_center[2], None])

        # Рассчитываем фактическое расстояние
        actual_distance = np.linalg.norm(cam_pos - defect_center)
        expected_distance = pos.get('distance', actual_distance)

        # Формируем текст для всплывающих подсказок
        info_text.append(
            f"Кластер: {pos.get('cluster_id', 'N/A')}<br>" +
            f"Дистанция: {actual_distance:.4f}<br>" +
            f"Ожидаемая: {expected_distance:.4f}<br>" +
            f"Дефектов: {pos.get('defects_count', 'N/A')}"
        )

        # Выводим информацию в консоль
        print(f"Кластер {pos.get('cluster_id', 'N/A')}:")
        print(f"  Позиция камеры: [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
        print(f"  Центр дефекта: [{defect_center[0]:.4f}, {defect_center[1]:.4f}, {defect_center[2]:.4f}]")
        print(f"  Фактическое расстояние: {actual_distance:.4f}")
        print(f"  Указанное расстояние: {expected_distance:.4f}")
        print(f"  Разница: {abs(actual_distance - expected_distance):.6f}")
        print()

    # Добавляем позиции камер
    fig.add_trace(go.Scatter3d(
        x=camera_x, y=camera_y, z=camera_z,
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='diamond'
        ),
        name='Камеры',
        text=info_text,
        hoverinfo='text'
    ))

    # Добавляем центры дефектов
    fig.add_trace(go.Scatter3d(
        x=defect_x, y=defect_y, z=defect_z,
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            symbol='circle'
        ),
        name='Центры кластеров',
        text=info_text,
        hoverinfo='text'
    ))

    # Добавляем линии между камерами и дефектами
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(
            color='green',
            width=4
        ),
        name='Направление',
        hoverinfo='none'
    ))

    # Настраиваем внешний вид
    fig.update_layout(
        title=f'Интерактивная проверка позиций камеры для образца {sample_idx}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # Сохраняет соотношение сторон
            zaxis=dict(
                autorange="reversed"  # Инвертируем ось Z, чтобы она была направлена вниз
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(r=20, l=10, b=10, t=50),
        height=800,
    )

    # Добавляем кнопки для управления камерой с учётом инвертированной оси Z
    camera_buttons = [
        dict(
            method="relayout",
            args=[{"scene.camera": {"eye": {"x": 0, "y": 0, "z": -2},
                                    "center": {"x": 0, "y": 0, "z": 0}}}],
            label="Вид спереди"
        ),
        dict(
            method="relayout",
            args=[{"scene.camera": {"eye": {"x": 2, "y": 0, "z": 0},
                                    "center": {"x": 0, "y": 0, "z": 0}}}],
            label="Вид сбоку"
        ),
        dict(
            method="relayout",
            args=[{"scene.camera": {"eye": {"x": 0, "y": 0, "z": 2},
                                    "center": {"x": 0, "y": 0, "z": 0}}}],
            label="Вид сзади"
        ),
        dict(
            method="relayout",
            args=[{"scene.camera": {"eye": {"x": 1.5, "y": 1.5, "z": -1.5},
                                    "center": {"x": 0, "y": 0, "z": 0}}}],
            label="Изометрия"
        )
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=camera_buttons,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                y=-0.2,
                xanchor='center',
                yanchor='bottom',
            )
        ]
    )

    # Добавляем надписи над точками
    annotations = []
    for i, (x, y, z) in enumerate(zip(camera_x, camera_y, camera_z)):
        annotations.append(
            dict(
                showarrow=False,
                x=x,
                y=y,
                z=z,
                text=f"К{i}",
                xanchor="left",
                xshift=10,
                opacity=0.7
            )
        )

    for i, (x, y, z) in enumerate(zip(defect_x, defect_y, defect_z)):
        annotations.append(
            dict(
                showarrow=False,
                x=x,
                y=y,
                z=z,
                text=f"Д{i}",
                xanchor="left",
                xshift=10,
                opacity=0.7
            )
        )

    # scene_annotations удалены

    # Сохраняем в HTML-файл, если указан путь
    if html_output:
        pio.write_html(fig, file=html_output, auto_open=True)

    return fig


# Пример использования
fig = interactive_camera_validation(
    dataset_path='/mnt/d/Downloads/MVTec3D-AD/cookie/test/combined/',
    scores_path='./defects_detector/output/result/score_maps.npz',
    results_path='./defects_detector/results.json',
    sample_idx=24,
    html_output='camera_validation.html',  # Сохраняем результат в HTML
    stride=3  # Увеличиваем шаг для более быстрой визуализации
)

fig.show()