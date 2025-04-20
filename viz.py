import numpy as np
from scipy.spatial.distance import cdist


def compare_patch_similarity(patches1, patches2):
    """Сравнивает соответствующие патчи используя расстояние Хаусдорфа."""
    patches1_data = patches1["points_gt"] if "points_gt" in patches1 else patches1["patches"]
    patches2_data = patches2["points_gt"] if "points_gt" in patches2 else patches2["patches"]

    min_len = min(len(patches1_data), len(patches2_data))

    hausdorff_dists = []
    chamfer_dists = []

    for idx in range(min_len):
        # Хаусдорфово расстояние
        d1 = cdist(patches1_data[idx], patches2_data[idx])
        hausdorff = max(np.min(d1, axis=0).max(), np.min(d1, axis=1).max())
        hausdorff_dists.append(hausdorff)

        # Приближённое Chamfer расстояние
        chamfer = np.mean(np.min(d1, axis=0)) + np.mean(np.min(d1, axis=1))
        chamfer_dists.append(chamfer)

    print(f"Среднее Хаусдорфово расстояние между патчами: {np.mean(hausdorff_dists)}")
    print(f"Среднее Chamfer расстояние между патчами: {np.mean(chamfer_dists)}")

# import matplotlib.pyplot as plt
#
#
# def visualize_patches(patches1, patches2, sample_indices=[0, 1, 2]):
#     """Визуализирует и сравнивает патчи из двух разных источников."""
#     for idx in sample_indices:
#         fig = plt.figure(figsize=(12, 6))
#
#         # Визуализация патча из первой реализации
#         ax1 = fig.add_subplot(121, projection='3d')
#         points1 = patches1["points_gt"][idx] if "points_gt" in patches1 else patches1["patches"][idx]
#         ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=1)
#         ax1.set_title('Патч из новой реализации')
#
#         # Визуализация патча из второй реализации
#         ax2 = fig.add_subplot(122, projection='3d')
#         points2 = patches2["points_gt"][idx] if "points_gt" in patches2 else patches2["patches"][idx]
#         ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=1)
#         ax2.set_title('Патч из исходной реализации')
#
#         plt.tight_layout()
#         plt.show()


new_patches = np.load('/mnt/d/Downloads/MVTec3D-AD/cookie/test/combined/npz/000.npz')
original_patches = np.load('data/cookie/test/combined/npz/000.npz')
# Сравнение патчей по метрикам
compare_patch_similarity(new_patches, original_patches)
# visualize_patches(new_patches, original_patches)