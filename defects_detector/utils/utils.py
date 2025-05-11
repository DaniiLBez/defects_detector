import os
import math
import numpy as np
import os.path as osp
import random
import torch
import open3d as o3d
from torchvision import transforms
import torch.nn.functional as F
from PIL import ImageFilter
# configuration
from six.moves import cPickle
from scipy.spatial.ckdtree import cKDTree

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map

def get_relative_rgb_f_indices(target_pc_idices, img_size=224, f_size=28):
    scale = int(img_size / f_size)
    row = torch.div(target_pc_idices,img_size,rounding_mode='floor')
    col = target_pc_idices % img_size
    rgb_f_row = torch.div(row,scale,rounding_mode='floor')
    rgb_f_col = torch.div(col,scale,rounding_mode='floor')
    rgb_f_indices = rgb_f_row * f_size + rgb_f_col
    rgb_f_indices = torch.unique(rgb_f_indices)

    # More Background feature #
    B = 2
    rgb_f_indices = torch.cat([rgb_f_indices+B,rgb_f_indices-B,rgb_f_indices+28*B,rgb_f_indices-28*B],dim=0)
    rgb_f_indices[rgb_f_indices<0] = torch.max(rgb_f_indices)
    rgb_f_indices[rgb_f_indices>783] = torch.min(rgb_f_indices)
    rgb_f_indices = torch.unique(rgb_f_indices)

    return rgb_f_indices

def data_process(npz_dir_path):
    points_gt_all = []
    points_idx_all = []
    points_tran_all = []
    files_path = npz_dir_path
    if (os.path.exists(files_path)):
        load_data = np.load(files_path)
        points_gt_set = np.asarray(load_data['points_gt'])
        points_idx_set = np.asarray(load_data['points_idx'])

        # print('data patches number:',points_gt_set.shape[0])
        for patch in range(points_gt_set.shape[0]):
            points_gt = points_gt_set[patch]
            points_idx = points_idx_set[patch]
            points_gt, points_tran = normal_points(points_gt, True)

            points_gt_all.append(points_gt)
            points_idx_all.append(points_idx)
            points_tran_all.append(points_tran)

    return points_gt_all, points_idx_all, points_tran_all


def normal_points(ps_gt, translation=False):
    """
    Нормализует облако точек путем масштабирования и опционального центрирования.

    Args:
        ps_gt: Облако точек формы (N, 3)
        translation: Если True, центрирует точки относительно их среднего значения

    Returns:
        tuple: (нормализованные точки, параметры трансформации (смещение, масштаб))
    """
    # Вычисление диапазонов по каждой оси
    x_range = np.max(ps_gt[:, 0]) - np.min(ps_gt[:, 0])
    y_range = np.max(ps_gt[:, 1]) - np.min(ps_gt[:, 1])
    z_range = np.max(ps_gt[:, 2]) - np.min(ps_gt[:, 2])

    # Определение максимального диапазона
    max_range = max(x_range, y_range, z_range)

    # Расчет коэффициента масштабирования
    scale_factor = 10 / (10 * max_range)

    # Масштабирование точек
    ps_gt = ps_gt * scale_factor

    # Инициализация вектора смещения
    t = np.zeros(3)

    # Опциональное центрирование
    if translation:
        t = np.mean(ps_gt, axis=0)
        ps_gt = ps_gt - t

    return ps_gt, (t, scale_factor)


def export_test_image(test_img, scores, path):
    kernel = morphology.disk(2)
    scores_norm = 1.0 / scores.max()

    for i in tqdm(range(0, len(test_img), 1), desc="export heat map image"):
        img = test_img[i]
        img = denormalization(img, IMAGENET_MEAN, IMAGENET_STD)

        # scores
        threshold = scores[i].mean() + 3 * scores[i].std()
        score_mask = np.zeros_like(scores[i])
        score_mask[scores[i] > threshold] = 1.0
        score_mask = morphology.opening(score_mask, kernel)
        score_mask = (255.0 * score_mask).astype(np.uint8)
        score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
        score_map = (255.0 * scores[i] * scores_norm).astype(np.uint8)
        #
        fig_img, ax_img = plt.subplots(2, 1, figsize=(2 * cm, 4 * cm))
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['bottom'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
        #
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        ax_img[0].imshow(img, cmap='gray', interpolation='none')
        ax_img[0].imshow(score_map, cmap='jet', norm=norm, alpha=0.5, interpolation='none')
        ax_img[1].imshow(score_img)
        image_file = os.path.join(path, f'{i:08d}.png')
        fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.0)
        plt.close()