import glob
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from defects_detector.core.base import DataLoaderBase
from defects_detector.utils.utils import data_process


class MVTec3D(Dataset):
    def __init__(self, split, class_name, img_size, datasets_path, grid_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(datasets_path, self.cls, split)
        self.npz_path = os.path.join(grid_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])


class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size, datasets_path, grid_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, datasets_path=datasets_path,
                         grid_path=grid_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        npz_paths = glob.glob(os.path.join(self.npz_path, 'good', 'npz') + "/*.npz")
        rgb_paths.sort()
        tiff_paths.sort()
        npz_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths, npz_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        npz_path = img_path[2]
        # load image data
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        # load npz data
        points_gt_all, points_idx_all, points_tran_all = data_process(npz_path)
        return (img, points_gt_all, points_idx_all), label


class MVTecDataLoader(DataLoaderBase):
    """Загрузчик данных для MVTec3D"""
    def __init__(self, img_size, datasets_path, grid_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.size = img_size
        self.img_path = datasets_path
        self.npz_path = grid_path
        self.samples = self.load_data()
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

    def load_data(self, *args: Any):
        """
        Загружает данные из указанного пути и возвращает их в формате для предсказания.
        """
        return [(rgb, tiff, npz) for rgb, tiff, npz in zip(
            sorted(glob.glob(os.path.join(self.img_path, 'rgb', '*.png'))),
            sorted(glob.glob(os.path.join(self.img_path, 'xyz', '*.tiff'))),
            sorted(glob.glob(os.path.join(self.npz_path, 'npz', '*.npz')))
        )]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Загружает один элемент данных по индексу.
        """
        rgb_path, tiff_path, npz_path = self.samples[idx]
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        points_gt_all, points_idx_all, points_tran_all = data_process(npz_path)
        return img, points_gt_all, points_idx_all