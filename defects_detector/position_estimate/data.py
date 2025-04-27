import glob
import os

import cv2
import numpy as np
import tifffile


class DataLoader:
    def __init__(self, dataset: str, scores: str):
        self.samples = []
        self.dataset_path = dataset
        self.scores_path = scores

    def load_data(self):
        scores = np.asarray(np.load(self.scores_path)["score_map"])
        depth_paths = glob.glob(os.path.join(self.dataset_path, "xyz", "*.tiff"))

        assert len(depth_paths) == scores.shape[0], "Number of depth maps and score maps do not match"
        for i, path in enumerate(sorted(depth_paths)):
            assert os.path.exists(path), f"Depth map {path} does not exist"

            pc = tifffile.imread(path)

            score = cv2.resize(scores[i], dsize=(pc.shape[1], pc.shape[0]), interpolation=cv2.INTER_CUBIC)
            self.samples.append((pc, score))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]