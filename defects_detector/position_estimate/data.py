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
        raw_scores = np.load(self.scores_path)
        scores = np.asarray(raw_scores["maps"])

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

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        result = self[self.current_idx]
        self.current_idx += 1
        return result