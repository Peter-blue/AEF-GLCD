from os import listdir
from os.path import join
from pathlib import Path

import numpy as np
from PIL import Image
import torch.utils.data as data

from utils import is_image_file


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super().__init__()
        self.direction = direction
        self.a_path = join(image_dir, "Image")
        self.b_path = join(image_dir, "Image2")
        self.label_path = join(image_dir, "label")
        self.image_filenames = sorted(x for x in listdir(self.a_path) if is_image_file(x))
        self.b_stem_map = self._build_stem_map(self.b_path)
        self.label_stem_map = self._build_stem_map(self.label_path)

    @staticmethod
    def _build_stem_map(folder):
        return {
            Path(name).stem: name
            for name in listdir(folder)
            if is_image_file(name)
        }

    def __getitem__(self, index):
        a_name = self.image_filenames[index]
        stem = Path(a_name).stem
        b_name = self.b_stem_map.get(stem)
        label_name = self.label_stem_map.get(stem)
        if b_name is None or label_name is None:
            raise FileNotFoundError(f"Cannot match pair/label for {a_name} by stem {stem}")

        a = Image.open(join(self.a_path, a_name)).convert("RGB")
        b = Image.open(join(self.b_path, b_name)).convert("RGB")
        label = Image.open(join(self.label_path, label_name))

        a = np.asarray(a, dtype=np.float32) / 255.0
        b = np.asarray(b, dtype=np.float32) / 255.0
        a = np.transpose(a, (2, 0, 1))
        b = np.transpose(b, (2, 0, 1))

        label = np.asarray(label)
        if label.ndim == 3:
            label = label[..., 0]
        lbl = (label > 0).astype(np.int64)

        if self.direction == "a2b":
            return a, b, lbl
        if self.direction == "b2a":
            return b, a, lbl
        raise ValueError(f"Unsupported direction: {self.direction}")

    def __len__(self):
        return len(self.image_filenames)
