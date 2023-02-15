import os
import pandas as pd
import numpy as np

from src.utils import hard_split_df

from typing import Callable

import torch
import torch.nn as nn
from torch.nn.functional import threshold, normalize, pad
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import pytorch_lightning as pl

class TransformDistanceMatrix(object):
    """Transform distance matrix:
     - numpy image to tensor
     - pad image (right and bottom with 0 value)
     - normalize tensor (L1 norm)
    """
    def __init__(self, output_size, distance_treshold):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        assert isinstance(distance_treshold, (int, float))
        self.distance_treshold = distance_treshold
        
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        assert img.ndim >= 2
        img[img>self.distance_treshold] = 0
        img = to_tensor(img)
        if isinstance(self.output_size, int):
            img = pad(img, [0, self.output_size - img.shape[-1], 0, self.output_size - img.shape[-2]], "constant", 0)
        else:
            img = pad(img, [0, self.output_size[1] - img.shape[-1], 0, self.output_size[0] - img.shape[-2]], "constant", 0)
        img = normalize(img, p=1).float()
        return img

class TransformSequence(object):
    def __init__(self, embedding_function: Callable, padding: int) -> None:
        assert isinstance(embedding_function, callable)
        self.emb_fn = embedding_function



class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dist_mat_dir: str, seq_dir: str, id_col: str ='uuid', y_col: str ='binder'):
        self.dist_mat_dir = dist_mat_dir
        self.seq_dir = seq_dir

        npy_ar = np.array(df[[id_col, y_col]].values)
        self.names = npy_ar[:, 0]
        self.labels = npy_ar[:, 1]

        self.img_transform = TransformDistanceMatrix(output_size=427, distance_treshold=10.)

        # embed sequences


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdb_id = str(self.names[idx])
        img_path = os.path.join(self.dist_mat_dir, pdb_id+'.npy')
        img = np.load(img_path)

        # transform image
        img = self.img_transform(img)

        label = torch.tensor(self.labels[idx])
        return img, label