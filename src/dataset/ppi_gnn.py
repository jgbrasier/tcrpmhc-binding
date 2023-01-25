import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join

from typing import Callable, Dict, Generator, List, Optional
from collections import namedtuple

from torch.utils.data import Dataset as Dataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl

class PPIDataset(Dataset):
    """
    Dataset from:
    Jha, K., Saha, S. & Singh, H. Prediction of protein–protein interaction using graph neural networks. 
    Sci Rep 12, 8360 (2022). https://doi.org/10.1038/s41598-022-12201-9
    """
    def __init__(self, npy_file, processed_dir):
        self.npy_ar = np.load(npy_file)
        self.processed_dir = processed_dir
        self.protein_1 = self.npy_ar[:,2]
        self.protein_2 = self.npy_ar[:,5]
        self.label = self.npy_ar[:,6].astype(float)
        self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        Graph = namedtuple("graph", "prot1 prot2")
        prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
        prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
        prot_1 = torch.load(glob.glob(prot_1)[0])
        prot_2 = torch.load(glob.glob(prot_2)[0])
        label = torch.tensor(self.label[index])
        return Graph(prot_1, prot_2), label

class PPIDataModule(pl.LightningDataModule):
    def __init__(self, npy_file: str = None, processed_dir: str = None, 
                batch_size: int = 32, num_worker: int = 0):
        super(self).__init__()
        self.save_hyperparameters()

        self.dataset = PPIDataset(npy_file = npy_file, processed_dir= processed_dir)

        self.train: PPIDataset = None
        self.val: PPIDataset = None
        self.test: PPIDataset = None

    def setup(self, train_size: int = 0.8):
        assert train_size > 0 and train_size <= 1, "train_size must be in (0, 1]"
        self.train, self.test = torch.utils.data.random_split(self.dataset, [int(train_size*len(self.dataset)), len(self.dataset)-int(train_size*len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore
