import os
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join

from typing import Callable, Dict, Generator, List, Optional
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch


import pytorch_lightning as pl

class PartialDataset(Dataset):
    """
    Dataset for loading list of .pt graph files
    """
    def __init__(self, paths: List[str], 
                _device: torch.device = torch.device('cpu')) -> None:
        self.paths = paths
        self._device = _device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        data = torch.load(self.paths[index], map_location=self._device)
        # data.x = data.x.type(torch.float)
        # data.edge_index = data.edge_index.type(torch.int64)
        # data.edge_index = data.edge_index.type(torch.int64)
        return data

class PPIDataset(Dataset):
    """
    Dataset from:
    Jha, K., Saha, S. & Singh, H. Prediction of protein–protein interaction using graph neural networks. 
    Sci Rep 12, 8360 (2022). https://doi.org/10.1038/s41598-022-12201-9
    """
    def __init__(self, npy_file, processed_dir, device=torch.device('cpu')):
        super().__init__()
        self.npy_ar = np.load(npy_file)
        self.processed_dir = processed_dir
        self.protein_1_paths = [os.path.join(self.processed_dir, str(prot1)+'.pt') for prot1 in self.npy_ar[:,2]]
        self.protein_2_paths = [os.path.join(self.processed_dir, str(prot1)+'.pt') for prot1 in self.npy_ar[:,5]]

        self.protein_1_dataset = PartialDataset(self.protein_1_paths, _device=device)
        self.protein_2_dataset = PartialDataset(self.protein_2_paths, _device=device)


        self.label = self.npy_ar[:,6].astype(float)
        self.n_samples = self.npy_ar.shape[0]

        self._device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        prot_1 = self.protein_1_dataset[index]
        prot_2 = self.protein_2_dataset[index]
        label = torch.tensor(self.label[index])
        return prot_1, prot_2, label

class PPIDataModule(pl.LightningDataModule):
    """
    Dataset from:
    Jha, K., Saha, S. & Singh, H. Prediction of protein–protein interaction using graph neural networks. 
    Sci Rep 12, 8360 (2022). https://doi.org/10.1038/s41598-022-12201-9
    """
    def __init__(self, npy_file: str = None, processed_dir: str = None, 
                batch_size: int = 32, num_workers: int = 0, device=torch.device('cpu')):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = PPIDataset(npy_file=npy_file, processed_dir=processed_dir, device=device)

        self.train: PPIDataset = None
        self.val: PPIDataset = None
        self.test: PPIDataset = None

    def setup(self, train_size: int = 0.8, random_seed: int = 42):
        assert train_size > 0 and train_size <= 1, "train_size must be in (0, 1]"
        self.train, self.test = torch.utils.data.random_split(self.dataset, \
            [int(train_size*len(self.dataset)), len(self.dataset)-int(train_size*len(self.dataset))], 
            generator=torch.Generator().manual_seed(random_seed))
    # custom collate for paired dataset 
    # see: https://github.com/pyg-team/pytorch_geometric/issues/781
    def collate(self, data_list):
        batch_A = Batch.from_data_list([data[0] for data in data_list])
        batch_B = Batch.from_data_list([data[1] for data in data_list])
        batch_label = torch.tensor([data[2] for data in data_list]).view(-1, 1)
        return batch_A, batch_B, batch_label

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore

    def val_dataloader(self):
        # TODO:
        raise NotImplementedError
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore
