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
from torch_geometric.data import DataLoader as DataLoader


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
