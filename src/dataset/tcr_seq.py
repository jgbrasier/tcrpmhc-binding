import os 
import sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from typing import Optional

from src.utils import enc_list_bl_max_len, blosum50_20aa, blosum50_full, hard_split_df

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl

class TCRSeqDataset(Dataset):
    """
    DataLoader for:
    Montemurro, A., Schuster, V., Povlsen, H.R. et al. 
    NetTCR-2.0 enables accurate prediction of TCR-peptide binding 
    by using paired TCRα and β sequence data. 
    Commun Biol 4, 1060 (2021). https://doi.org/10.1038/s42003-021-02610-3
    """
    def __init__(self, data: pd.DataFrame, test: Optional[bool]=False, 
                encoder= enc_list_bl_max_len, encoding: dict = blosum50_20aa, 
                peptide_len: int = 9, cdra_len: int = 30,cdrb_len: int = 30, 
                device: torch.device = torch.device('cpu')) -> None:

        super().__init__()
        self.data = data
        self.encoding = encoding
        self._len = len(data.index)
        self._test = test

        self.peptide = encoder(data['epitope'], encoding, peptide_len)
        self.cdr3a = encoder(data['cdr3a'], encoding, cdra_len)
        self.cdr3b = encoder(data['cdr3b'], encoding, cdrb_len)
        self.y = np.array(data['binder']) if not test else None

        self._device = device

    @property
    def istest(self):
        return self.test

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        peptide = self.peptide[index]
        tcra = self.cdr3a[index]
        tcrb = self.cdr3b[index]

        if self.istest:
            return (torch.tensor(peptide, device=self._device, dtype=torch.float), torch.tensor(tcra, device=self._device, dtype=torch.float), torch.tensor(tcrb, device=self._device, dtype=torch.float))
        else:
            y = self.y[index]
            return (torch.tensor(peptide, device=self._device, dtype=torch.float), torch.tensor(tcra, device=self._device, dtype=torch.float), torch.tensor(tcrb, device=self._device, dtype=torch.float)), torch.tensor(y, device=self._device, dtype=torch.float)

class TCRSeqDataModule(pl.LightningDataModule):
    def __init__(self, path_to_file, test: Optional[bool]=None, 
                device: torch.device = torch.device('cpu'), num_workers=0) -> None:
        super().__init__()

        if not test:
            self.train_path = path_to_file
            self.test_path = None
        else:
            self.train_path = None
            self.test_path = path_to_file
        self._test = test

        self.data = pd.DataFrame()

        self.train = None
        self.val = None
        self.test = None

        self._num_workers = num_workers
        self._device = device

    def setup(self, train_size=0.8, sep='\t', target='epitope', low=50, high=800, random_seed=42) -> None:
        
        if not self._test:
            self.data = pd.read_csv(self.train_path, sep=sep)

            self.train, self.test, selected_targets = hard_split_df(self.data, target_col=target, min_ratio=train_size,
                                                                    low=low, high=high, random_seed=random_seed)

        else:
            self.test = pd.read_csv(self.test_path, sep=sep)
        


if __name__=="__main__":

    # for testing purposes
    train_file = "data/sample_train.csv"

    peptide_len, cdra_len, cdrb_len = 9, 30, 30

    train_dataset = TCRSeqDataset(file = train_file, peptide_len=peptide_len, cdra_len=cdra_len, cdrb_len=cdrb_len)

    for batch in train_dataset:
        print(batch)
        break