import os 
import sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from typing import Optional

from src.utils import enc_list_bl_max_len, blosum50_20aa, blosum50_full, hard_split_df

import torch
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, path_to_file: str, test: Optional[bool]=None, batch_size: int = 32, n_splits: int = 5,
                peptide_len: int = 9, cdra_len: int = 30, cdrb_len: int = 30,
                target='epitope', low: int =50, high: int =800, random_seed: int =42,
                device: torch.device = torch.device('cpu'), num_workers: int =0) -> None:
        """_summary_

        :param path_to_file: relative path to .csv or .tsv file
        :type path_to_file: str
        :param test: flag to put dataloader in test-only mode, defaults to None
        :type test: Optional[bool], optional
        :param batch_size: batch size, defaults to 32
        :type batch_size: int, optional
        :param n_splits: train/val stratified k fold number of splits, defaults to 5
        :type n_splits: int, optional
        :param peptide_len: _description_, defaults to 9
        :type peptide_len: int, optional
        :param cdra_len: encoding dimension of cdr3b sequence, defaults to 30
        :type cdra_len: int, optional
        :param cdrb_len: encoding dimension of cdr3b sequence, defaults to 30
        :type cdrb_len: int, optional
        :param target: target column name in dataset, defaults to 'epitope'
        :type target: str, optional
        :param low: hard split target occurence low bound, defaults to 50
        :type low: int, optional
        :param high: hard split target occurence high bound, defaults to 800
        :type high: int, optional
        :param random_seed: random seed for train/test hardsplit and train/val stratified k fold, defaults to 42
        :type random_seed: int, optional
        :param device: device to write data tensors to, defaults to torch.device('cpu')
        :type device: torch.device, optional
        :param num_workers: number of workers for torch.DataLoader, defaults to 0
        :type num_workers: int, optional
        """
        super().__init__()
        self.save_hyperparameters()

        if not test:
            self.train_path = path_to_file
            self.test_path = None
        else:
            self.train_path = None
            self.test_path = path_to_file

        self.data = pd.DataFrame()

        self.train = None
        self.val = None
        self.test = None

        self.selected_targets = None

    def setup(self, train_size=0.85, sep='\t', encoder= enc_list_bl_max_len, encoding: dict = blosum50_full) -> None:
        
        if not self._test:
            self.data = pd.read_csv(self.train_path, sep=sep)

            train_df, test_df, self.selected_targets = hard_split_df(self.data, target_col=self.hparams.target, min_ratio=train_size,
                                                                    low=self.hparams.low, high=self.hparams.high, random_seed=self.hparams.random_seed)

            # stratified kfold split train/val
            skf = StratifiedKFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.random_seed)
            all_splits = [k for k in skf.split(train_df)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            train_df, val_df = train_df[train_indexes], train_df[val_indexes]

            # encoding will be done at dataset initalization
            self.train = TCRSeqDataset(train_df, encoder=encoder, encoding=encoding, device = self.hparams.device,
                                peptide_len = self.hparams.peptide_len, cdra_len = self.hparams.cdra_len, cdrb_len = self.hparams.cdrb_len)

            self.val = TCRSeqDataset(val_df, encoder=encoder, encoding=encoding, device = self.hparams.device,
                                peptide_len = self.hparams.peptide_len, cdra_len = self.hparams.cdra_len, cdrb_len = self.hparams.cdrb_len)                  

            self.test = TCRSeqDataset(test_df, encoder=encoder, encoding=encoding, device = self._device,
                                peptide_len = self.hparams.peptide_len, cdra_len = self.hparams.cdra_len, cdrb_len = self.hparams.cdrb_len)

        else:
            test_df = pd.read_csv(self.test_path, sep=sep)

            self.test = TCRSeqDataset(test_df, encoder=encoder, encoding=encoding, device = self.hparams.device,
                                peptide_len = self.hparams.peptide_len, cdra_len = self.hparams.cdra_len, cdrb_len = self.hparams.cdrb_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)  # type: ignore
        


if __name__=="__main__":

    # for testing purposes
    train_file = "data/sample_train.csv"

    peptide_len, cdra_len, cdrb_len = 9, 30, 30

    train_dataset = TCRSeqDataset(file = train_file, peptide_len=peptide_len, cdra_len=cdra_len, cdrb_len=cdrb_len)

    for batch in train_dataset:
        print(batch)
        break