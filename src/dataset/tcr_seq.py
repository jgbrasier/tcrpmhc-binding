import os 
import sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from src.utils import enc_list_bl_max_len, blosum50_20aa

import torch
from torch.utils.data import Dataset

class TCRSeqDataset(Dataset):
    def __init__(self, file: str, test: bool =False, 
                encoder= enc_list_bl_max_len, encoding: dict = blosum50_20aa, 
                peptide_len: int = 9, cdra_len: int = 30,cdrb_len: int = 30, 
                device: torch.device = torch.device('cpu')) -> None:

        super().__init__()
        self.file = file
        data = pd.read_csv(file)
        self._len = len(data)
        self.encoding = encoding
        self.test = test

        self.peptide = encoder(data.peptide, encoding, peptide_len)
        self.tcra = encoder(data.CDR3a, encoding, cdra_len)
        self.tcrb = encoder(data.CDR3b, encoding, cdrb_len)
        self.y = np.array(data.binder) if not test else None

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
        tcra = self.tcra[index]
        tcrb = self.tcrb[index]

        if self.istest:
            return (torch.tensor(peptide, device=self._device, dtype=torch.float), torch.tensor(tcra, device=self._device, dtype=torch.float), torch.tensor(tcrb, device=self._device, dtype=torch.float))
        else:
            y = self.y[index]
            return (torch.tensor(peptide, device=self._device, dtype=torch.float), torch.tensor(tcra, device=self._device, dtype=torch.float), torch.tensor(tcrb, device=self._device, dtype=torch.float)), torch.tensor(y, device=self._device, dtype=torch.float)

if __name__=="__main__":

    # for testing purposes
    train_file = "data/sample_train.csv"

    peptide_len, cdra_len, cdrb_len = 9, 30, 30

    train_dataset = TCRSeqDataset(file = train_file, peptide_len=peptide_len, cdra_len=cdra_len, cdrb_len=cdrb_len)

    for batch in train_dataset:
        print(batch)
        break