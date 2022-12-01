import os 
import sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from src.utils import enc_list_bl_max_len, blosum50_20aa

import torch
from torch.utils.data import Dataset

class TCRSeqDataset(Dataset):
    def __init__(self, file: str, test: bool =False, encoder= enc_list_bl_max_len, encoding: dict = blosum50_20aa ) -> None:
        super().__init__()
        self.file = file
        data = pd.read_csv(file)
        self._len = len(data)
        self.encoding = encoding
        self.test = test

        self.peptide = encoder(data.peptide, encoding, 9)
        self.tcra = encoder(data.CDR3a, encoding, 30)
        self.tcrb = encoder(data.CDR3b, encoding, 30)
        self.y = np.array(data.binder) if not test else None

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
            return torch.from_numpy(peptide), torch.from_numpy(tcra), torch.from_numpy(tcrb)
        else:
            y = self.y[index]
            return torch.from_numpy(peptide), torch.from_numpy(tcra), torch.from_numpy(tcrb), torch.from_numpy(y)