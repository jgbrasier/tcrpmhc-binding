from typing import Callable, Dict, Generator, List, Optional
from collections import namedtuple
import os

from tqdm import tqdm
import pandas as pd

from src.utils import hard_split_df
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from graphein.protein.features.sequence.embeddings import compute_esm_embedding
from src.utils import PartialDataset, GraphDataset

class TCRpMHCDataModule(pl.LightningDataModule):
    """
    Dataloader inspired by DeepRank-GNN: A Graph Neural Network Framework to Learn Patterns in Protein-Protein Interfaces
    M. Réau, N. Renaud, L. C. Xue, A. M. J. J. Bonvin, bioRxiv 2021.12.08.471762; doi: https://doi.org/10.1101/2021.12.08.471762
    """
    def __init__(self, tsv_path: str = None, processed_dir: str = None, id_col: str ='uuid', y_col='binder', \
                batch_size: int = 32, num_workers: int = 0, device=torch.device('cpu')):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(tsv_path, sep='\t')
        # self.df = pd.concat((self.df[self.df[y_col]==0], self.df[self.df[y_col]==1].sample(frac=0.2, random_state=1)))

        self.train: GraphDataset = None
        self.val: GraphDataset = None
        self.test: GraphDataset = None

        self.selected_targets = None

    def setup(self, train_size: int = 0.8, split='random', target='epitope', low: int = 50, high: int = 800, random_seed: int = None):
        assert train_size > 0 and train_size <= 1, "train_size must be in (0, 1]"
        assert split in ['random', 'hard']
        print("Dataset train/val split method:", split)
        if split == 'hard':
            train_df, test_df, self.selected_targets = hard_split_df(self.df, target_col=target, min_ratio=train_size,
                                                        low=low, high=high, random_seed=random_seed)
            self.train = GraphDataset(train_df, self.hparams.processed_dir, self.hparams.id_col, self.hparams.y_col)
            if train_size == 1:
                self.test = None
            else:
                self.test = GraphDataset(test_df, self.hparams.processed_dir, self.hparams.id_col, self.hparams.y_col)
        elif split == 'random':
            dataset = GraphDataset(self.df, self.hparams.processed_dir, self.hparams.id_col, self.hparams.y_col)
            self.train, self.test = torch.utils.data.random_split(dataset, \
                    [int(train_size*len(dataset)), len(dataset)-int(train_size*len(dataset))], 
                    generator=torch.Generator().manual_seed(random_seed))

    # custom collate see: https://github.com/pyg-team/pytorch_geometric/issues/781
    def collate(self, data_list):
        batch_A = Batch.from_data_list([data[0] for data in data_list])
        batch_label = torch.tensor([data[1] for data in data_list]).view(-1, 1)
        # batch_name = [data[-1] for data in data_list]
        return batch_A, batch_label # , batch_name

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore

    def val_dataloader(self):
        # TODO:
        raise NotImplementedError
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.collate)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.collate)  # type: ignore
