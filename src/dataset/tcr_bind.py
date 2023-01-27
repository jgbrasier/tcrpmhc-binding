from typing import Callable, Dict, Generator, List, Optional
from collections import namedtuple
import os

from tqdm import tqdm
import pandas as pd

from src.utils import hard_split_df

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from graphein.protein.features.sequence.embeddings import compute_esm_embedding

class TCRPartialDataset(Dataset):
    """
    Dataset for loading tcr data stored in list of folders
    """
    def __init__(self, paths: List[str], _type: str, 
                _device: torch.device = torch.device('cpu')) -> None:
        assert _type in ['cdr3a_seq_emb', 'cdr3b_seq_emb', 'epitope_seq_emb','tcr_graph', 'pmhc_graph']
        self._type = _type
        self.paths = paths
        self._device = _device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_name = +str(self._type)+'.pt'
        file_paths = [os.path.join(path, file_name) for path in self.paths[index]]
        torch.load(file_paths, map_location=self._device)

class TCRBindDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, include_seq_data: bool = False): 

        # load data
        self.df = df
        self.data_dir = data_dir
        self.df['path'] = [os.path.join(self.data_dir, str(id)) for id in self.df['id']]


        self.tcr_graph_dataset = TCRPartialDataset(self.df['path'], _type='tcr_graph')
        self.pmhc_graph_dataset = TCRPartialDataset(self.df['path'], _type='pmhc_graph')

        assert len(self.tcr_graph_dataset) == len(self.pmhc_graph_dataset)

        self._include_seq_data = include_seq_data

        if include_seq_data:
            self.cdr3a_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='cdr3a_seq_emb')
            self.cdr3b_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='cdr3b_seq_emb')
            self.epitope_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='epitope_seq_emb')
            assert len(self.cdr3a_seq_emb_dataset) == len(self.cdr3b_seq_emb_dataset) \
            and len(self.cdr3b_seq_emb_dataset) == len(self.epitope_seq_emb_dataset)
            assert len(self.cdr3a_seq_emb_dataset) == len(self.tcr_graph_dataset)
        else:
            self.cdr3a_seq_emb_dataset = None
            self.cdr3b_seq_emb_dataset = None
            self.epitope_seq_emb_dataset = None

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        label = torch.tensor(self.df.iloc[index]['binding'])
        if self._include_seq_data:
            return self.tcr_graph_dataset[index], self.pmhc_graph_dataset[index], \
                self.cdr3a_seq_emb_dataset[index], self.cdr3b_seq_emb_dataset[index], self.epitope_seq_emb_dataset[index], \
                    label
        else:
            return self.tcr_graph_dataset[index], self.pmhc_graph_dataset[index], label

class TCRBindDataModule(pl.LightningDataModule):
    """
    Dataset from:
    Jha, K., Saha, S. & Singh, H. Prediction of protein–protein interaction using graph neural networks. 
    Sci Rep 12, 8360 (2022). https://doi.org/10.1038/s41598-022-12201-9
    """
    def __init__(self, tsv_path: str = None, processed_dir: str = None, 
                target='epitope', low: int = 50, high: int = 800, include_seq_data: bool=False,
                batch_size: int = 32, num_workers: int = 0, device=torch.device('cpu')):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(tsv_path, sep='\t')

        self.dataset = None

        self.train: TCRBindDataset = None
        self.val: TCRBindDataset = None
        self.test: TCRBindDataset = None

        self.selected_targets = None

    def setup(self, train_size: int = 0.8, random_seed: int = 42):
        assert train_size > 0 and train_size <= 1, "train_size must be in (0, 1]"
        train_df, test_df, self.selected_targets = hard_split_df(self.df, target_col=self.hparams.target, min_ratio=train_size,
                                                    low=self.hparams.low, high=self.hparams.high, random_seed=self.hparams.random_seed)

        self.train = TCRBindDataset(train_df, self.hparams.processed_dir, self.hparams.include_seq_data)
        self.test = TCRBindDataset(test_df, self.hparams.processed_dir, self.hparams.include_seq_data)

        return self.selected_targets

    # custom collate for paired dataset 
    # see: https://github.com/pyg-team/pytorch_geometric/issues/781
    def collate(self, data_list):
        batch_A = Batch.from_data_list([data[0] for data in data_list])
        batch_B = Batch.from_data_list([data[1] for data in data_list])
        batch_label = Batch.from_data_list([data[-1] for data in data_list])
        if self.hparams.include_seq_data:
            return batch_A, batch_B, batch_label
        else:
            batch_seq_A = Batch.from_data_list([data[2] for data in data_list])
            batch_seq_B = Batch.from_data_list([data[3] for data in data_list])
            batch_seq_C = Batch.from_data_list([data[4] for data in data_list])
            return batch_A, batch_B, batch_seq_A, batch_seq_B, batch_seq_C, batch_label

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore

    def val_dataloader(self):
        # TODO:
        raise NotImplementedError
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.collate)  # type: ignore
