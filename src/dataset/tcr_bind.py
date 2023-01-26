from typing import Callable, Dict, Generator, List, Optional
from collections import namedtuple
import os

from tqdm import tqdm
import pandas as pd

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

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
    def __init__(self, tsv_path: str, data_dir: str): 

        # load data
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.data_dir = data_dir
        self.df['path'] = [os.path.join(self.data_dir, str(id)) for id in self.df['id']]

        self.cdr3a_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='cdr3a_seq_emb')
        self.cdr3b_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='cdr3b_seq_emb')
        self.epitope_seq_emb_dataset = TCRPartialDataset(self.df['path'], _type='epitope_seq_emb')
        self.tcr_graph_dataset = TCRPartialDataset(self.df['path'], _type='tcr_graph')
        self.pmhc_graph_dataset = TCRPartialDataset(self.df['path'], _type='pmhc_graph')

        assert len(self.cdr3a_seq_emb_dataset) == len(self.cdr3b_seq_emb_dataset) \
            and len(self.cdr3b_seq_emb_dataset) == len(self.epitope_seq_emb_dataset) \
            and len(self.epitope_seq_emb_dataset) == len(self.tcr_graph_dataset) \
            and len(self.tcr_graph_dataset) == len(self.pmhc_graph_dataset), \
            "Dataset length mismatch: \n \
                cdr3a_seq_emb_dataset: {} \n \
                cdr3b_seq_emb_dataset: {} \n \
                epitope_seq_emb_dataset: {} \n \
                tcr_graph_dataset: {} \n \
                pmhc_graph_dataset: {} \
                    ".format(len(self.cdr3a_seq_emb_dataset), len(self.cdr3b_seq_emb_dataset), len(self.epitope_seq_emb_dataset), \
                        len(self.tcr_graph_dataset), len(self.pmhc_graph_dataset))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        Graph = namedtuple("graph", "prot1 prot2")
        Emb_Seq = namedtuple("emb_seq", "cdr3a cdr3b epitope")

        # if self._test:
        #     graph = Graph(self.tcr_graph_dataset[index], self.pmhc_graph_dataset[index])
        #     emb_seq = Emb_Seq(self.cdr3a_seq_emb_dataset[index], self.cdr3b_seq_emb_dataset[index], \
        #             self.epitope_seq_emb_dataset[index])
        #     return graph, emb_seq
        # else:
        label = torch.tensor(self.df.iloc[index]['binding'])
        graph = Graph(self.tcr_graph_dataset[index], self.pmhc_graph_dataset[index])
        emb_seq = Emb_Seq(self.cdr3a_seq_emb_dataset[index], self.cdr3b_seq_emb_dataset[index], \
                self.epitope_seq_emb_dataset[index])
        return graph, emb_seq, label