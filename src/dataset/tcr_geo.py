from typing import Callable, Dict, Generator, List, Optional
import os

from tqdm import tqdm
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset

from src.processing import process_bound_pdb

class TCRpMHCGraphDataset(Dataset):
    def __init__(self, pdb_dir: str, tsv_path: str): 
        
        self.pdb_dir = pdb_dir
        self.tsv_path = tsv_path

        # load data
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.data['path'] = [os.path.join(self.pdb_dir, str(id)+'.pdb') for id in self.data['id']]

        self.node_embedding_func: Callable = None

    def process_bound_data(self, out_path: Optional[str] = None, node_embedding_function: Callable = None, ignore: List[str] = list()):
        """ reads TCR-pMHC files in a directory, splits them into 
        TCR and pMHC residue level graphs with node level embedings
        and saves these graphs to a specified directory.

        :param out_path: Path so save .pt graphs, defaults to None
        :type out_path: Optional[str], optional
        :param node_embedding_function: function to assign residue embeddings. 
        Input a nx.Graph and outputs a nx.Graph, defaults to None
        :type node_embedding_function: Callable, optional
        :param ignore: List of potential problematic pdb files to ignore., defaults to list()
        :type ignore: List[str], optional
        """
        
        self.node_embedding_func = node_embedding_function

        for i in tqdm(range(len(self.data.index))):
            seq_data = self.data.iloc[i]
            # ignore problematic files 
            if seq_data['id'] in ignore:
                continue

            # make dir
            save_dir = os.path.join(out_path, seq_data['id'])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if len(os.listdir(save_dir)) == 0:
                tcr_pt, pmhc_pt = process_bound_pdb(pdb_path=seq_data['path'], pdb_id=seq_data['id'],
                                                node_embedding_function=node_embedding_function,
                                                egde_dist_threshold=10.)
                # save graphs
                torch.save(tcr_pt, os.path.join(save_dir, "tcr_graph.pt"))
                torch.save(pmhc_pt, os.path.join(save_dir, "pmhc_graph.pt"))
            else:
                continue

    def get(self, idx: int):
        """
        Returns a tuple of 2 PyTorch Geometric Data object for TCR graph and pMHC graph respectively,
        a tuple of The associated sequential information
        And the binding label

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        if self.chain_selection_map is not None:
            return torch.load(
                os.path.join(
                    self.processed_dir,
                    f"{self.structures[idx]}_{self.chain_selection_map[idx]}.pt",
                )
            )
        else:
            return torch.load(
                os.path.join(self.processed_dir, f"{self.structures[idx]}.pt")
            )