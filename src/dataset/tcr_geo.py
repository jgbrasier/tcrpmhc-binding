from typing import Callable, Dict, Generator, List, Optional
import os

from tqdm import tqdm
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset

from src.processing import read_pdb_to_dataframe, seperate_tcr_pmhc, convert_nx_to_pyg_data, build_residue_graph

class TCRpMHCGraphDataset(Dataset):
    def __init__(self, pdb_dir: str, tsv_path: str): 
        
        self.pdb_dir = pdb_dir
        self.tsv_path = tsv_path

        # load data
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.data['path'] = [os.path.join(self.pdb_dir, str(id)+'.pdb') for id in self.data['id']]

        self.node_embedding_func: Callable = None

    def process_pdb(self, out_path: Optional[str] = None, node_embedding_function: Callable = None, ignore: List[str] = list()):
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
                raw_df, header = read_pdb_to_dataframe(pdb_path=seq_data['path'])
                tcr_raw_df, pmhc_raw_df = seperate_tcr_pmhc(raw_df, header['chain_key_dict'])
                
                # TCR graph
                tcr_g = build_residue_graph(tcr_raw_df, seq_data['id'], egde_dist_threshold=10.)
                tcr_g = node_embedding_function(tcr_g)
                # tra_seq_data =  seq_data[['va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b']]
                tcr_pt = convert_nx_to_pyg_data(tcr_g)

                # pMHC graph
                pmhc_g = build_residue_graph(pmhc_raw_df, seq_data['id'],  egde_dist_threshold=10.)
                pmhc_g = node_embedding_function(pmhc_g)
                # pmh_seq_data =  seq_data[['epitope', 'mhc_class', 'mhc']]
                pmh_pt = convert_nx_to_pyg_data(pmhc_g)

                # save graphs
                torch.save(tcr_pt, os.path.join(save_dir, f"{seq_data['id']}_tcr.pt"))
                torch.save(tcr_pt, os.path.join(save_dir, f"{seq_data['id']}_pmhc.pt"))
            else:
                continue