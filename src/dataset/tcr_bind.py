from typing import Callable, Dict, Generator, List, Optional
import os

from tqdm import tqdm
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset

from src.processing import process_bound_pdb

from graphein.protein.features.sequence.embeddings import esm_residue_embedding, compute_esm_embedding


class TCRBindDataset(Dataset):
    def __init__(self, pdb_dir: str, tsv_path: str): 
        
        self.pdb_dir = pdb_dir
        self.tsv_path = tsv_path

        # load data
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.data['path'] = [os.path.join(self.pdb_dir, str(id)+'.pdb') for id in self.data['id']]

        # initialize sequence embeddings dictionary
        self.embeddings = {"cdr3a": [], "cdr3b": [], "epitope": []}

        self.node_embedding_func: Callable = None
        self.seq_embedding_func: Callable = None

    def process_bound_data(self, out_path: Optional[str] = None, node_embedding_function: Callable = None, ignore: List[str] = list()):
        """ reads TCR-pMHC files in a directory, splits them into 
        TCR and pMHC residue level graphs with node level embedings
        and saves these graphs as well as the corresponding stack 
        of cdr3a, cdr3b and epitope embeddings, to a specified directory.


        :param out_path: Path so save .pt graphs, defaults to None
        :type out_path: Optional[str], optional
        :param node_embedding_function: function to assign residue embeddings. 
        Input a nx.Graph and outputs a nx.Graph, defaults to None
        :type node_embedding_function: Callable, optional
        :param ignore: List of potential problematic pdb files to ignore., defaults to list()
        :type ignore: List[str], optional
        """
        
        self.node_embedding_func = node_embedding_function

        if isinstance(self.embeddings, dict):
            raise TypeError("Sequence residue embeddings need to be computed using `embed_seq_data`")

        for i in tqdm(range(len(self.data.index))):
            seq_data = self.data.iloc[i]
            seq_emb = self.embeddings.iloc[i]
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

                # concat and save sequential embeddings
                seq_emb_stack = torch.vstack((torch.tensor(seq_emb['cdr3a'], seq_emb['cdr3b'], seq_emb['epitope'])))
                torch.save(seq_emb_stack, os.path.join(save_dir, "seq_emb.pt"))

                # we do not need to save the label as it is stored in self.data

            else:
                continue

    def embed_seq_data(self, seq_embedding_function: Callable):
        """Embed cdr3a, cdr3b and epitope sequences using an embedding function of choice

        :param seq_embedding_function: residue level embedding function must take a string of fasta residues as input 
                                        eg: "AVRPTSGGSYIPT"
        :type seq_embedding_function: Callable, optional
        """
        embedding_dict = {'cdr3a': [], 'cdr3b': [], 'epitope': []}
        for i in tqdm(range(len(self.data.index))):
            cdr3a_emb = seq_embedding_function(str(self.data.iloc[i]['cdr3a']))
            cdr3b_emb = seq_embedding_function(str(self.data.iloc[i]['cdr3b']))
            epitope_emb = seq_embedding_function(str(self.data.iloc[i]['epitope']))

            embedding_dict['cdr3a'].append(cdr3a_emb)
            embedding_dict['cdr3b'].append(cdr3b_emb)
            embedding_dict['epitope'].append(epitope_emb)

            self.embeddings = pd.DataFrame.from_dict(embedding_dict)


    def get(self, idx: int):
        # TODO
        """
        Returns 
         - a tuple of 2 `pytorch_geometric.data.Data` object for TCR graph and pMHC graph respectively,
         - a tuple of the cdr3a, cdr3b, epitope embeddings
         - the binary binding label

        :param idx: Index to retrieve.
        :type idx: int
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