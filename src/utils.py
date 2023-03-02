from __future__ import print_function
import argparse
import sys
import os
import math
import numpy as np
import pandas as pd

from typing import Callable, Dict, Generator, List, Optional

import torch
from torch.utils.data import Dataset

class PartialDataset(Dataset):
    """
    Dataset for loading list of .pt graph files
    """
    def __init__(self, paths: List[str], 
                _device: torch.device = torch.device('cpu')) -> None:
        self.paths = paths
        self._device = _device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        data = torch.load(self.paths[index], map_location=self._device)
        # data.x = data.x.type(torch.float)
        # data.edge_index = data.edge_index.type(torch.int64)
        # data.edge_index = data.edge_index.type(torch.int64)
        return data

class GraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, id_column: str = 'id', label_column: str ='binder', device=torch.device('cpu')): 

        # load data
        self.df = df
        self.data_dir = data_dir
        npy_ar = np.array(df[[id_column, label_column]].values)
        self.names = npy_ar[:, 0]
        self.labels = npy_ar[:, 1]

        self._device = device
        self._label_column = label_column

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data = torch.load(os.path.join(self.data_dir, str(self.names[index])+'.pt'), map_location=self._device)
        label = torch.tensor(self.labels[index], device=self._device)
        return data, label

AA_3to1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}
AA_1to3 = {v: k for k, v in AA_3to1.items()}



def enc_list_bl_max_len(aa_seqs, blosum, max_seq_len, keep_ends=False):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        seq = seq[1:-1] if keep_ends else seq
        e_seq=np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq

blosum50_full = {
    'A': np.array([ 5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -2, -2, -1, -1, -5]),
    'R': np.array([-2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3, -1, -3,  0, -1, -5]),
    'N': np.array([-1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3,  5, -4,  0, -1, -5]),
    'D': np.array([-2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4,  6, -4,  1, -1, -5]),
    'C': np.array([-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -2, -3, -1, -5]),
    'Q': np.array([-1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3,  0, -3,  4, -1, -5]),
    'E': np.array([-1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3,  1, -3,  5, -1, -5]),
    'G': np.array([ 0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4, -1, -4, -2, -1, -5]),
    'H': np.array([-2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4,  0, -3,  0, -1, -5]),
    'I': np.array([-1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4, -4,  4, -3, -1, -5]),
    'L': np.array([-2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1, -4,  4, -3, -1, -5]),
    'K': np.array([-1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3,  0, -3,  1, -1, -5]),
    'M': np.array([-1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1, -3,  2, -1, -1, -5]),
    'F': np.array([-3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1, -4,  1, -4, -1, -5]),
    'P': np.array([-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -3, -1, -1, -5]),
    'S': np.array([ 1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2,  0, -3,  0, -1, -5]),
    'T': np.array([ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0,  0, -1, -1, -1, -5]),
    'W': np.array([-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3, -5, -2, -2, -1, -5]),
    'Y': np.array([-2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1, -3, -1, -2, -1, -5]),
    'V': np.array([ 0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5, -3,  2, -3, -1, -5]),
    'B': np.array([-2, -1,  5,  6, -3,  0,  1, -1,  0, -4, -4,  0, -3, -4, -2,  0,  0, -5, -3, -3,  6, -4,  1, -1, -5]),
    'J': np.array([-2, -3, -4, -4, -2, -3, -3, -4, -3,  4,  4, -3,  2,  1, -3, -3, -1, -2, -1,  2, -4,  4, -3, -1, -5]),
    'Z': np.array([-1,  0,  0,  1, -3,  4,  5, -2,  0, -3, -3,  1, -1, -4, -1,  0, -1, -2, -2, -3,  1, -3,  5, -1, -5]),
    'X': np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5]),
    '*': np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,  1])
}

blosum50_20aa = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5)),
    }

from typing import Tuple, List

def drop_top_k(df: pd.DataFrame, target: str, k:int, frac: float = 1.0) -> pd.DataFrame:
    """
    Drops a fraction of rows from a pandas DataFrame based on the 
    top k most frequent values in a target column.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        target (str): The name of the target column to be used for dropping rows.
        k (int): The number of top most frequent values to be used for determining which rows to drop.
        frac (float): The fraction of rows to be dropped. 
                    Defaults to 1.0 (i.e. all rows are dropped).

    Returns:
        pd.DataFrame: A new DataFrame with the specified fraction of rows dropped 
                    based on the top k most frequent values in the target column.
    """
    top_k = list(df[target].value_counts()[:k].index)
    idx_to_drop = df[df[target].isin(top_k)].sample(frac=frac).index
    new_df = df.drop(index=idx_to_drop, axis=0)
    return new_df

def hard_split_df(
        df: pd.DataFrame, target_col: str, min_ratio: float, random_seed: float, low: int, high: int, target_values: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """ Assume a target column, e.g. `epitope`.
    Then:
        1) Select random sample
        2) All samples sharing the same value of that column
        with the randomly selected sample are used for test
        3)Repeat until test budget (defined by train/test ratio) is
        filled.
    """
    if target_values:
        # if test target values are given, return train/test df directly
        train_df = df[~df[target_col].isin(target_values)]
        test_df = df[df[target_col].isin(target_values)]
        print("Train size = {:.2f}".format(len(train_df)/len(df)))
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), target_values
    else:
        min_test_len = round((1-min_ratio) * len(df))
        test_len = 0
        selected_target_val = []

        train_df = df.copy()
        test_df = pd.DataFrame()
        
        target_count_df = df.groupby([target_col]).size().reset_index(name='counts')
        target_count_df = target_count_df[target_count_df['counts'].between(low, high, inclusive='both')]
        possible_target_val = list(target_count_df[target_col].unique())
        max_target_len = len(possible_target_val)

        while test_len < min_test_len:
    #         sample = train_df.sample(n=1, random_state=random_state)
    #         target_val = sample[target_col].values[0]
            rng = np.random.default_rng(seed=random_seed)
            target_val = rng.choice(possible_target_val)

            if target_val not in selected_target_val:
                to_test = train_df[train_df[target_col] == target_val]

                train_df = train_df.drop(to_test.index)
                test_df = pd.concat((test_df, to_test), axis=0)
                test_len = len(test_df)

                selected_target_val.append(target_val)
                possible_target_val.remove(target_val)

            if len(selected_target_val) == max_target_len:
                print(f"Possible targets left {possible_target_val}")
                raise Exception('No more values to sample from.')

        print(f"Target {target_col} sequences: {selected_target_val}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), selected_target_val

