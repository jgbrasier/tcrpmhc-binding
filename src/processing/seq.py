import pandas as pd

from tqdm import tqdm

from typing import Callable, Dict, Generator, List, Optional


def embed_seq_data(df: pd.DataFrame, seq_embedding_function: Callable)  -> pd.DataFrame:
    """Embed cdr3a, cdr3b and epitope sequences using an embedding function of choice

    :param df: Dataframe containing cdr3a, cdr3b and epitope sequences
    :type df: pd.DataFrame
    :param seq_embedding_function: residue level embedding function must take a string of fasta residues as input 
                                    eg: "AVRPTSGGSYIPT"
    :type seq_embedding_function: Callable, optional
    :return: embedded sequence dictionary where sequence ordering is preserved
    :rtype: pd.DataFrame
    """
    embedding_dict = {'cdr3a': [], 'cdr3b': [], 'epitope': []}
    for i in tqdm(range(len(df.index))):
        cdr3a_emb = seq_embedding_function(str(df.iloc[i]['cdr3a']))
        cdr3b_emb = seq_embedding_function(str(df.iloc[i]['cdr3b']))
        epitope_emb = seq_embedding_function(str(df.iloc[i]['epitope']))

        embedding_dict['cdr3a'].append(cdr3a_emb)
        embedding_dict['cdr3b'].append(cdr3b_emb)
        embedding_dict['epitope'].append(epitope_emb)

    return pd.DataFrame.from_dict(embedding_dict)