import os
import sys
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from tqdm import tqdm

from graphein.protein.graphs import construct_graph
from graphein.protein.features.sequence.embeddings import esm_residue_embedding, compute_esm_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph
from src.processing.graph import (convert_nx_to_pyg_data, 
                                  read_pdb_to_dataframe, 
                                  seperate_tcr_pmhc,
                                  build_residue_graph,
                                  split_af2_tcrpmhc_df,
                                  bound_pdb_to_pyg,
                                )


tsv_path = 'data/preprocessed/run329_results.tsv'
pdb_dir = 'data/pdb/run329_results_for_jg'
out_dir = 'data/graphs/run329_results'

df = pd.read_csv(tsv_path, sep='\t')

encoder = partial(compute_esm_embedding, representation='residue', model_name = "esm1b_t33_650M_UR50S", output_layer = 33)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i in tqdm(df.index):
    pdb_id = str(df.iloc[i]['uuid'])
    pdb_path = os.path.join(pdb_dir, 'model_'+str(pdb_id)+'.pdb')
    chain_seq = str(df.iloc[i]['chainseq']).split('/')
    # make dir
    tcr_pt, pmhc_pt = bound_pdb_to_pyg(pdb_path=pdb_path, pdb_id=pdb_id,
                                    embedding_function=encoder,
                                    df_processing_function=partial(split_af2_tcrpmhc_df, chain_seq=chain_seq),
                                    egde_dist_threshold=6.)
    # save graphs
    torch.save(tcr_pt, os.path.join(out_dir, f"{pdb_id}_tcr.pt"))
    torch.save(pmhc_pt, os.path.join(out_dir, f"{pdb_id}_pmhc.pt"))

        # we do not need to save the label as it is stored in self.data