import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (add_distance_threshold,
                                             add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )
from graphein.protein.graphs import construct_graph
from graphein.protein.features.sequence.embeddings import esm_residue_embedding, compute_esm_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph
from src.processing.graph import (convert_nx_to_pyg_data, 
                                  read_pdb_to_dataframe,
                                  seperate_tcr_pmhc,
                                  build_residue_dist_threshold_graph,
                                  compute_residue_embedding,
                                  split_af2_tcrpmhc_df,
                                  bound_pdb_to_pyg,
                                )

import torch

import warnings
warnings.filterwarnings("ignore")

pdb_codes = os.listdir('data/pdb/pan_human')
pdb_dir = 'data/pdb/pan_human'
out_dir =  'data/graphs/pan_human_new'

encoder = partial(compute_esm_embedding, representation='residue', model_name = "esm1b_t33_650M_UR50S", output_layer = 33)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
ignore = []
for pdb in tqdm(pdb_codes):
    try:
        # params = {
        #     "pdb_dir": processed_dir,
        #     "granularity": 'CA',
        #     'verbose': False,
        #     'exclude_waters': True,
        #     'deprotonate': True,
        #     "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=6.0)],
        #     'graph_metadata_functions': [partial(esm_residue_embedding, model_name="esm1b_t33_650M_UR50S", output_layer=33)],
        # }
        # config = ProteinGraphConfig(**params)
        # g = construct_graph(config=config, pdb_code=code)
        # pyg = convert_nx_to_pyg_data(g, 'esm_embedding')
        # torch.save(pyg, os.path.join(processed_dir, code+'.pt'))
        pdb_id = pdb.split('.')[0]
        pdb_path = os.path.join(pdb_dir, pdb)
        raw_df, header = read_pdb_to_dataframe(pdb_path=pdb_path, parse_header=False)

        # TCR graph
        g = build_residue_dist_threshold_graph(raw_df, pdb_id, egde_dist_threshold=6.0)
        g = compute_residue_embedding(g, encoder)
        pt = convert_nx_to_pyg_data(g, node_feat_name='embedding')
        torch.save(pt, os.path.join(out_dir, f"{pdb_id}_tcr.pt"))
    except:
        pass




