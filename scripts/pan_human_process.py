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
from graphein.protein.features.sequence.embeddings import esm_residue_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph
from src.processing.graph import convert_nx_to_pyg_data

import torch

import warnings
warnings.filterwarnings("ignore")

pdb_codes = pd.read_csv('data/utils/pan_pdb_codes.txt', header=None)[0].values.tolist()
processed_dir =  'data/graphs/pan_human'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
ignore = []
for code in tqdm(pdb_codes):
    try:
        params = {
            "pdb_dir": processed_dir,
            "granularity": 'CA',
            'verbose': False,
            'exclude_waters': True,
            'deprotonate': True,
            "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=6.0)],
            'graph_metadata_functions': [partial(esm_residue_embedding, model_name="esm1b_t33_650M_UR50S", output_layer=33)],
        }
        config = ProteinGraphConfig(**params)
        g = construct_graph(config=config, pdb_code=code)
        pyg = convert_nx_to_pyg_data(g, 'esm_embedding')
        torch.save(pyg, os.path.join(processed_dir, code+'.pt'))
    except:
        pass




