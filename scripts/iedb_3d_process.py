import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial

from graphein.protein.features.sequence.embeddings import esm_residue_embedding, compute_esm_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph

from src.processing import process_pdb

import warnings
warnings.filterwarnings("ignore")

tsv_path = 'data/preprocessed/iedb_3d_binding.tsv'
pdb_dir = 'data/pdb/iedb_3d_resolved'
pt_save_dir = 'data/graphs/iedb_3d_resolved'

pdb_codes = pd.read_csv('data/utils/iedb_3d_pdb_codes.txt', header=None)[0].values.tolist()

process_pdb(pdb_list=pdb_codes,
            pdb_dir=pdb_dir,
            out_path=pt_save_dir,
            seq_embedding_function=partial(compute_esm_embedding, representation='residue',\
                 model_name="esm1b_t33_650M_UR50S", output_layer=33),
            is_bound=True)
