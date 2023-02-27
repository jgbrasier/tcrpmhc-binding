import os
import sys
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from graphein.protein.graphs import (ProteinGraphConfig,
                                     construct_graph, 
                                     deprotonate_structure, 
                                     process_dataframe, 
                                     initialise_graph_with_metadata, 
                                     add_nodes_to_graph,
                                     compute_edges
                                    )
from graphein.protein.edges.distance import (add_distance_threshold,
                                             add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )
from graphein.protein.features.sequence.embeddings import esm_residue_embedding, compute_esm_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph
from src.processing.graph import (convert_nx_to_pyg_data, 
                                  read_pdb_to_dataframe, 
                                  seperate_tcr_pmhc,
                                  build_residue_dist_threshold_graph,
                                  build_residue_contact_graph,
                                  split_af2_tcrpmhc_df,
                                  bound_pdb_to_pyg,
                                  compute_residue_embedding,
                                )


tsv_path = 'data/preprocessed/run329_results.tsv'

df = pd.read_csv(tsv_path, sep='\t')

# encoder = partial(compute_esm_embedding, representation='residue', model_name = "esm1b_t33_650M_UR50S", output_layer = 33)


### ----------------------------------------------------------------------------------
### seperate --> unbind tcr and pmhc structures
# out_dir = 'data/graphs/run329_results'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# for i in tqdm(df.index):
#     pdb_id = str(df.iloc[i]['uuid'])
#     pdb_path = os.path.join(pdb_dir, 'model_'+str(pdb_id)+'.pdb')
#     chain_seq = str(df.iloc[i]['chainseq']).split('/')
#     # make dir
#     tcr_pt, pmhc_pt = bound_pdb_to_pyg(pdb_path=pdb_path, pdb_id=pdb_id,
#                                     embedding_function=encoder,
#                                     df_processing_function=partial(split_af2_tcrpmhc_df, chain_seq=chain_seq),
#                                     egde_dist_threshold=6.)
#     # save graphs
#     torch.save(tcr_pt, os.path.join(out_dir, f"{pdb_id}_tcr.pt"))
#     torch.save(pmhc_pt, os.path.join(out_dir, f"{pdb_id}_pmhc.pt"))
#     # we do not need to save the label as it is stored in self.data


### ----------------------------------------------------------------------------------
### keep them together
# out_dir = 'data/graphs/run329_results_bound'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# for i in tqdm(df.index):
#     pdb_id = str(df.iloc[i]['uuid'])
#     pdb_path = os.path.join(pdb_dir, 'model_'+str(pdb_id)+'.pdb')
#     params = {
#             "granularity": 'CA',
#             'verbose': False,
#             'exclude_waters': True,
#             'deprotonate': True,
#             "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=6.0)],
#             'graph_metadata_functions': [partial(esm_residue_embedding, model_name="esm1b_t33_650M_UR50S", output_layer=33)],
#         }
#     config = ProteinGraphConfig(**params)
#     g = construct_graph(config=config, pdb_path=pdb_path)
#     pt = convert_nx_to_pyg_data(g, node_feat_name='esm_embedding')
#     # save graphs
#     torch.save(pt, os.path.join(out_dir, f"{pdb_id}.pt"))


### ---------------------------------------------------------------------------------
### DISTANCE MATRIX
# dist_mat_dir = '/n/data1/hms/dbmi/zitnik/lab/users/jb611/dist_mat/run329_results_bound'
# pdb_dir = '/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/run329_results_for_jg'


# if not os.path.exists(dist_mat_dir):
#     os.makedirs(dist_mat_dir)
# params = {
#         "granularity": 'CA',
#         'verbose': False,
#         'exclude_waters': True,
#         'deprotonate': True,
#         "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=6.0)],
#         'graph_metadata_functions': [partial(esm_residue_embedding, model_name="esm1b_t33_650M_UR50S", output_layer=33)],
#     }
# config = ProteinGraphConfig(**params)

# # pdb_paths = [os.path.join(pdb_dir, 'model_'+str(str(df.iloc[i]['uuid']))+'.pdb') for i in df.index][:5]
# # graphs_dict = construct_graphs_mp(pdb_code_it=[pdb_paths], config=config, return_dict=True, num_cores=8)
# # for k, v in graphs_dict.items():
# #     np.save(v, k)
# for i in tqdm(df.index):
#     pdb_id = str(df.iloc[i]['uuid'])
#     pdb_path = os.path.join(pdb_dir, 'model_'+str(pdb_id)+'.pdb')
#     save_path = os.path.join(dist_mat_dir, pdb_id+'.npy')
#     if os.path.exists(save_path):
#         continue
#     else:
#         g = construct_graph(config=config, pdb_path=pdb_path)
#         np.save(save_path, np.array(g.graph['dist_mat']))

### ---------------------------------------------------------------------------------
### CONTACT GRAPH

print(os.listdir())

tsv_path = 'data/preprocessed/run329_results.tsv'
pdb_dir = 'data/pdb/run329_results_for_jg'
out_dir = 'data/graphs/run329_results_contact'

data = pd.read_csv(tsv_path, sep='\t')

encoder = partial(compute_esm_embedding, representation='residue', model_name = "esm1b_t33_650M_UR50S", output_layer = 33)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for i in tqdm(df.index):
    if os.path.exists(os.path.join(out_dir, f"{pdb_id}.pt")):
        continue
    pdb_id = str(df.iloc[i]['uuid'])
    pdb_path = os.path.join(pdb_dir, 'model_'+str(pdb_id)+'.pdb')
    chain_seq = (data.iloc[i]['chainseq']).split('/')

    raw_df, header = read_pdb_to_dataframe(pdb_path=pdb_path, parse_header=False)
    g = build_residue_contact_graph(raw_df, pdb_id, chain_seq, \
                                    intra_edge_dist_threshold=5.,
                                    contact_dist_threshold=8.)
    g = compute_residue_embedding(g, encoder)
    pt = convert_nx_to_pyg_data(g, node_feat_name='embedding')
    torch.save(pt, os.path.join(out_dir, f"{pdb_id}.pt"))