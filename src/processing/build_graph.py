from functools import partial
from typing import Optional

import re
import pandas as pd
import networkx as nx
from biopandas.pdb import PandasPdb

import torch
from torch_geometric.data import Data

from prody import parsePDBHeader
from graphein.protein.graphs import process_dataframe, deprotonate_structure, convert_structure_to_centroids, subset_structure_to_atom_type, filter_hetatms, remove_insertions
from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph, compute_edges
from graphein.protein.edges import add_peptide_bonds, add_hydrogen_bond_interactions, add_distance_threshold
from graphein.protein import plotly_protein_structure_graph

def find_chain_names(header: dict):
    flag_dict = {
    'tra': {'base': ['tcr', 't-cell', 't cell', 't'], 'variant': ['alpha', 'valpha', 'light']},
    'trb': {'base': ['tcr', 't-cell', 't cell', 't'], 'variant': ['beta', 'vbeta', 'heavy']},
    'b2m': ['beta-2-microglobulin', 'beta 2 microglobulin', 'b2m'],
    'epitope': ['peptide', 'epitope', 'protein', 'self-peptide', 'nuclear'],
    'mhc': ['mhc', 'hla', 'hla class i', 'mhc class i']
    }

    chain_key_dict = {k: list() for k in flag_dict}

    chain_keys = [key for key in header.keys() if len(key)==1]

    for chain in chain_keys:
        name = re.split(';|,| ', str(header[chain].name).lower())
        for key in flag_dict:
            if key in ['tra', 'trb']:
                if bool(set(name) & set(flag_dict[key]['base'])) & bool(set(name) & set(flag_dict[key]['variant'])):
                    chain_key_dict[key].append(chain)
            else:
                if bool(set(name) & set(flag_dict[key])):
                    chain_key_dict[key].append(chain)

    for k, v in chain_key_dict.items():
        if len(v)==0:
            raise ValueError('Header parsing error for key: {} in protein {}'.format(k, header['identifier']))
    return chain_key_dict

def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
    ) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a dataframe enumerating all atoms and their cartesian coordinates in 3D space. Also
    contains associated metadata from the PDB file.

    :param pdb_path: path to PDB file. Defaults to ``None``.
    :type pdb_path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to ``None``.
    :type pdb_code: str, optional
    :param uniprot_id: UniProt ID to build graph from AlphaFoldDB. Defaults to ``None``.
    :type uniprot_id: str, optional
    :param model_index: Index of model to read. Only relevant for structures containing ensembles. Defaults to ``1``.
    :type model_index: int, optional
    :param verbose: print dataframe?
    :type verbose: bool
    :param granularity: Specifies granularity of dataframe. See :class:`~graphein.protein.config.ProteinGraphConfig` for further
        details.
    :type granularity: str
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and pdb_path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, pdb_path or uniprot_id must be specified!"
        )

    if pdb_path is not None:
        atomic_df = PandasPdb().read_pdb(pdb_path)
        header = parsePDBHeader(pdb_path)
        header['chain_key_dict'] = find_chain_names(header)
    elif uniprot_id is not None:
        atomic_df = PandasPdb().fetch_pdb(
            uniprot_id=uniprot_id, source="alphafold2-v2"
        )
    else:
        atomic_df = PandasPdb().fetch_pdb(pdb_code)

    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header

def seperate_tcr_pmhc(df: pd.DataFrame, chain_key_dict: dict):
    # each value of chain_key_dict is a list, can concatenate using +
    tcr_df = df.loc[df['chain_id'].isin(chain_key_dict['tra']+chain_key_dict['trb'])]
    # tcr_df = df.loc[df['chain_id'].isin(chain_key_dict['tra'])]

    pmhc_df = df.loc[df['chain_id'].isin(chain_key_dict['mhc']+chain_key_dict['b2m']+chain_key_dict['epitope'])]
    return tcr_df, pmhc_df


def build_residue_graph(raw_df: pd.DataFrame, pdb_code: str, egde_dist_threshold: int =10.):

    atom_processing_funcs = [deprotonate_structure, remove_insertions, convert_structure_to_centroids]
    
    df = process_dataframe(raw_df, atom_df_processing_funcs=atom_processing_funcs)
    g = initialise_graph_with_metadata(protein_df=df, # from above cell
                                   raw_pdb_df=raw_df, # Store this for traceability
                                   pdb_code = pdb_code, #and again
                                   granularity = "centroid" # Store this so we know what kind of graph we have
                                  )
    g = add_nodes_to_graph(g)
    g = compute_edges(g, get_contacts_config=None, funcs=[partial(add_distance_threshold, long_interaction_threshold=5, threshold=egde_dist_threshold)])

    return g

def convert_nx_to_pyg_data(G: nx.Graph) -> Data:
    # Initialise dict used to construct Data object
    # data = {k: v for k, v in sequence_data.items()}
    data = {"node_id": list(G.nodes())}
    
    G = nx.convert_node_labels_to_integers(G)

    # Construct Edge Index
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    # Add node features
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    # Add edge features
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = (
                [value] if i == 0 else data[str(key)] + [value]
            )

    # Add graph-level features
    for feat_name in G.graph:
        data[str(feat_name)] = [G.graph[feat_name]]

    data["edge_index"] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

if __name__=='__main__':
    import os
    datadir = 'data/pdb/iedb_3d_resolved'
    pdb_code = '1QRN'
    pdb_path = os.path.join(datadir, pdb_code+'.pdb')

    data = pd.read_csv('data/preprocessed/iedb_3d_binding_test.tsv', sep='\t')
    epitope = data[data['id'] == pdb_code]['epitope'].str.lower().values
    raw_df, header = read_pdb_to_dataframe(pdb_path=pdb_path)

    print(header['chain_key_dict'])
    tcr_raw_df, pmhc_raw_df = seperate_tcr_pmhc(raw_df, header['chain_key_dict'])

    tcr_g = build_residue_graph(tcr_raw_df, pdb_code)
    pmhc_g = build_residue_graph(pmhc_raw_df, pdb_code)

    p = plotly_protein_structure_graph(
        tcr_g,
        colour_edges_by="kind",
        colour_nodes_by="seq_position",
        label_node_ids=False,
        plot_title="{} TCR alpha/beta chain Residue Graph".format(pdb_code),
        node_size_multiplier=1
        )
    p.show()
