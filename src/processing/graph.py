from typing import Callable, Dict, Generator, List, Optional
import string
import re
import os

from tqdm import tqdm

import pandas as pd
from biopandas.pdb import PandasPdb

import networkx as nx

import torch
from torch_geometric.data import Data, Dataset

from prody import parsePDBHeader

from functools import partial
from graphein.protein.graphs import process_dataframe, deprotonate_structure, convert_structure_to_centroids, subset_structure_to_atom_type, filter_hetatms, remove_insertions
from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph, compute_edges
from graphein.protein.edges import add_peptide_bonds, add_hydrogen_bond_interactions, add_distance_threshold
from graphein.protein import plotly_protein_structure_graph
from graphein.protein.features.sequence.utils import (
    compute_feature_over_chains,
    subset_by_node_feature_value,
)


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

def compute_residue_embedding(
    G: nx.Graph,
    embedding_function: Callable = None
    ) -> nx.Graph:
    """
    Computes residue embeddings from a protein sequence and adds the to the graph.

    :param G: ``nx.Graph`` to add esm embedding to.
    :type G: nx.Graph
    :param embedding_function: function to compute residue embedding from protein sequence
    :type embedding_function: Callable
    :return: ``nx.Graph`` with esm embedding feature added to nodes.
    :rtype: nx.Graph
    """

    for chain in G.graph["chain_ids"]:
        embedding = embedding_function(G.graph[f"sequence_{chain}"])
        # remove start and end tokens from per-token residue embeddings
        embedding = embedding[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["embedding"] = embedding[i]

    return G

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

def bound_pdb_to_pyg(pdb_path: str, pdb_id: str, embedding_function: Callable, egde_dist_threshold: int = 10.):
    """ 
    reads bound TCR-pMHC files in a directory, splits them into 
    TCR and pMHC residue level graphs with node level embedings

    :param pdb_path: path/to/pdb_file
    :type pdb_path: str
    :param pdb_id: pdb id (or uuid) for storage redundancy
    :type pdb_id: str
    :param embedding_function: function to compute residue embedding from protein sequence
    :type embedding_function: Callable
    :param egde_dist_threshold: inter-residue distance to build graph edges, defaults to 10.
    :type egde_dist_threshold: int, optional
    :return: TCR and pMHC residue level graphs
    :rtype: tuple(PyTorch Geometric graphs)
    """
    raw_df, header = read_pdb_to_dataframe(pdb_path=pdb_path)
    tcr_raw_df, pmhc_raw_df = seperate_tcr_pmhc(raw_df, header['chain_key_dict'])
    
    # TCR graph
    tcr_g = build_residue_graph(tcr_raw_df, pdb_id, egde_dist_threshold=egde_dist_threshold)
    tcr_g = compute_residue_embedding(tcr_g, embedding_function)
    # tra_seq_data =  seq_data[['va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b']]
    tcr_pt = convert_nx_to_pyg_data(tcr_g)

    # pMHC graph
    pmhc_g = build_residue_graph(pmhc_raw_df, pdb_id,  egde_dist_threshold=egde_dist_threshold)
    pmhc_g = compute_residue_embedding(pmhc_g, embedding_function)
    # pmh_seq_data =  seq_data[['epitope', 'mhc_class', 'mhc']]
    pmh_pt = convert_nx_to_pyg_data(pmhc_g)

    return tcr_pt, pmh_pt

def process_pdb(self, path_to_tsv: str = None, pdb_dir: str = None, out_path: Optional[str] = None, seq_embedding_function: Callable = None, is_bound: bool = True, ignore: List[str] = list()):
    """ reads bound or unbound TCR-pMHC files in a directory
    if bound; splits the TCR and pMHC complexes
    then computes seperate residue level graphs with node level embedings
    and saves these graphs as well as the corresponding stack 
    of cdr3a, cdr3b and epitope embeddings to a specified directory.

    :param path_to_df: path/to/sequence_dataframe, defaults to None
    :type data_df: str, optional
    :param out_path: Path so save data, defaults to None
    :type out_path: Optional[str], optional
    :param seq_embedding_function: _description_, defaults to None
    :type seq_embedding_function: Callable, optional
    :param is_bound: _description_, defaults to True
    :type is_bound: bool, optional
    :param ignore: _description_, defaults to list()
    :type ignore: List[str], optional
    """
    
    data = pd.read_csv(path_to_tsv, sep='\t')
    data['path'] = [os.path.join(pdb_dir, str(id)+'.pdb') for id in data['id']]

    for i in tqdm(range(len(data.index))):
        seq_data = data.iloc[i]
        # ignore problematic files 
        if seq_data['id'] in ignore:
            continue

        # make dir
        save_dir = os.path.join(out_path, seq_data['id'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if len(os.listdir(save_dir)) == 0:
            if is_bound:
                tcr_pt, pmhc_pt = bound_pdb_to_pyg(pdb_path=seq_data['path'], pdb_id=seq_data['id'],
                                                embedding_function=seq_embedding_function,
                                                egde_dist_threshold=10.)
            else:
                # TODO: 
                raise NotImplementedError
            # compute sequence embeddings
            cdr3a_emb = seq_embedding_function(str(seq_data['cdr3a']))
            cdr3b_emb = seq_embedding_function(str(seq_data['cdr3b']))
            epitope_emb = seq_embedding_function(str(seq_data['epitope']))

            # save graphs
            torch.save(tcr_pt, os.path.join(save_dir, "tcr_graph.pt"))
            torch.save(pmhc_pt, os.path.join(save_dir, "pmhc_graph.pt"))

            # concat and save sequential embeddings
            seq_emb_stack = torch.vstack((torch.tensor(cdr3a_emb, cdr3b_emb, epitope_emb)))
            torch.save(seq_emb_stack, os.path.join(save_dir, "seq_emb.pt"))

            # we do not need to save the label as it is stored in self.data

        else:
            continue