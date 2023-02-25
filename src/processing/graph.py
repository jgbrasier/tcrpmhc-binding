from typing import Callable, Dict, Generator, List, Optional, Union, Tuple
import string
import re
import os

import numpy as np

from tqdm import tqdm

import pandas as pd
from biopandas.pdb import PandasPdb

import networkx as nx

import torch
from torch_geometric.data import Data, Dataset

from prody import parsePDBHeader

from functools import partial
from graphein.protein.utils import (
    compute_rgroup_dataframe,
    filter_dataframe,
    get_protein_name_from_filename,
    three_to_one_with_mods,
)
from graphein.protein.graphs import (
    process_dataframe, 
    deprotonate_structure, 
    subset_structure_to_atom_type, 
    filter_hetatms, 
    remove_insertions, 
    add_nodes_to_graph, 
    compute_edges,
)
from graphein.protein.edges import (
    add_peptide_bonds, 
    add_hydrogen_bond_interactions, 
    add_distance_threshold,
)
from graphein.protein import compute_distmat, get_interacting_atoms
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
    parse_header: bool = True,
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
        if parse_header:
            header = parsePDBHeader(pdb_path)
            header['chain_key_dict'] = find_chain_names(header)
        else:
            header = None
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

def split_af2_tcrpmhc_df(df: pd.DataFrame, chain_seq, rescale_residue_number: Optional[bool] = False):
    d = []
    out = []
    for res in df.groupby('residue_number'):
        aa = three_to_one_with_mods(res[1]['residue_name'].drop_duplicates().values[0])
        d.append((aa, res[1]['residue_name'].index.tolist()))

    for idx, seq in enumerate(chain_seq):
        slice = []
        z = list(zip(seq, d)).copy()
        assert [x[0] for x in z] == [x[1][0] for x in z]
        for i, x in enumerate(z):
            slice += x[1][1]
            del(d[0])
        seq_df = df.iloc[slice]
        seq_df['chain_id'] = string.ascii_uppercase[idx]
        if rescale_residue_number:
            seq_df['residue_number'] = seq_df['residue_number'] - seq_df['residue_number'].min()
        out.append(seq_df)
    # from finetuned AF2 model: sequence are in order: pmhc, epitope, tra, trb
    # TODO: make this more generalizable
    pmhc_df = pd.concat((out[0], out[1]))
    tcr_df = pd.concat((out[2], out[3]))
    return tcr_df, pmhc_df

def get_contact_atoms(df1: pd.DataFrame, df2:pd.DataFrame, threshold:float, deprotonate=True, coord_names=['x_coord', 'y_coord', 'z_coord']):
    assert all(df1.columns == df2.columns), "DataFrame column names must match"
    if deprotonate:
        df1 = deprotonate_structure(df1)
        df2 = deprotonate_structure(df2)

    # Extract coordinates from dataframes
    coords1 = df1[coord_names].to_numpy()
    coords2 = df2[coord_names].to_numpy()

    # Compute pairwise distances between atoms
    dist_matrix = np.sqrt(((coords1[:, None] - coords2) ** 2).sum(axis=2))

    # Create a new dataframe containing pairs of atoms whose distance is below the threshold
    pairs = np.argwhere(dist_matrix < threshold)
    atoms1, atoms2 = df1.iloc[pairs[:, 0]], df2.iloc[pairs[:, 1]]
    atoms1_id = atoms1['chain_id'].map(str) + ":" + atoms1['residue_name'].map(str) + ":" + atoms1['residue_number'].map(str)
    atoms2_id = atoms2['chain_id'].map(str) + ":" + atoms2['residue_name'].map(str) + ":" + atoms2['residue_number'].map(str)
    node_pairs = np.vstack((atoms1_id.values, atoms2_id.values)).T
    result = pd.concat([df1.iloc[np.unique(pairs[:, 0])], df2.iloc[np.unique(pairs[:, 1])]])
    return result, node_pairs

def get_all_residue_atoms(partial_df: pd.DataFrame, full_df: pd.DataFrame):
    assert all(partial_df.columns == full_df.columns), "DataFrame column names must match"
    return full_df[full_df['residue_number'].isin(partial_df['residue_number'])]



def add_intra_chain_distance_threshold(G: nx.Graph, chains: Union[List[str], str], threshold: float):
    if isinstance(chains, str) and len(chains) == 1:
        chains = list(chains)
    
    node_list = [n for n, d in G.nodes(data=True) if d["chain_id"] in chains]
    pdb_df = filter_dataframe(G.graph["pdb_df"], "node_id", node_list, True)
    dist_mat = compute_distmat(pdb_df)
    interacting_nodes = get_interacting_atoms(threshold, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))
    for a1, a2 in interacting_nodes:
        n1 = pdb_df.loc[a1, "node_id"]
        n2 = pdb_df.loc[a2, "node_id"]

        if (n1 != n2) and not G.has_edge(n1, n2):
            G.add_edge(n1, n2, kind={"intra_distance_threshold"})


def add_inter_chain_distance_threshold(G: nx.Graph, chains_1: Union[List[str], str], \
                                       chains_2: Union[List[str], str], threshold: float):
    if isinstance(chains_1, str) and len(chains_1) == 1:
        chains_1 = list(chains_1)
    if isinstance(chains_2, str) and len(chains_2) == 1:
        chains_2 = list(chains_2)
    pdb_df = filter_dataframe(G.graph["pdb_df"], "node_id", list(G.nodes()), True)
    dist_mat = compute_distmat(pdb_df)
    interacting_nodes = np.argwhere(dist_mat <= threshold)
    get_interacting_atoms(threshold, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))
    for a1, a2 in interacting_nodes:
        n1 = G.graph["pdb_df"].loc[a1, "node_id"]
        n2 = G.graph["pdb_df"].loc[a2, "node_id"]
        n1_chain_cat = 1 if G.graph["pdb_df"].loc[a1, "chain_id"] in chains_1 else 2
        n2_chain_cat = 1 if G.graph["pdb_df"].loc[a2, "chain_id"] in chains_1 else 2

        cond_1 = (n1_chain_cat != n2_chain_cat)
        cond_2 = G.has_edge(n1, n2)
        cond_3 = (n1 != n2)
        if cond_1 and not cond_2 and cond_3:
            print(n1, n2)
            G.add_edge(n1, n2, kind={"inter_distance_threshold"})

def add_edge_from_pairs(G: nx.Graph, pairs: List[Union[List[str], Tuple[str]]], kind: Optional[str] ='from_pair'):
    """Add edges between pairs of nodes on a graph G

    :param G: Graph to add the edges
    :type G: nx.Graph
    :param pairs: Array or List of  pairs, ex: [(A:GLY:332, B:LEU:928), ...]
    :type pairs: List(Union[tuple, list])
    """
    for n1, n2 in pairs:
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, kind={kind})


def seperate_tcr_pmhc(df: pd.DataFrame, chain_key_dict: dict = None, include_b2m=False):
    # each value of chain_key_dict is a list, can concatenate using +
    tcr_df = df.loc[df['chain_id'].isin(chain_key_dict['tra']+chain_key_dict['trb'])]
    # tcr_df = df.loc[df['chain_id'].isin(chain_key_dict['tra'])]
    if include_b2m:
        pmhc_df = df.loc[df['chain_id'].isin(chain_key_dict['mhc']+chain_key_dict['b2m']+chain_key_dict['epitope'])]
    else:
        pmhc_df = df.loc[df['chain_id'].isin(chain_key_dict['mhc']+chain_key_dict['epitope'])]
    return tcr_df, pmhc_df

def build_residue_contact_graph(raw_df: pd.DataFrame, pdb_code: str,  chain_seq: List[str], intra_edge_dist_threshold: int = 5., contact_dist_threshold: int = 8.):
    raw_df = deprotonate_structure(raw_df)
    tcr_df, pmhc_df = split_af2_tcrpmhc_df(raw_df, chain_seq)
    contact_df, pairs = get_contact_atoms(tcr_df, pmhc_df, threshold=contact_dist_threshold)
    df = get_all_residue_atoms(contact_df, pd.concat((tcr_df, pmhc_df)))
    df = process_dataframe(df,
                            chain_selection = "all",
                            insertions = False,
                            deprotonate = True,
                            keep_hets = [],
                            granularity='CA') # alpha carbon
    g = initialise_graph_with_metadata(protein_df=df, # from above cell
                                    raw_pdb_df=raw_df, # Store this for traceability
                                    pdb_code = pdb_code, #and again
                                    granularity = 'CA' # Store this so we know what kind of graph we have
                                    )
    g = add_nodes_to_graph(g)
    # g = compute_edges(g, funcs=[add_atomic_edges, add_bond_order])
    g = compute_edges(g, funcs=[
        partial(add_intra_chain_distance_threshold, chains=['A', 'B'], threshold=intra_edge_dist_threshold),
        partial(add_intra_chain_distance_threshold, chains=['C', 'D'], threshold=intra_edge_dist_threshold),
        partial(add_edge_from_pairs, pairs=pairs, kind='inter_chain')
        # partial(add_inter_chain_distance_threshold, chains_1=['A', 'B'], chains_2=['C', 'D'], threshold=8.),
    ])
    return g

def build_residue_dist_threshold_graph(raw_df: pd.DataFrame, pdb_code: str, egde_dist_threshold: int =6.):
    df = process_dataframe(raw_df,
                            chain_selection = "all",
                            insertions = False,
                            deprotonate = True,
                            keep_hets = [],
                            granularity='CA') # alpha carbon
    g = initialise_graph_with_metadata(protein_df=df, # from above cell
                                   raw_pdb_df=raw_df, # Store this for traceability
                                   pdb_code = pdb_code, #and again
                                   granularity = "CA" # Store this so we know what kind of graph we have
                                  )
    g = add_nodes_to_graph(g)
    g = compute_edges(g, funcs=[partial(add_distance_threshold, long_interaction_threshold=1, threshold=egde_dist_threshold)])
    return g

def compute_residue_embedding(
    G: nx.Graph,
    embedding_function: Callable = None
    ) -> nx.Graph:
    """
    Computes residue embeddings from a protein sequence and adds the to the graph.

    :param G: ``nx.Graph`` to add    esm embedding to.
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
            pass
            # G.nodes[n]["embedding"] = embedding[int(G.nodes[n]['residue_number'])]

    return G

def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    full_df: pd.DataFrame,
    granularity: str,
    name: Optional[str] = None,
    pdb_code: Optional[str] = None,
    pdb_path: Optional[str] = None,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param protein_df: Processed Dataframe of prot
    ein structure.
    :type protein_df: pd.DataFrame
    :param full_df: Full dataframe of protein structure for comparison and traceability downstream.
    :type full_df: pd.DataFrame
    :param granularity: Granularity of the graph (eg ``"atom"``, ``"CA"``, ``"CB"`` etc or ``"contact"``).
        If ``"contact"``, will automatically fetch the entire protein sequenec (and not just the contact subset)
        See: :const:`~graphein.protein.config.GRAPH_ATOMS` and :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param name: specified given name for the graph. If None, the PDB code or the file name will be used to name the graph.
    :type name: Optional[str], defaults to ``None``
    :param pdb_code: PDB ID / Accession code, if the PDB is available on the PDB database.
    :type pdb_code: Optional[str], defaults to ``None``
    :param pdb_path: path to local PDB file, if constructing a graph from a local file.
    :type pdb_path: Optional[str], defaults to ``None``
    :return: Returns initial protein structure graph with metadata.
    :rtype: nx.Graph
    """

    # Get name for graph if no name was provided
    if name is None:
        if pdb_path is not None:
            name = get_protein_name_from_filename(pdb_path)
        else:
            name = pdb_code

    G = nx.Graph(
        name=name,
        pdb_code=pdb_code,
        pdb_path=pdb_path,
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        full_df=full_df,
        rgroup_df=compute_rgroup_dataframe(remove_insertions(full_df)),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        if granularity == "rna_atom":
            sequence = protein_df.loc[protein_df["chain_id"] == c][
                "residue_name"
            ].str.cat()
        elif granularity == "contact":
            sequence = (
                full_df.loc[full_df["chain_id"] == c]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        else:
            sequence = (
                protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        G.graph[f"sequence_{c}"] = sequence
    return G

def convert_nx_to_pyg_data(G: nx.Graph, node_feat_name: str, edge_feat_name: Union[List[str], str] = None, graph_features:bool =False) -> Data:
    # Initialise dict used to construct Data object
    # data = {k: v for k, v in sequence_data.items()}
    data = {"node_id": list(G.nodes())}
    
    G = nx.convert_node_labels_to_integers(G)

    # Construct Edge Index
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    # Add node features
    node_features = []
    for i, (n, d) in enumerate(G.nodes(data=True)):
        node_features.append(G.nodes[n][node_feat_name])
    data['x'] = torch.from_numpy(np.array(node_features))

    # Add edge features
    if edge_feat_name:
        pos_dict = {n: i for i, n in enumerate(edge_feat_name)}
        edge_features = np.zeros((len(G.edges(data=True)), len(edge_feat_name)))
        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                edge_features[i, pos_dict[key]] = value
        data['edge_features'] = torch.from_numpy(edge_features)

    # Add graph-level features
    if graph_features:
        for feat_name in G.graph:
            if not str(feat_name).startswith('sequence'):
                data[str(feat_name)] = [G.graph[feat_name]]

    data["edge_index"] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

def bound_pdb_to_pyg(pdb_path: str, 
                    pdb_id: str,
                    df_processing_function: Callable = None,
                    embedding_function: Callable = None, 
                    include_b2m=False,
                    egde_dist_threshold: int = 6.):
    """ 
    reads bound TCR-pMHC files in a directory, splits them into 
    TCR and pMHC residue level graphs with node level embedings

    :param pdb_path: path/to/pdb_file
    :type pdb_path: str
    :param pdb_id: pdb id (or uuid) for storage redundancy
    :type pdb_id: str
    :param embedding_function: function to compute residue embedding from protein sequence
    :type embedding_function: Callable
    :param egde_dist_threshold: inter-residue distance to build graph edges, defaults to 6.
    :type egde_dist_threshold: int, optional
    :return: TCR and pMHC residue level graphs
    :rtype: tuple(PyTorch Geometric graphs)
    """
    parse_header = False if df_processing_function else True
    raw_df, header = read_pdb_to_dataframe(pdb_path=pdb_path, parse_header=parse_header)
    if not parse_header:
        tcr_raw_df, pmhc_raw_df = df_processing_function(raw_df)
    else:
        tcr_raw_df, pmhc_raw_df = seperate_tcr_pmhc(raw_df, header['chain_key_dict'], include_b2m=include_b2m)
    
    # TCR graph
    tcr_g = build_residue_dist_threshold_graph(tcr_raw_df, pdb_id, egde_dist_threshold=egde_dist_threshold)
    tcr_g = compute_residue_embedding(tcr_g, embedding_function)
    # tra_seq_data =  seq_data[['va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b']]
    tcr_pt = convert_nx_to_pyg_data(tcr_g, node_feat_name='embedding')

    # pMHC graph
    pmhc_g = build_residue_dist_threshold_graph(pmhc_raw_df, pdb_id,  egde_dist_threshold=egde_dist_threshold)
    pmhc_g = compute_residue_embedding(pmhc_g, embedding_function)
    # pmh_seq_data =  seq_data[['epitope', 'mhc_class', 'mhc']]
    pmh_pt = convert_nx_to_pyg_data(pmhc_g, node_feat_name='embedding')

    return tcr_pt, pmh_pt

def process_pdb(pdb_list: List[str], pdb_dir: str = None, out_path: str = None, seq_embedding_function: Callable = None, \
                is_bound: bool = True, df_processing_function: Callable = None, save_sequence: bool = False, include_b2m: bool = False, ignore: List[str] = list()):
    """reads bound or unbound TCR-pMHC files in a directory
    if bound; splits the TCR and pMHC complexes
    then for each respective complex, computes residue level graphs with node level embedings
    then saves these graphs as well as the corresponding stack of cdr3a, cdr3b and epitope embeddings.
    A unique subdirectory is created for each TCR-pMHC complex

    :param pdb_list: list of pdb ids to processx
    :type pdb_list: List[str]
    :param pdb_dir: directory where pdb files live, defaults to None
    :type pdb_dir: str, optional
    :param out_path: directory to save processed graphs, defaults to None
    :type out_path: str, optional
    :param seq_embedding_function: _description_, defaults to None
    :type seq_embedding_function: Callable, optional
    :param save_sequence: _description_, defaults to False
    :type save_sequence: bool, optional
    :param ignore: list of pdb ids to ignore, defaults to list()
    :type ignore: List[str], optional
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for pdb_id in tqdm(pdb_list):
        # ignore problematic files
        if pdb_id in ignore:
            continue
        pdb_path = os.path.join(pdb_dir, str(pdb_id)+'.pdb')
        # make dir
        if is_bound:
            tcr_pt, pmhc_pt = bound_pdb_to_pyg(pdb_path=pdb_path, pdb_id=pdb_id,
                                            embedding_function=seq_embedding_function,
                                            df_processing_function=df_processing_function,
                                            include_b2m = include_b2m,
                                            egde_dist_threshold=6.)
        else:
            # TODO: for unbound data 
            raise NotImplementedError
        # compute sequence embeddings
        if save_sequence:
            # TODO: sequence embeddings
            raise NotImplementedError
            cdr3a_emb = torch.tensor(seq_embedding_function(str(seq_data['cdr3a'])))
            cdr3b_emb = torch.tensor(seq_embedding_function(str(seq_data['cdr3b'])))
            epitope_emb = torch.tensor(seq_embedding_function(str(seq_data['epitope'])))
            # save sequential embeddings
            torch.save(cdr3a_emb, os.path.join(save_dir, "cdr3a_seq_emb.pt"))
            torch.save(cdr3b_emb, os.path.join(save_dir, "cdr3b_seq_emb.pt"))
            torch.save(epitope_emb, os.path.join(save_dir, "epitope_seq_emb.pt"))

        # save graphs
        torch.save(tcr_pt, os.path.join(out_path, f"{pdb_id}_tcr.pt"))
        torch.save(pmhc_pt, os.path.join(out_path, f"{pdb_id}_pmhc.pt"))

        # we do not need to save the label as it is stored in self.data