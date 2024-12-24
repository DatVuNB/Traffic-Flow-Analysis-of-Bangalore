from copy import deepcopy
import networkx as nx
# from statistics import geometric_mean
from typing import Dict, Optional, Tuple

from utils.hits import weighted_hits
# from utils.k_clique import k_clique_communities

def _normalize_metric(metric_dict: Dict[str, float]) -> Dict[str, float]:
    """Function to normalize the metric across the whole dictionary according to min-max scale.

    Parameters
    ----------
    metric_dict : { str: float }
        Dictionary to normalize where the keys are nodes and the values the relative metrics results.

    Returns
    -------
    { str: float }
        The normalized dictionary.
    """
    return { n: (v - min(metric_dict.values())) / (max(metric_dict.values()) - min(metric_dict.values())) 
                for n, v in metric_dict.items() }

def get_nodes_in_degree_centrality(network: nx.Graph, normalize: bool = True,
                                   weight: Optional[str] = None) -> Dict[str, float]:
    """Get the in-degree centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = { n: network.in_degree(n, weight=weight) for n in network.nodes() }
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_out_degree_centrality(network: nx.Graph, normalize: bool = True,
                                    weight: Optional[str] = None) -> Dict[str, float]:
    """Get the out-degree centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = { n: network.out_degree(n, weight=weight) for n in network.nodes() }
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_betweenness_centrality(network: nx.Graph, normalize: bool = True, weight: Optional[str] = None,
                                     seed: int = 42) -> Dict[str, float]:
    """Get the betweenness centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True.
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None.
    seed : int, optional
        The seed to use, by default 42.

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.betweenness_centrality(network, k=None, normalized=False, weight=weight, endpoints=False, seed=seed)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_closeness_centrality(network: nx.Graph, normalize: bool = True,
                                   weight: Optional[str] = None) -> Dict[str, float]:
    """Get the closeness centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.closeness_centrality(network, u=None, distance=weight)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_pagerank_centrality(network: nx.Graph, normalize: bool = True,
                                  weight: Optional[str] = None) -> Dict[str, float]:
    """Get the PageRank centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.pagerank(network, alpha=0.85, max_iter=100, tol=1e-06, nstart=None, weight=weight, dangling=None)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_hits_centrality(network: nx.Graph, normalize: bool = True,
                              weight: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get the HITS centrality (Hubs and Authorities) of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = weighted_hits(network, normalized=normalize, weight=weight)
    return metric_dict

def normalize_centrality_measures(centrality_dict: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """Function to normalize with min-max scale the centrality measures across a dictionary of dictionaries of metrics results

    Parameters
    ----------
    centrality_dict : { int : { str : float } }
        The dictionary of dictionaries of metrics.

    Returns
    -------
    { int : { str : float } }
        The normalized dictionary
    """
    centrality_dict = deepcopy(centrality_dict)
    
    values_list = [v for centralities in centrality_dict.values() for v in centralities.values()]
    min_value = min(values_list)
    max_value = max(values_list)
    
    for centralities in centrality_dict.values():
        for k, v in centralities.items():
            centralities[k] = (v - min_value) / (max_value - min_value)

    return centrality_dict

def get_girvan_newman_communities(network: nx.Graph, weight: Optional[str] = None, k: int = 2,
                                  seed: int = 42) -> Dict[str, int]:
    """Function to get the Girvan-Newmann partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.
    k : int, optional
        How many communities to find, by default 2
    seed : int, optional
        The seed to use, by default 42

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    def most_central_edge(network: nx.Graph):
        """
        Finds the most central edge in the given network based on edge betweenness centrality.

        Parameters:
            network (nx.Graph): The network for which to calculate the most central edge.

        Returns:
            tuple: The most central edge in the network.

        """
        centrality = nx.edge_betweenness_centrality(network, weight=weight, seed=seed)
        return max(centrality, key=centrality.get)

    communities_iterator = nx.community.girvan_newman(network, most_valuable_edge=most_central_edge)
    
    for _ in range(k - 1):
        communities = next(communities_iterator)
    
    return { c: i for i, community in enumerate(communities) for c in community }

def get_greedy_modularity_communities(network: nx.Graph, weight: Optional[str] = None) -> Dict[str, int]:
    """Function to get the Greedy Modularity partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    communities = nx.community.greedy_modularity_communities(network, weight=weight)
    return { c: i for i, community in enumerate(communities) for c in community }

def get_modularity_score(network: nx.Graph, node_community_dict: Dict[str, int], weight: str) -> float:
    """Function to get the modularity score of a network partition.

    Parameters
    ----------
    network : Graph
        The network partition
    node_community_dict : { str: int }
        Dictionary describing for each node (key) the relative partition (value).
    weight : str
        The edge weight to use to compute the odularity score.

    Returns
    -------
    float
        The modularity score.
    """
    communities_labels = set(node_community_dict.values())
    communities = {l: [] for l in communities_labels}
    for k, v in node_community_dict.items():
        communities[v].append(k)
    return nx.community.modularity(network, communities.values(), weight=weight)