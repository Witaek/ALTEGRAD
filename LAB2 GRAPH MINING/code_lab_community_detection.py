"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    ##################
    A = nx.adjacency_matrix(G)

    degrees = dict(G.degree())
    nodes = list(G.nodes())
    diagonal_values = np.array([degrees[node] for node in nodes])
    D_inverse = diags(1.0 / diagonal_values)
    
    L = eye(A.shape[0]) - D_inverse@A
    
    #Eigen decomposition of L
    _,U = eigs(L, k=k, which='SR')

    U = U.real

    #K-means on rows of U
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    labels = kmeans.labels_
    clustering = dict(zip(G.nodes(), labels))
    ##################
    
    return clustering





############## Task 7

##################
#Apply the Spectral Clustering algorithm to the giant connected component
G = nx.read_edgelist('CA-HepTh.txt', delimiter="\t", comments='#')

#Retrieve largest connected component
largest_connected_component = max(nx.connected_components(G), key=len)

#Subgraph of largest connected component
S = G.subgraph(largest_connected_component).copy()

clustering = spectral_clustering(S, 50)
##################





############## Task 8

def modularity(G, clustering):
    ##################
    #number of clusters
    nc = len(set(clustering.values()))

    #number of edges
    m = G.number_of_edges()

    modularity = 0

    for cluster in range(nc):
        nodes_c = [node for node in G.nodes() if clustering[node] == cluster]
        subgraph = G.subgraph(nodes_c)
        lc = subgraph.number_of_edges()
        dc = sum([G.degree(node) for node in nodes_c])
        modularity += lc / m - (dc / (2 * m)) ** 2
    ##################
    return modularity


############## Task 9

##################
print("Modularity Spectral Clustering: ", modularity(S, clustering))

random_clustering = {node: randint(0, 50) for node in S.nodes()}
print("Modularity Random Clustering: ", modularity(S, random_clustering))
##################







