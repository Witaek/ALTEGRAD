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
#Apply the Spectral Clustering algorithm to the giant connected component of the CA-HepTh dataset, trying to identify 50 clusters

G = nx.read_edgelist('CA-HepTh.txt', delimiter="\t", comments='#')
#retrieve largest connected component
largest_connected_component = max(nx.connected_components(G), key=len)

#subgraph of largest connected component
S = G.subgraph(largest_connected_component).copy()


clustering = spectral_clustering(S, 50)

cluster_sizes = {cluster_id: list(clustering.values()).count(cluster_id) for cluster_id in set(clustering.values())}


print(cluster_sizes)
##################





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    A = nx.adjacency_matrix(G).toarray()
    m = G.number_of_edges()

    modularity = 0.0

    # Create a mapping from node indices in the graph to matrix indices
    node_to_matrix_index = {node: i for i, node in enumerate(G.nodes())}

    for i in G.nodes():
        for j in G.nodes():
            ki = G.degree(i)
            kj = G.degree(j)
            Ci = clustering[i]  # Use the clustering directly
            Cj = clustering[j]

            # Use the mapping to get the matrix indices
            matrix_i = node_to_matrix_index[i]
            matrix_j = node_to_matrix_index[j]

            # Kronecker delta function
            delta = 1 if Ci == Cj else 0

            modularity += (A[matrix_i, matrix_j] - (ki * kj) / (2 * m)) * delta

    modularity /= (2 * m)

    return modularity


############## Task 9

##################
print("Modularity Spectral Clustering: ", modularity(S, clustering))

random_clustering = {node: randint(0, 50) for node in S.nodes()}
print("Modularity Random Clustering: ", modularity(S, random_clustering))
##################







