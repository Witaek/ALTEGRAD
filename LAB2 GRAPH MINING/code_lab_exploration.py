"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################

G = nx.read_edgelist('CA-HepTh.txt', delimiter="\t", comments='#')
print(G)

############## Task 2

##################
# your code here #
##################

print(nx.number_connected_components(G))

largest = max(nx.connected_components(G), key=len)
S = G.subgraph(largest).copy()
print(S)



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################

print("max : ", np.max(degree_sequence))
print("min : ", np.min(degree_sequence))
print("mean : ", np.mean(degree_sequence))


############## Task 4

##################
# your code here #
##################

histogram = nx.degree_histogram(G)

deg = range(len(histogram))
plt.figure()
plt.bar(deg,histogram)
plt.show()

plt.loglog(deg, histogram, 'x')

plt.show()




############## Task 5

##################
# your code here #
##################

print(nx.transitivity(G))