"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
G = nx.read_edgelist('CA-HepTh.txt', delimiter="\t", comments='#')
print(G)
##################



############## Task 2

##################
print(nx.number_connected_components(G))
#retrieve largest connected component
largest_connected_component = max(nx.connected_components(G), key=len)

#subgraph of largest connected component
S = G.subgraph(largest_connected_component).copy()
print(S)
##################





############## Task 3

#compute degree of each node
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

#compute degree histogram
frequencies = nx.degree_histogram(G)

degrees = range(len(frequencies))

plt.figure()
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree distribution')

plt.bar(degrees,frequencies)
plt.show()

plt.loglog(degrees, frequencies, 'x')
plt.show()




############## Task 5

##################
# your code here #
##################

print("Global clustering coefficient = ",nx.transitivity(G))






