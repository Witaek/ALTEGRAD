"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
Gs = [nx.cycle_graph(i) for i in range(10,20)]


############## Task 5
        
adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs]) 
x = np.ones((adj.shape[0], 1))

idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
x = torch.FloatTensor(x).to(device)
idx = torch.LongTensor(idx).to(device)



############## Task 8
        
model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
print("\n======== aggr = mean, readout = mean ========\n")
print(model(x, adj, idx))

print("\n======== aggr = mean, readout = sum ========\n")
model = GNN(1, hidden_dim, output_dim, 'sum', readout, dropout).to(device)
print(model(x, adj, idx))

print("\n======== aggr = sum, readout = mean ========\n")
model = GNN(1, hidden_dim, output_dim, neighbor_aggr, 'sum', dropout).to(device)
print(model(x, adj, idx))

print("\n======== aggr = sum, readout = sum ========\n")
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
print(model(x, adj, idx))


############## Task 9

G1 =  nx.Graph()
G1.add_nodes_from([0,1,2,3,4,5])
G1.add_edges_from([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])      
G2 = nx.cycle_graph(6)



############## Task 10

adj = sp.block_diag(nx.adjacency_matrix(G) for G in [G1,G2])

x = np.ones((adj.shape[0], 1))

idx = []
for i,G in enumerate([G1,G2]):
    idx += [i]*G.number_of_nodes()

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
x = torch.FloatTensor(x).to(device)
idx = torch.LongTensor(idx).to(device)

############## Task 11
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)

print("\n======== aggr = sum, readout = sum  for G1 & G2 ========\n")
print(model(x, adj, idx))