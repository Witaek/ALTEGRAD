"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network with different colors for the two classes


nx.draw_networkx(G, node_color=y, cmap=plt.cm.Set1)
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy for Skipgram embeddings:", accuracy_score(y_test, y_pred))


############## Task 8
# Generates spectral embeddings

# Here I used my code from lab 2
A = nx.adjacency_matrix(G).toarray()
D = np.diag(np.sum(A, axis=1))
L = D - A
L_norm = np.dot(np.dot(np.diag(np.power(np.sum(A, axis=1), -0.5)), L), np.diag(np.power(np.sum(A, axis=1), -0.5)))

eigenvalues, eigenvectors = np.linalg.eig(L_norm)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the first two eigenvalues and eigenvectors
selected_eigenvalues = sorted_eigenvalues[:2]
selected_eigenvectors = sorted_eigenvectors[:, :2]

X_train_spectral = selected_eigenvectors[idx_train, :]
X_test_spectral = selected_eigenvectors[idx_test, :]

clf_spectral = LogisticRegression().fit(X_train_spectral, y_train)
y_pred_spectral = clf_spectral.predict(X_test_spectral)
print("Accuracy for Spectral embeddings:", accuracy_score(y_test, y_pred_spectral))





