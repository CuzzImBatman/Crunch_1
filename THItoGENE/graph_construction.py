import numpy as np
import torch
from scipy.spatial import distance
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

# def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA', batch_size=10000):
#     # Convert coordinates to numpy array
#     spatialMatrix = np.array(coord)
#     nodes = len(spatialMatrix)
    
#     # Prepare lists to store sparse matrix data
#     row_indices = []
#     col_indices = []
#     values = []
    
#     # Use NearestNeighbors for k-NN search with batch processing
#     knn = NearestNeighbors(n_neighbors=k + 1, metric=distanceType).fit(spatialMatrix)

#     for start in range(0, nodes, batch_size):
#         end = min(start + batch_size, nodes)
        
#         # Get k+1 nearest neighbors (including itself at 0 distance)
#         distances, neighbors = knn.kneighbors(spatialMatrix[start:end], n_neighbors=k + 1)
        
#         # Process each batch
#         for i in range(distances.shape[0]):
#             node_index = start + i
#             tmpdist = distances[i, 1:]  # Exclude self (0 distance)
#             boundary = np.mean(tmpdist) + np.std(tmpdist)

#             for j in range(1, k + 1):  # Skip the first neighbor (self)
#                 neighbor_index = neighbors[i, j]
#                 dist_to_neighbor = distances[i, j]
                
#                 # Check if the neighbor should be added based on pruneTag
#                 if pruneTag == 'NA' or \
#                    (pruneTag == 'STD' and dist_to_neighbor <= boundary) or \
#                    (pruneTag == 'Grid' and dist_to_neighbor <= 2.0):
                    
#                     # Store only the non-zero entries
#                     row_indices.append(node_index)
#                     col_indices.append(neighbor_index)
#                     values.append(1.0)

#     # Create sparse adjacency matrix
#     indices = torch.LongTensor([row_indices, col_indices])
#     values = torch.FloatTensor(values)
#     Adj = torch.sparse.FloatTensor(indices, values, torch.Size([nodes, nodes]))

#     return Adj

# def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
#     spatialMatrix = np.array(coord)
#     nodes = spatialMatrix.shape[0]
#     Adj = torch.zeros((nodes, nodes))
#     for i in np.arange(spatialMatrix.shape[0]):
#         tmp = spatialMatrix[i, :].reshape(1, -1)
#         distMat = distance.cdist(tmp, spatialMatrix, distanceType)
#         if k == 0:
#             k = spatialMatrix.shape[0] - 1
#         res = distMat.argsort()[:k + 1]
#         tmpdist = distMat[0, res[0][1:k + 1]]
#         boundary = np.mean(tmpdist) + np.std(tmpdist)
#         for j in np.arange(1, k + 1):
#             # No prune
#             if pruneTag == 'NA':
#                 Adj[i][res[0][j]] = 1.0
#             elif pruneTag == 'STD':
#                 if distMat[0, res[0][j]] <= boundary:
#                     Adj[i][res[0][j]] = 1.0
#             elif pruneTag == 'Grid':
#                 if distMat[0, res[0][j]] <= 2.0:
#                     Adj[i][res[0][j]] = 1.0
#     return Adj

def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    spatialMatrix = np.array(coord)
    nodes = spatialMatrix.shape[0]
    
    # Create KDTree for fast k-NN search
    tree = KDTree(spatialMatrix)
    
    # Initialize sparse adjacency matrix (can use torch.sparse for memory efficiency)
    Adj = torch.zeros((nodes, nodes))
    
    for i in range(nodes):
        # Query the k-nearest neighbors (including the point itself)
        dist, indices = tree.query(spatialMatrix[i], k + 1)  # k+1 to include self
        
        # Exclude self (the first index is always the node itself)
        indices = indices[1:]
        dist = dist[1:]
        
        # Calculate pruning based on distance and pruneTag
        if pruneTag == 'NA':
            # No pruning, add all k neighbors
            Adj[i, indices] = 1.0
        elif pruneTag == 'STD':
            # Prune based on mean + std deviation
            boundary = np.mean(dist) + np.std(dist)
            Adj[i, indices[dist <= boundary]] = 1.0
        elif pruneTag == 'Grid':
            # Prune based on a fixed threshold (e.g., 2.0)
            Adj[i, indices[dist <= 2.0]] = 1.0
    
    return Adj
