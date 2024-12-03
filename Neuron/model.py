
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F

class GATModel(nn.Module):
    def __init__(self, input_dim=16*5*5, hidden_dim=512, output_dim=1024, num_heads=4,n_classes=460):
        super(GATModel, self).__init__()

        # MLP for flattening emb_cells_in_cluster
        self.flatten_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # GATConv for graph processing
        self.gat_conv = GATv2Conv(output_dim, output_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(output_dim, n_classes)
    def forward(self, data):
        # Node features and edge indices from DataLoader
        emb_data, exps= data
        x, edge_index = emb_data.x, emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        emb_centroids = emb_data.emb_centroids  # Shape: [num_clusters, 1024], centroids for each cluster

        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        x_processed = self.flatten_mlp(x)  # Shape: [total_cells_in_batch, 1024]
        
        # Combine emb_centroids with processed cell features
        # print(x_processed.shape, emb_centroids.shape)
        emb_centroids= emb_centroids.view(-1,1024)
        x_combined = torch.cat([emb_centroids, x_processed], dim=0)  # Shape: [total_nodes, 1024]
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        h = self.gat_conv(x_combined, edge_index.T)
        h = self.fc(h).squeeze(0)
        # print(h.shape,exps.shape)
        return h,exps