
import torch.nn as nn
from torch_geometric.nn import GATv2Conv,TransformerConv
import torch
import torch.nn.functional as F
####
from torch.nn import Parameter, LSTM
from torch_geometric.utils import softmax

from torchvision.models import DenseNet121_Weights
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x
class ThresholdedReLU(nn.Module):
    def __init__(self, theta=1.0,min_bound= 0.001):
        super(ThresholdedReLU, self).__init__()
        self.theta = theta

    def forward(self, x):
        return torch.where(x > self.theta, x, torch.full_like(x, 0.001))
class GATModel(nn.Module):
    def __init__(self, input_dim=16*5*5, hidden_dim=512, output_dim=1024, num_heads=4,n_classes=460,centroid_layer=False):
        super(GATModel, self).__init__()

        # MLP for flattening emb_cells_in_cluster
        self.flatten_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # GATConv for graph processing
        # self.centroid_layer=centroid_layer
        # self.gat_conv_centroid = GATv2Conv(output_dim, output_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(output_dim, output_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(output_dim, n_classes)
    def forward(self, data):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        x, edge_index = emb_data.x[centroid_num:], emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        emb_centroids = emb_data.x[:centroid_num],  # # Shape: [num_clusters, 1024], centroids for each cluster

        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
        try:
            if self.centroid_layer ==True:
                emb_centroids= self.gat_conv_centroid(emb_centroids,emb_data.edge_index_centroid.T)
                h_c= self.fc(emb_centroids).squeeze(0)
        except:
            tex='next'
        if x.numel() != 0:
            x = self.flatten_mlp(x)  # Shape: [total_cells_in_batch, 1024]
            
            # Combine emb_centroids with processed cell features
            # print(x_processed.shape, emb_centroids.shape)
            x = torch.cat([emb_centroids, x], dim=0)  # Shape: [total_nodes, 1024]
        else:
            x=emb_centroids
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        
        h = self.gat_conv(x, edge_index.T)
        h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        return h,exps,h_c,exps_c
    
    
class GATModel_Softmax(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(GATModel_Softmax, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        
        h = self.gat_conv(x, edge_index.T)
        h = self.fc(h).squeeze(0)
        h = F.softmax(h,dim=1)
        # print(h.shape,exps.shape)
        return h,exps,h_c,exps_c
    
    
class GATModel_3(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(GATModel_3, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index
    
class GATModel_thres(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(GATModel_thres, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.thresholded_relu = ThresholdedReLU(theta=0.24)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
      
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
      
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        
        h = self.fc(h)
        h=self.thresholded_relu(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index
class TransConv(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(TransConv, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.trans_conv_centroid = TransformerConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.trans_conv = TransformerConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        
        h = self.trans_conv_centroid(x, edge_index.T)
        h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        return h,exps,h_c,exps_c

class Encoder_GAT(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(Encoder_GAT, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.encoder= ImageEncoder()
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x.float()
            x= self.encoder(x)
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
      
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index

class GATModel_SAT(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=3,n_classes=460,centroid_layer=False):
        super(GATModel_SAT, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.activate = F.elu
        self.SAT= nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=int(hidden_dim/64),dropout=0.1)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        h = h.unsqueeze(1)  # Shape: (N, 1, d)
        h,_ = self.SAT(h,h,h)
        h = h.squeeze(1)  # Shape: (N, d)
        h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index

class GATModel_5(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_heads=8,n_classes=460,centroid_layer=False):
        super(GATModel_5, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        # self.gat_conv = GATv2Conv(input_dim, int(n_classes/num_heads), heads=num_heads, concat=True)
        # self.gat_conv_0 = GATv2Conv(n_classes*num_heads, n_classes, heads=num_heads, concat=False)
        self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv_0 = GATv2Conv(hidden_dim, n_classes, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        x=None
        del x
        h= self.gat_conv_0(self.activate(h),edge_index.T)
        # h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index

class GATModel_4(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_heads=12,n_classes=460,centroid_layer=False):
        super(GATModel_4, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
       
        self.gat_conv = GATv2Conv(input_dim, n_classes, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        x=None
        del x
        # h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index
from typing import Union, Tuple, Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import LSTM, Linear, Parameter

class OrderedGATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True,
                 bias=True, edge_dim=None, share_weights=False, residual=False):
        super().__init__(node_dim=0, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.residual = residual

        # Shared linear transformation
        self.W1 = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        if share_weights:
            self.W2 = self.W1
        else:
            self.W2 = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))

        # Self-attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        # LSTM for ordered sequence
        self.lstm = nn.LSTM(out_channels, out_channels, batch_first=True)

        if bias:
    # Adjust bias size based on whether concatenation is used
            total_out_channels = out_channels * heads if self.concat else out_channels
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        if not self.share_weights:
            nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        
        input_indices = torch.arange(x.size(0), device=x.device)
        if self.add_self_loops:
            edge_index, edge_attr = self.add_self_loops_if_needed(x.size(0), edge_index, edge_attr)

        x_l = torch.matmul(x, self.W1).view(-1, self.heads, self.out_channels)
        x_r = torch.matmul(x, self.W2).view(-1, self.heads, self.out_channels) if not self.share_weights else x_l

        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)

        if self.bias is not None:
            out += self.bias

        if self.concat:
            out= out.view(-1, self.heads * self.out_channels)
        else:
            out=  out.mean(dim=1)
        return out[input_indices]
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Compute attention scores
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((self.att * x).sum(dim=-1), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        # Sort neighbors by attention scores
        # perm = torch.argsort(alpha, descending=True, dim=1)
        # sorted_alpha = torch.gather(alpha, 1, perm)
        # sorted_x_j = torch.gather(x_j, 1, perm.unsqueeze(-1).expand(-1, -1, x_j.size(-1)))

        # # Normalize attention scores and construct sorted sequence
        # sorted_alpha = F.softmax(sorted_alpha, dim=1).unsqueeze(-1)
        alpha =F.softmax(alpha, dim=1).unsqueeze(-1)
        h_sorted = alpha * x_j

        # Pass sorted sequence through LSTM
        h_sorted, _ = self.lstm(h_sorted)

        return h_sorted

    def add_self_loops_if_needed(self, num_nodes, edge_index, edge_attr):
        if not self.add_self_loops:
            return edge_index, edge_attr

        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        edge_index = torch.cat([edge_index, loop_index], dim=1)
        if edge_attr is not None:
            loop_attr = torch.zeros(num_nodes, edge_attr.size(-1), device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        return edge_index, edge_attr
    
    
class GATModel_LSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_heads=2,n_classes=460,centroid_layer=False):
        super(GATModel_LSTM, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        self.gat_conv_centroid = OrderedGATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        # self.gat_conv = GATv2Conv(input_dim, int(n_classes/num_heads), heads=num_heads, concat=True)
        # self.gat_conv_0 = GATv2Conv(n_classes*num_heads, n_classes, heads=num_heads, concat=False)
        self.gat_conv = OrderedGATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat_conv_0 = GATv2Conv(hidden_dim, n_classes, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        # emb_centroids = emb_data.x[:centroid_num]  # Shape: [num_clusters, 1024], centroids for each cluster
        
        # Process emb_cells_in_cluster with flatten_mlp (this processes cell features)
        # emb_centroids= emb_centroids.view(-1,1024)
        h_c=None
       
        if self.centroid_layer ==True:
            emb_centroids= self.gat_conv_centroid(emb_data.x,emb_data.edge_index_centroid.T)
            h_c= self.fc(emb_centroids).squeeze(0)
            x= emb_centroids
        else:
            x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
        # print(get_size(x))
        # print(emb_centroids.shape,x_processed.shape,exps.shape)
        # print(x_combined.shape)
        # Adjust edge_index: The first 'num_clusters' nodes represent centroids.
        # edge_index = edge_index.clone()  # Copy the original edge_index to avoid modifying in place
        # edge_index[0] += emb_centroids.size(0)  # Shift the source indices for cells
        # edge_index[1] += emb_centroids.size(0)  # Shift the target indices for cells

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
        x=None
        del x
        h= self.gat_conv_0(self.activate(h),edge_index.T)
        # h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index

def get_size(x):
    print("Shape of tensor:", x.shape)

# Get the number of elements in the tensor
    num_elements = x.numel()
    # print(f"Number of elements: {num_elements}")

    # Get the size in bytes (assuming float32, which is 4 bytes per element)
    size_in_bytes = num_elements * 4  # 4 bytes for float32
    # print(f"Size of tensor in bytes: {size_in_bytes} bytes")

    # Optionally, print the size in MB (divide by 1024^2)
    size_in_MB = size_in_bytes / (1024 ** 2)
    return size_in_MB