
import torch.nn as nn
from torch_geometric.nn import GATv2Conv,TransformerConv,RGATConv
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
from typing import List, NamedTuple, Optional, Union

import typing
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import LSTM, Linear, Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from torch import Tensor
from torch_geometric.nn.conv.gatv2_conv import *

from torch_geometric import EdgeIndex
from torch_geometric.index import ptr2index




if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload
    
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils import is_sparse
from torch_geometric.typing import Size, SparseTensor
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
    
    
class GATModel_TRANS(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_heads=8,n_classes=460,centroid_layer=False):
        super(GATModel_TRANS, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        self.centroid_layer=centroid_layer
        # self.gat_conv_1 = GATv2Conv(hidden_dim*int(num_heads/2), hidden_dim*int(num_heads/2), heads=int(num_heads/2), concat=False,residual=True)
        self.gat_conv = TransformerConv(input_dim, hidden_dim, heads=int(num_heads/2), concat=True)
        # self.gat_conv = TransformerConv(input_dim, hidden_dim, heads=int(num_heads/2), concat=True)
        # self.gat_conv_add = TransformerConv(hidden_dim*int(num_heads/2), hidden_dim, heads=int(num_heads), concat=True)
        self.gat_conv_0 = GATv2Conv(hidden_dim*int(num_heads/2), n_classes, heads=num_heads, concat=False)
        # self.gat_conv_0 = GATv2Conv(hidden_dim, n_classes, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        
        dest_nodes = edge_index[:, 1]

        # Sort indices based on destination nodes
        sorted_indices = torch.argsort(dest_nodes)

        # Reorder edge_index_t using the sorted indices
        edge_index = edge_index[sorted_indices]
        h_c=None
       
        
        x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
      

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h = self.gat_conv(x, edge_index.T)
            # h = self.gat_conv(x, emb_data.edge_index_testing.T)
        x=None
        del x
        # h = self.gat_conv_1(self.activate(h),edge_index.T)
        h= self.gat_conv_0(self.activate(h),edge_index.T)
        # h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index,edge_index


class HyperbolicGATv2Conv(GATv2Conv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True,
                 edge_dim=None, fill_value="mean", bias=True,
                 curvature=0.1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
        )
        self.curvature = torch.tensor(curvature, dtype=torch.float32)
    class CollectArgs(NamedTuple):
        x_i: Tensor
        x_j: Tensor
        edge_attr: Tensor
        index: Tensor

    def collect(
    self,
    edge_index: Union[Tensor, SparseTensor],
    x: PairTensor,
    alpha: Tensor,
    size: List[Optional[int]],
) -> CollectArgs:

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        # Collect special arguments:
        if isinstance(edge_index, Tensor):
            if is_torch_sparse_tensor(edge_index):
                adj_t = edge_index
                if adj_t.layout == torch.sparse_coo:
                    edge_index_i = adj_t.indices()[0]
                    edge_index_j = adj_t.indices()[1]
                    ptr = None
                elif adj_t.layout == torch.sparse_csr:
                    ptr = adj_t.crow_indices()
                    edge_index_j = adj_t.col_indices()
                    edge_index_i = ptr2index(ptr, output_size=edge_index_j.numel())
                else:
                    raise ValueError(f"Received invalid layout '{adj_t.layout}'")
                if edge_attr is None:
                    _value = adj_t.values()
                    edge_attr = None if _value.dim() == 1 else _value

            else:
                edge_index_i = edge_index[i]
                edge_index_j = edge_index[j]

                ptr = None
                if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
                    if i == 0 and edge_index.is_sorted_by_row:
                        (ptr, _), _ = edge_index.get_csr()
                    elif i == 1 and edge_index.is_sorted_by_col:
                        (ptr, _), _ = edge_index.get_csc()

        elif isinstance(edge_index, SparseTensor):
            adj_t = edge_index
            edge_index_i, edge_index_j, _value = adj_t.coo()
            ptr, _, _ = adj_t.csr()
            if edge_attr is None:
                edge_attr = None if _value is None or _value.dim() == 1 else _value

        else:
            raise NotImplementedError
        if torch.jit.is_scripting():
            assert edge_attr is not None

        # Collect user-defined arguments:
        # (1) - Collect `x_i`:
        if isinstance(x, (tuple, list)):
            assert len(x) == 2
            _x_0, _x_1 = x[0], x[1]
            if isinstance(_x_0, Tensor):
                self._set_size(size, 0, _x_0)
            if isinstance(_x_1, Tensor):
                self._set_size(size, 1, _x_1)
                x_i = self._index_select(_x_1, edge_index_i)
            else:
                x_i = None
        elif isinstance(x, Tensor):
            self._set_size(size, i, x)
            x_i = self._index_select(x, edge_index_i)
        else:
            x_i = None
        # (2) - Collect `x_j`:
        if isinstance(x, (tuple, list)):
            assert len(x) == 2
            _x_0, _x_1 = x[0], x[1]
            if isinstance(_x_0, Tensor):
                self._set_size(size, 0, _x_0)
                x_j = self._index_select(_x_0, edge_index_j)
            else:
                x_j = None
            if isinstance(_x_1, Tensor):
                self._set_size(size, 1, _x_1)
        elif isinstance(x, Tensor):
            self._set_size(size, j, x)
            x_j = self._index_select(x, edge_index_j)
        else:
            x_j = None

        # Collect default arguments:

        index = edge_index_i
        size_i = size[i] if size[i] is not None else size[j]
        size_j = size[j] if size[j] is not None else size[i]
        dim_size = size_i

        return self.CollectArgs(
            x_i,
            x_j,
            None,
            index,
        )
    def mobius_matvec(self, M: Tensor, p: Tensor) -> Tensor:
        """
        Möbius matrix-vector multiplication in hyperbolic space.

        Args:
            M (Tensor): Weight matrix (Euclidean space).
            p (Tensor): Vector in hyperbolic space.

        Returns:
            Tensor: Transformed vector in hyperbolic space.
        """
        curvature = self.curvature

        # Compute the norm of the input vector in Euclidean space
        norm_p = torch.norm(p, dim=-1, keepdim=True)

        # Perform Euclidean matrix-vector multiplication
        Mp = torch.matmul(p, M.T).view(-1, self.heads, self.out_channels)
        norm_Mp = torch.norm(Mp, dim=-1, keepdim=True)
        norm_p = norm_p.repeat(1, norm_Mp.shape[1] ).unsqueeze(-1)

        # Apply Möbius matrix-vector multiplication formula
        # print(Mp.shape,norm_Mp.shape) 
        
        scaled_norm = norm_Mp / (norm_p + 1e-8) * torch.atanh(torch.sqrt(curvature) * norm_p)
        # print('aaaaaaaaaaaaaa',torch.isnan(scaled_norm).any())
        return torch.tanh(scaled_norm)* Mp / (torch.sqrt(curvature)*norm_Mp+ 1e-8)

    def exponential_map(self, x):
        """Projects Euclidean features into hyperbolic space."""
        # curvature = torch.tensor(self.curvature, dtype=x.dtype, device=x.device)
        curvature= self.curvature
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.tanh(torch.sqrt(curvature) * norm) / (norm + 1e-8)
        return scale * x

    def logarithmic_map(self, x, base_point=None):
        """
        Projects hyperbolic features back to Euclidean tangent space.

        Args:
            x (Tensor): Point in hyperbolic space.
            base_point (Tensor, optional): Base point for the logarithmic map.
                                         Defaults to the origin.

        Returns:
            Tensor: Point in Euclidean tangent space.
        """
        curvature = self.curvature

        if base_point is None:
            # Logarithmic map at the origin
            norm = torch.norm(x, dim=-1, keepdim=True)
            scale = torch.atanh(torch.sqrt(curvature) * norm) / (norm + 1e-8)
            return scale * x

        else:
            # Logarithmic map at a base point
            mobius_diff = self.mobius_add(-base_point, x)
            norm = torch.norm(mobius_diff, dim=-1, keepdim=True)
            lambda_c = 2 / (1 - curvature * torch.sum(base_point**2, dim=-1, keepdim=True))
            scale = (2 / (torch.sqrt(curvature) * lambda_c)) * torch.atanh(torch.sqrt(curvature) * norm) / (norm + 1e-8)
            return scale * mobius_diff

    def hyperbolic_distance(self, x_i, x_j):
        """Computes hyperbolic distance between pairs of02822164216 points."""
        curvature = self.curvature
        mobius_diff = self.mobius_add(-x_i, x_j)
        # print(mobius_diff)
        norm = torch.norm(mobius_diff, dim=-1, keepdim=True)
        # print(norm)
        contains_nan = torch.isnan(x_j).any()
        contains_nan = torch.isnan(x_i).any()
        # print(" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",contains_nan)
        return 2.0 / torch.sqrt(curvature) * torch.atanh(torch.sqrt(curvature) * norm)

    def mobius_add(self, x, y):
        """Möbius addition in hyperbolic space."""
        curvature = self.curvature
        xy = (2 * torch.sum(x * y, dim=-1, keepdim=True))
        xx = torch.sum(x * x, dim=-1, keepdim=True)
        yy = torch.sum(y * y, dim=-1, keepdim=True)
        denominator = 1 + curvature * xy + curvature * yy * xx
        return (1 + curvature * xy) * x + (1 - curvature * xx) * y / denominator

    def mobius_scalar_mul(self, r, x):
        """Möbius scalar multiplication in hyperbolic space."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        curvature = self.curvature.to(x.device).to(x.dtype)
        return torch.tanh(r * torch.atanh(torch.sqrt(curvature) * norm)) / (torch.sqrt(curvature) * norm + 1e-8) * x

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # x = self.mobius_add(x_i, x_j)
        # print(x_i,x_j)
        x = -self.hyperbolic_distance(x_i, x_j)  # Negative for attention scores
        # print(x)
        # alpha = softmax(dist, index, ptr, dim_size)
        # x = F.leaky_relu(x, self.negative_slope)
        
        # alpha = (x * self.att).sum(dim=-1)
        # print(alpha)
        alpha = softmax(x, index, ptr, dim_size)
        # print(alpha)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha=alpha.squeeze_(-1)
        return alpha
    def message(self,x_i: Tensor, x_j: Tensor, alpha: Tensor,edge_attr :OptTensor) -> Tensor:
        """Compute messages using hyperbolic distance and Möbius operations."""
        # Compute attention scores
        # dist = -self.hyperbolic_distance(x_i, x_j)  # Negative for attention scores
        # alpha = softmax(dist, index=None)
        
        # Scale and aggregate using Möbius operations
        # dist = -self.hyperbolic_distance(x_i, x_j)
        alpha=alpha.unsqueeze(-1)
        # print(x_i.shape,x_j.shape,alpha.shape)
        
        weighted_x_j = self.mobius_scalar_mul(alpha, x_j)
        return weighted_x_j

   
    def propagate(
    self,
    edge_index: Union[Tensor, SparseTensor],
    x: PairTensor,
    alpha: Tensor,
    size: Size = None,
) -> Tensor:

    # Begin Propagate Forward Pre Hook #########################################
        torch.autograd.set_detect_anomaly(True)
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._propagate_forward_pre_hooks.values():
                hook_kwargs = dict(
                    x=x,
                    alpha=alpha,
                )
                res = hook(self, (edge_index, size, hook_kwargs))
                if res is not None:
                    edge_index, size, hook_kwargs = res
                    x = hook_kwargs['x']
                    alpha = hook_kwargs['alpha']
        # End Propagate Forward Pre Hook ###########################################

        mutable_size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        fuse = False
        if self.fuse:
            if is_sparse(edge_index):
                fuse = True
            elif not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
                if self.SUPPORTS_FUSED_EDGE_INDEX and edge_index.is_sorted_by_col:
                    fuse = True

        if fuse:
            raise NotImplementedError("'message_and_aggregate' not implemented")

        else:

            kwargs = self.collect(
                edge_index=edge_index,
                x=x,
                alpha=alpha,
                size=mutable_size,
            )

            # Begin Message Forward Pre Hook #######################################
            if not torch.jit.is_scripting() and not is_compiling():
                for hook in self._message_forward_pre_hooks.values():
                    hook_kwargs = dict(
                        x_i=kwargs.x_i,
                        x_j=kwargs.x_j,
                        edge_attr=kwargs.edge_attr,
                    )
                    res = hook(self, (hook_kwargs, ))
                    hook_kwargs = res[0] if isinstance(res, tuple) else res
                    if res is not None:
                        kwargs = self.CollectArgs(
                            x_i=hook_kwargs['x_i'],
                            x_j=hook_kwargs['x_j'],
                            edge_attr=hook_kwargs['edge_attr'],
                            index=kwargs.index,
                        )
            # End Message Forward Pre Hook #########################################

            out = self.message(
                x_i=kwargs.x_i,
                x_j=kwargs.x_j,
                alpha=alpha,
                edge_attr=kwargs.edge_attr,
            )

            # Begin Message Forward Hook ###########################################
            if not torch.jit.is_scripting() and not is_compiling():
                for hook in self._message_forward_hooks.values():
                    hook_kwargs = dict(
                        x_i=kwargs.x_i,
                        x_j=kwargs.x_j,
                        edge_attr=kwargs.edge_attr,
                    )
                    res = hook(self, (hook_kwargs, ), out)
                    out = res if res is not None else out
            # End Message Forward Hook #############################################

            # Begin Aggregate Forward Pre Hook #####################################
            if not torch.jit.is_scripting() and not is_compiling():
                for hook in self._aggregate_forward_pre_hooks.values():
                    hook_kwargs = dict(
                        index=kwargs.index,
                    )
                    res = hook(self, (hook_kwargs, ))
                    hook_kwargs = res[0] if isinstance(res, tuple) else res
                    if res is not None:
                        kwargs = self.CollectArgs(
                            x_i=kwargs.x_i,
                            x_j=kwargs.x_j,
                            edge_attr=kwargs.edge_attr,
                            index=hook_kwargs['index'],
                        )
            # End Aggregate Forward Pre Hook #######################################
            out = self.logarithmic_map(out)
            out = self.aggregate(
                out,
                index=kwargs.index,
                dim_size=mutable_size[1],
            )

            # Begin Aggregate Forward Hook #########################################
            if not torch.jit.is_scripting() and not is_compiling():
                for hook in self._aggregate_forward_hooks.values():
                    hook_kwargs = dict(
                        index=kwargs.index,
                    )
                    res = hook(self, (hook_kwargs, ), out)
                    out = res if res is not None else out
            # End Aggregate Forward Hook ###########################################

            out = self.update(
                out,
            )

        # Begin Propagate Forward Hook ############################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._propagate_forward_hooks.values():
                hook_kwargs = dict(
                    x=x,
                    alpha=alpha,
                )
                res = hook(self, (edge_index, mutable_size, hook_kwargs), out)
                out = res if res is not None else out
        # End Propagate Forward Hook ##############################################

        return out
    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: int = None) -> Tensor:
        """Aggregate messages using Möbius addition in hyperbolic space."""
        # Initialize the output tensor for aggregation
        output = torch.zeros(( dim_size, *inputs.shape[1:]), device=inputs.device, dtype=inputs.dtype)

        # Perform Möbius addition for aggregation
        for i in range(inputs.size(0)):
            output[index[i]] = self.mobius_add(output[index[i]], inputs[i])

        return output
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        # print(x.shape)
        x = self.exponential_map(x)
        # print('aaaaaaaaaaaaaa',torch.isnan(x).any())
        # print(x.shape)
        # print(x)
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)
            # x_l=self.lin_l(x).view(-1, H, C)
            # print(x_l.shape)
            x_l = self.mobius_matvec(self.lin_l.weight, x)
            # print(x_l.shape)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.mobius_matvec(self.lin_r.weight, x)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.mobius_matvec(self.lin_l.weight, x)
            if x_r is not None:
                x_r = self.mobius_matvec(self.lin_r.weight, x)

        assert x_l is not None
        assert x_r is not None
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        # print(x_l,x_r)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r),
                                  edge_attr=edge_attr)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        # print(alpha)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias
        
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out




class GATModel_test(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_heads=8,n_classes=460,centroid_layer=False):
        super(GATModel_test, self).__init__()

        # MLP for flattening emb_cells_in_cluster
       
        # GATConv for graph processing
        num_relation=3
        self.centroid_layer=centroid_layer
        self.gat_conv_1 = TransformerConv(input_dim, input_dim, heads=int(num_heads/2), concat=False)
        # self.gat_conv_1 = GATv2Conv(input_dim, input_dim, heads=int(num_heads/2), concat=False, residual=True)
        # self.gat_conv = RGATConv(input_dim, hidden_dim, heads=int(num_heads/2),num_relations =num_relation, concat=True)
        self.gat_conv = TransformerConv(input_dim, hidden_dim, heads=int(num_heads/2), concat=True)
        # self.gat_conv = GATv2Conv(input_dim, hidden_dim, heads=int(num_heads/2), concat=True, residual=True)    
        # self.gat_conv_add = TransformerConv(hidden_dim*int(num_heads/2), hidden_dim, heads=int(num_heads), concat=True)
        self.gat_conv_0 = GATv2Conv(hidden_dim*int(num_heads/2), n_classes, heads=num_heads, concat=False)
        # self.gat_conv_00 = TransformerConv(n_classes, n_classes, heads=num_heads, concat=False)
        self.activate = F.elu
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, data, return_attention=False):
        # Node features and edge indices from DataLoader
        emb_data, exps, exps_c= data
        centroid_num = exps_c.shape[0]
        edge_index = emb_data.edge_index  # x: [total_nodes, feature_dim], edge_index: [2, num_edges]
        
        # dest_nodes = edge_index[:, 1]

        # # Sort indices based on destination nodes
        # sorted_indices = torch.argsort(dest_nodes)

        # # Reorder edge_index_t using the sorted indices
        # edge_index = edge_index[sorted_indices]
        h_c=None
       
        
        x=emb_data.x
        try:
            centroid_index=emb_data.centroid_index
        except:
            centroid_index=[]
      

        # Apply GATConv across the entire batch graph
        
        # h=self.activate(self.gat_conv(x_combined, edge_index.T))
        # edge_type= torch.full((len(edge_index),), 0).to('cuda:0')
        # edge_type= emb_data.edge_type
        if return_attention:
            h, (edge_indices, attention_scores) = self.gat_conv(x, edge_index.T, return_attention_weights=True)
        else:
            h_1= self.gat_conv_1(x, edge_index.T)
            h = self.gat_conv(x-h_1, edge_index.T)
            
            # h = self.gat_conv(x, edge_index.T, edge_type)
            # h = self.gat_conv(x, emb_data.edge_index_testing.T)
        x=None
        del x
        # print(h)
        # h = self.gat_conv_1(self.activate(h),edge_index.T)
        # h= self.gat_conv_0(self.activate(h),edge_index.T, edge_type)
        h= self.gat_conv_0(self.activate(h),edge_index.T)
        # h_0=self.gat_conv_00(self.activate(h),edge_index.T)
        # h = self.fc(h).squeeze(0)
        
        # print(h.shape,exps.shape)
        if return_attention:
            return h,exps,h_c,exps_c,centroid_index,edge_indices, attention_scores
        return h,exps,h_c,exps_c,centroid_index,edge_index,h_1
    
    

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