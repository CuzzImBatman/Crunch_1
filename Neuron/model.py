
import torch.nn as nn
from torch_geometric.nn import GATv2Conv,TransformerConv
import torch
import torch.nn.functional as F
####
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
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024, num_heads=8,n_classes=460,centroid_layer=False):
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