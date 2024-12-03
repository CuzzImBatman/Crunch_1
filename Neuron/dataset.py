import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
from scipy.spatial import distance
from torch_geometric.data import Data

import pickle
import os


def get_edge_index(dataframe, k=6):
    # """
    # Calculate edge indices for a k-neighbor graph based on distances.
    
    # Args:
    #     dataframe (pd.DataFrame): DataFrame with 'x' and 'y' columns.
    #     k (int): Number of neighbors (default is 6).
    
    # Returns:
    #     edge_index (list of tuples): List of edges as (source, target).
    # """
    # Extract coordinates
        coordinates = dataframe[['x', 'y']].to_numpy()  # Shape: (n, 2)
        
        # Calculate pairwise distances
        dist_matrix = distance.cdist(coordinates, coordinates, metric='euclidean')  # Shape: (n, n)
        
        # For each point, find indices of k nearest neighbors (excluding itself)
        n = len(coordinates)
        edge_index = []
        for i in range(n):
            # Get indices of sorted distances (excluding itself)
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            
            # Create edges from i to each neighbor
            for neighbor in neighbors:
                edge_index.append((i, neighbor))  # (source, target)
                edge_index.append((neighbor, i))
            edge_index.append((-1,i))
        # edge_index= np.array(edge_index)
        
        # edge_index+=1
        return edge_index

class NeuronData(Dataset):
    def __init__(self, emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed', augmentation=True, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']):
        self.augmentation = augmentation
        emb_dir=emb_folder
        # emb_dir=  
        
        NAMES = name_list
        group='validation'
        dataset_type= None
        if train== False and split == True:
                dataset_type= 0 
        elif train== True and split == True:
                dataset_type= 1
        elif train== True and split == False:
                dataset_type= -1
        if split ==True:
            group='train'
            try:
                NAMES.remove('DC1')
            except:
                print('DC1 is not in list')
        # NAMES= NAMES[:1]
        print(f'Type: {dataset_type}')
        centroids_dict={}
        valid_clusters_dict={}
        valid_cell_list_cluster_dict={}
        emb_cells_dict={}
        emb_centroids_dict={}
        lenngths=[]
        for name in NAMES:
            preload_dir=f'../pre_load'
            with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
                cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'./cluster/train/cluster_data_split'
            elif train == False:
                cluster_dir=f'./cluster/validation'
                
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
            with open(f'{cluster_dir}/{name}_kmeans.pkl','rb') as f:
                kmeans = pickle.load(f)
            
            

            centroids = kmeans.cluster_centers_ # n x [x,y]
            # Filter out invalid clusters (those with 'train' = -1)
            
            valid_cell_list_cluster= cell_list_cluster[cell_list_cluster[group] != -1]
            valid_clusters = valid_cell_list_cluster['cluster'].unique()    
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]
            
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/20/{group}/{name}.npy'))
            emb_centroids= torch.from_numpy(np.load(f'{emb_dir}/256/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids
            # print(emb_cells.shape[0]/16)
            dim= int(emb_cells.shape[0]/16)
            
            emb_cells= emb_cells.view(dim,16,5,5)
            # print(emb_cells.shape, len( valid_cell_list_cluster),len(filtered_cells_index),len(cell_list_cluster))
            # len of emb_cells == len of cell_list_cluster
            emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
            # print(emb_cells.shape)
            # print(len(emb_cells), len( valid_cell_list_cluster))
            if dataset_type != None:
                if dataset_type ==1 :
                    temp_cluster            =cell_list_cluster[cell_list_cluster[group] == 1]['cluster'].unique()
                    cluster_index           =[list(valid_clusters).index(cluster_id) for cluster_id in temp_cluster ]
                    emb_centroids           =emb_centroids[cluster_index]
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 1]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 1]
                    valid_clusters          =temp_cluster
                elif dataset_type ==0:
                    temp_cluster            =cell_list_cluster[cell_list_cluster[group] == 0]['cluster'].unique()
                    cluster_index           =[list(valid_clusters).index(cluster_id) for cluster_id in temp_cluster ]
                    emb_centroids           =emb_centroids[cluster_index]
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 0]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 0]
                    valid_clusters          =temp_cluster
            
            centroids= centroids[valid_clusters]      
            lenngths.append(len(centroids))
            # len of centroids == len of valid_cluster == len of emb_centroids
            # len of valid_cell_list_cluster == len of  emb_cells
            centroids_dict[name]            =centroids
            valid_clusters_dict[name]       =valid_clusters
            valid_cell_list_cluster_dict[name]   =valid_cell_list_cluster
            emb_cells_dict[name]            =emb_cells
            emb_centroids_dict[name]        =emb_centroids
            
            
              
        self.centroids_dict             =centroids_dict
        self.valid_clusters_dict        =valid_clusters_dict
        self.emb_cells_dict             =emb_cells_dict
        self.emb_centroids_dict         =emb_centroids_dict
        self.lengths                    =lenngths
        self.valid_cell_list_cluster_dict    =valid_cell_list_cluster_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        
        valid_cell_list_cluster=None
        cell_list_org       =None
        cell_list_cluster   =None
        emb_cells           =None
        emb_centroids       =None
        print(self.cumlen[-1])
            
    def __len__(self):
        return self.cumlen[-1]
    
    
    def __getitem__(self, index):
        
        i = 0
        # print(index)
        item={}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        # len of centroids == len of valid_cluster == len of emb_centroids
        # len of valid_cell_list_cluster == len of  emb_cells
        
        centroid= self.centroids_dict[self.id2name[i]][idx]
        cluster_id= self.valid_clusters_dict[self.id2name[i]][idx]
        emb_cells= self.emb_cells_dict[self.id2name[i]]
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][idx]
        valid_cell_list_cluster= self.valid_cell_list_cluster_dict[self.id2name[i]]
        
        cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == cluster_id]
        emb_cells_in_cluster= emb_cells[valid_cell_list_cluster['cluster'].to_numpy() == cluster_id]
        
        x_center, y_center = centroid
        
        # Define square bounds
        half_side = int(256 / 2)
        x_min, x_max = x_center - half_side, x_center + half_side
        y_min, y_max = y_center - half_side, y_center + half_side
        
        cells_list_in_square = cells_list_in_cluster[
        (cells_list_in_cluster['x'] >= x_min) & (cells_list_in_cluster['x'] <= x_max) &
        (cells_list_in_cluster['y'] >= y_min) & (cells_list_in_cluster['y'] <= y_max)
    ]
        if len(cells_list_in_square)==0:
            cells_list_in_square=cells_list_in_cluster
        # print(centroid)
        centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
        # print( centroid_exps.shape,index)
        normalized_counts = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) * 100
        centroid_exps = np.log1p(normalized_counts)
        
        cell_counts= np.stack(cells_list_in_cluster['counts'].to_numpy())
        # print(cell_counts.shape)
        normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
        cell_exps = np.log1p(normalized_counts)
        
        
        cell_edge_index= get_edge_index(cells_list_in_cluster,k=6)
        # print(emb_centroid.shape, emb_cells_in_cluster.shape)
        
        # item['x']=emb_cells_in_cluster  # Node features
        # item['emb_centroid']=emb_centroid
        # # item['edge_index']=torch.tensor(cell_edge_index, dtype=torch.int).t().contiguous()  # Edge index
        # item['cluster_centroid']=torch.tensor(centroid, dtype=torch.float32)  # Centroid coordinates
        # item['cell_exps']=  cell_exps
        # item['centroid_exps']= centroid_exps
        # return item
        return  Data(
            x=emb_cells_in_cluster,  # Node features (including centroid and cells)
            edge_index=cell_edge_index,  # Edge indices (intra-cluster and centroid-to-cell)
            emb_centroid=emb_centroid,  # Embedding for centroid
            cluster_centroid=centroid, # Centroid coordinates
            cell_exps=cell_exps,
            centroid_exps=centroid_exps,
            cell_num= len(emb_cells_in_cluster)
        )
        # return emb_centroid, GE_centroid, centroid, emb_cells_in_cluster, cell_exps, cell_edge_index

       

    # @staticmethod
    # def partition(lst, n):
    #     division = len(lst) / float(n)
    #     return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
def build_batch_graph(batch,device):
    """Build a graph for a batch of 400 clusters."""
    all_x = []  # Store node features
    all_edge_index = []  # Store edges
    cluster_centroids = []  # Store centroids for all clusters
    centroid_exps=[]
    cell_exps=[]
    all_centroids_emb=[]
    node_offset = 0  # To adjust indices for each cluster
    # batch = batch.to(device)
    # print(len(batch.edge_index))
    # for edge_index in batch.edge_index:
    #     print(torch.tensor(edge_index).shape)
    #     all_edge_index.append(torch.tensor(edge_index))
    for i in range(len(batch.edge_index)):
        # print(torch.tensor(batch.edge_index[i]).shape)
        all_edge_index.append(torch.tensor(batch.edge_index[i]).to(device) + node_offset + len(batch.edge_index))
        # print(torch.tensor(edge_index).shape,len(edge_index))
        node_offset += batch.cell_num[i]
        # print(len(edge_index))
        # print(edge_index.size(0),batch.edge_index.size(0))
    edge_index = torch.cat(all_edge_index, dim=0)
    # print(edge_index.shape)
    all_x = batch.x  # All node features (batch size x feature_dim)

    # Assuming centroid embeddings are stored as a separate tensor
    emb_centroids = batch.emb_centroid  # Stack centroid embeddings from the batch
    cluster_centroids = torch.tensor(np.array(batch.cluster_centroid))  # Get centroid positions
    centroid_exps = np.vstack(batch.centroid_exps)  # Get centroid expression data
    cell_exps = np.vstack(batch.cell_exps)  # Get cell expression data
    # print(centroid_exps.shape, cell_exps.shape, all_x.shape)
    exps= np.vstack((centroid_exps, cell_exps))
    # You no longer need to adjust for node_offset since the batch is already stacked
    # Compute distances between cluster centroids for inter-cluster edges
    # print(cluster_centroids[0])
    dist_matrix = torch.cdist(cluster_centroids, cluster_centroids)
    # for  item in batch:
    #     # print(len(batch))
    #     # Store node features
    #     # all_x.append(item['x'])
    #     # all_centroids_emb.append(item['emb_centroid'])
    #     # # Adjust edge_index with offset and number of centroid
    #     # edge_index = item['edge_index'] + node_offset + len(batch)
    #     # all_edge_index.append(edge_index)
        
    #     # # Collect cluster centroid
    #     # cluster_centroids.append(item['cluster_centroid'])
        
    #     # # Update offset
    #     # node_offset += item['x'].size(0)
        
    #     # centroid_exps.append(item['centroid_exps'])
    #     # cell_exps.append(item['cell_exps'])
        
    #     all_x.append(item.x)
    #     all_centroids_emb.append(item.emb_centroid)
    #     # Adjust edge_index with offset and number of centroid
    #     edge_index = item.edge_index + node_offset + len(batch)
    #     all_edge_index.append(edge_index)
        
    #     # Collect cluster centroid
    #     cluster_centroids.append(item.x.cluster_centroid)
        
    #     # Update offset
    #     node_offset += item.x.size(0)
        
    #     centroid_exps.append(item.centroid_exps)
    #     cell_exps.append(item.cell_exps)
    
    # Combine node features and edge indices
    # x = torch.cat(all_x, dim=0)
    # centroids_emb = torch.stack(all_centroids_emb)
    # edge_index = torch.cat(all_edge_index, dim=1)
    # exps= centroid_exps+ cell_exps
    # # Compute distances between cluster centroids
    # cluster_centroids = torch.stack(cluster_centroids)
    # dist_matrix = torch.cdist(cluster_centroids, cluster_centroids)
    
    # Add edges for 6 nearest clusters
    for i in range(len(cluster_centroids)):
        neighbors = torch.topk(dist_matrix[i], k=6, largest=False).indices
        for neighbor in neighbors:
            # print(torch.tensor([[i], [neighbor]]).size())
            edge_index = torch.cat([edge_index, torch.tensor([[i, neighbor]]).to(device)], dim=0)
    # batch.cpu()
    return Data(x=all_x.to(device), edge_index=edge_index.to(device), emb_centroids= emb_centroids.to(device)),exps