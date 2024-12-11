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
from scipy.spatial.distance import cdist

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
    def __init__(self, emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed'
                 , augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']):
        self.augmentation = augmentation
        emb_dir=emb_folder
        # emb_dir=  
        self.encoder_mode= encoder_mode
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
        cluster_edge_index_dict={}
        cluster_index_cells_list_in_square_dict={}
        for name in NAMES:
            cluster_edge_index=[]
            cluster_index_cells_list_in_square=[]
            preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'../cluster/train/cluster_data_split'
            elif train == False:
                cluster_dir=f'../cluster/validation'
                
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
            for i in range(len(valid_clusters)):
                cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == valid_clusters[i]]
                edge_index=get_edge_index(cells_list_in_cluster,k=6)
                cluster_edge_index.append(edge_index)
                x_center, y_center = centroids[i]

                half_side = int(256 / 2)
                x_min, x_max = x_center - half_side, x_center + half_side
                y_min, y_max = y_center - half_side, y_center + half_side
                
                index_cells_list_in_square =(
                (cells_list_in_cluster['x'] >= x_min) & (cells_list_in_cluster['x'] <= x_max) &
                (cells_list_in_cluster['y'] >= y_min) & (cells_list_in_cluster['y'] <= y_max)
                )
                cluster_index_cells_list_in_square.append(index_cells_list_in_square)
            
            cluster_index_cells_list_in_square_dict[name]=cluster_index_cells_list_in_square
            cluster_edge_index_dict[name]= cluster_edge_index
            cluster_edge_index=None
            cluster_index_cells_list_in_square=None
                
            # cluster_exps_dict[name]= cluster_exps
            # valid_cell_exps_dict[name]= valid_cell_exps
            cluster_exps=None
            valid_cell_exps=None
        # self.valid_cell_exps = valid_cell_exps
        # self.cluster_exps_dict= cluster_exps_dict
        self.cluster_index_cells_list_in_square_dict=cluster_index_cells_list_in_square_dict
        self.cluster_edge_index_dict= cluster_edge_index_dict
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
        cluster_exps_dict   =None
        valid_cell_exps     =None
        # print(self.cumlen[-1])
        self.global_to_local = self._create_global_index_map()
    def __len__(self):
        return self.cumlen[-1]
    
    def _create_global_index_map(self):
        """Create a mapping from global index to (dataset index, local index)."""
        global_to_local = []
        for i, name in enumerate(self.id2name):
            local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
            global_to_local.extend(local_indices)
        return global_to_local
    
    def __getitem__(self, index):
        
        # i = 0
        # # print(index)
        # item={}
        # while index >= self.cumlen[i]:
        #     i += 1
        # idx = index
        # if i > 0:
        #     idx = index - self.cumlen[i - 1]
        # len of centroids == len of valid_cluster == len of emb_centroids
        # len of valid_cell_list_cluster == len of  emb_cells
        i, idx = self.global_to_local[index]
        centroid= self.centroids_dict[self.id2name[i]][idx]
        cluster_id= self.valid_clusters_dict[self.id2name[i]][idx]
        emb_cells= self.emb_cells_dict[self.id2name[i]]
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][idx]
        valid_cell_list_cluster= self.valid_cell_list_cluster_dict[self.id2name[i]]
        cluster_edge_index= self.cluster_edge_index_dict[self.id2name[i]][idx]
        cluster_index_cells_list_in_square= self.cluster_index_cells_list_in_square_dict[self.id2name[i]][idx]

        ######
        # centroid_exps = self.cluster_exps_dict[self.id2name[i]][idx]
        # cell_exps= self.valid_cell_exps[self.id2name[i]]
        ##########
        cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == cluster_id]
        emb_cells_in_cluster= emb_cells[valid_cell_list_cluster['cluster'].to_numpy() == cluster_id]
        
        x_center, y_center = centroid
        
        # Define square bounds
        ##############
        half_side = int(256 / 2)
        x_min, x_max = x_center - half_side, x_center + half_side
        y_min, y_max = y_center - half_side, y_center + half_side
        
        cells_list_in_square=[]
    #     cells_list_in_square = cells_list_in_cluster[
    #     (cells_list_in_cluster['x'] >= x_min) & (cells_list_in_cluster['x'] <= x_max) &
    #     (cells_list_in_cluster['y'] >= y_min) & (cells_list_in_cluster['y'] <= y_max)
    # ]
        # print(cluster_index_cells_list_in_square)
        cells_list_in_square=cells_list_in_cluster[cluster_index_cells_list_in_square]
        if len(cells_list_in_square)==0:
            cells_list_in_square=cells_list_in_cluster
        ###########################
            
        # print(centroid)
        
        ##########
        centroid_exps,cell_exps,cell_edge_index=None,None,None

        centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
        # print( centroid_exps.shape,index)
        normalized_counts = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) * 100
        centroid_exps = np.log1p(normalized_counts)
       
        cell_counts= np.stack(cells_list_in_cluster['counts'].to_numpy())
        # print(cell_counts.shape)
        normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
        cell_exps = np.log1p(normalized_counts)
        ############
        
        # cell_edge_index= get_edge_index(cells_list_in_cluster,k=6)
        cell_edge_index= cluster_edge_index
      
        # return item
        if self.encoder_mode==True:
            emb_cells_in_cluster=torch.empty(0)
            cell_exps=np.array([])
            cell_edge_index=[]
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
    
class NeuronData_2(Dataset):
    def __init__(self, emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed'
                 , augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']):
        self.augmentation = augmentation
        emb_dir=emb_folder
        # emb_dir=  
        self.encoder_mode= encoder_mode
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
        cluster_edge_index_dict={}
        cluster_index_cells_list_in_square_dict={}
        
        cluster_centroid_exps_dict={}
        cluster_cell_exps_dict={}
        cluster_cell_mask_dict={}
        for name in NAMES:
            print(name)
            cluster_edge_index=[]
            cluster_index_cells_list_in_square=[]
            cluster_centroid_exps=[]
            cluster_cell_exps=[]
            cluster_cell_mask=[]
            
            
            preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'../cluster/train/cluster_data_split'
            elif train == False:
                cluster_dir=f'../cluster/validation'
                
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
            # valid_cell_list_cluster_dict[name]   =valid_cell_list_cluster
            emb_cells_dict[name]            =emb_cells
            emb_centroids_dict[name]        =emb_centroids
            #####################################################
            
            for i in range(len(valid_clusters)):
                cell_mask=(valid_cell_list_cluster['cluster'].to_numpy() == valid_clusters[i])
                cluster_cell_mask.append(cell_mask)
                
                cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == valid_clusters[i]]
                edge_index=get_edge_index(cells_list_in_cluster,k=6)
                cluster_edge_index.append(edge_index)
                x_center, y_center = centroids[i]

                half_side = int(256 / 2)
                x_min, x_max = x_center - half_side, x_center + half_side
                y_min, y_max = y_center - half_side, y_center + half_side
                
                index_cells_list_in_square =(
                (cells_list_in_cluster['x'] >= x_min) & (cells_list_in_cluster['x'] <= x_max) &
                (cells_list_in_cluster['y'] >= y_min) & (cells_list_in_cluster['y'] <= y_max)
                )
                # cluster_index_cells_list_in_square.append(index_cells_list_in_square)
                
                
                ############
                cells_list_in_square=cells_list_in_cluster[index_cells_list_in_square]
                if len(cells_list_in_square)==0:
                    cells_list_in_square=cells_list_in_cluster
                centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
                normalized_counts = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) * 100
                centroid_exps = np.log1p(normalized_counts)
                
                cells_list_in_square=None
                del cells_list_in_square
                
                cell_exps= np.stack(cells_list_in_cluster['counts'].to_numpy())
                normalized_counts = cell_exps / cell_exps.sum(axis=1, keepdims=True) * 100
                cell_exps = np.log1p(normalized_counts)
                
                cells_list_in_cluster=None
                del cells_list_in_cluster
                #add to list
                cluster_centroid_exps.append(centroid_exps) 
                centroid_exps=None
                cluster_cell_exps.append(cell_exps)
                cell_exps=None
                del centroid_exps,cell_exps
             
            cluster_cell_mask_dict[name]=cluster_cell_mask    
            cluster_centroid_exps_dict[name]= cluster_centroid_exps
            cluster_cell_exps_dict[name]= cluster_cell_exps
            
            cluster_centroid_exps=None
            cluster_cell_exps=None
            valid_cell_list_cluster=None
            cluster_cell_mask=None
            del valid_cell_list_cluster,cluster_cell_mask
                
                ##############
                
            cluster_index_cells_list_in_square_dict[name]=cluster_index_cells_list_in_square
            cluster_edge_index_dict[name]= cluster_edge_index
            cluster_edge_index=None
            cluster_index_cells_list_in_square=None
                
            # cluster_exps_dict[name]= cluster_exps
            # valid_cell_exps_dict[name]= valid_cell_exps
        self.cluster_cell_mask_dict= cluster_cell_mask_dict
        self.cluster_cell_exps_dict = cluster_cell_exps_dict
        self.cluster_centroid_exps_dict= cluster_centroid_exps_dict
        # self.cluster_index_cells_list_in_square_dict=cluster_index_cells_list_in_square_dict
        self.cluster_edge_index_dict= cluster_edge_index_dict
        self.centroids_dict             =centroids_dict
        self.valid_clusters_dict        =valid_clusters_dict
        self.emb_cells_dict             =emb_cells_dict
        self.emb_centroids_dict         =emb_centroids_dict
        self.lengths                    =lenngths
        # self.valid_cell_list_cluster_dict    =valid_cell_list_cluster_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        
        cluster_cell_mask_dict=None
        cluster_cell_exps_dict=None
        cluster_centroid_exps_dict=None
        valid_cell_list_cluster=None
        cell_list_cluster   =None
        emb_cells           =None
        emb_centroids       =None
        # print(self.cumlen[-1])
        self.global_to_local = self._create_global_index_map()
    def __len__(self):
        return self.cumlen[-1]
    
    def _create_global_index_map(self):
        """Create a mapping from global index to (dataset index, local index)."""
        global_to_local = []
        for i, name in enumerate(self.id2name):
            local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
            global_to_local.extend(local_indices)
        return global_to_local
    
    def __getitem__(self, index):
        
     
        i, idx = self.global_to_local[index]
        centroid= self.centroids_dict[self.id2name[i]][idx]
        cluster_id= self.valid_clusters_dict[self.id2name[i]][idx]
        
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][idx]
        # valid_cell_list_cluster= self.valid_cell_list_cluster_dict[self.id2name[i]]
        cluster_edge_index= self.cluster_edge_index_dict[self.id2name[i]][idx]
        # cluster_index_cells_list_in_square= self.cluster_index_cells_list_in_square_dict[self.id2name[i]][idx]
        cell_exps=self.cluster_cell_exps_dict[self.id2name[i]][idx]
        centroid_exps= self.cluster_centroid_exps_dict[self.id2name[i]][idx]
        cluster_cell_mask= self.cluster_cell_mask_dict[self.id2name[i]][idx]
        emb_cells_in_cluster= self.emb_cells_dict[self.id2name[i]][cluster_cell_mask]
        ######
        # centroid_exps = self.cluster_exps_dict[self.id2name[i]][idx]
        # cell_exps= self.valid_cell_exps[self.id2name[i]]
        ##########
        
        # cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == cluster_id]
        # print(emb_cells.shape, len(cluster_cell_mask))
        # emb_cells_in_cluster= emb_cells[cluster_cell_mask]
        
        x_center, y_center = centroid
        
        # Define square bounds
        ##############
        # half_side = int(256 / 2)
        # x_min, x_max = x_center - half_side, x_center + half_side
        # y_min, y_max = y_center - half_side, y_center + half_side
        
        # cells_list_in_square=[]
   
        # print(cluster_index_cells_list_in_square)
        # cells_list_in_square=cells_list_in_cluster[cluster_index_cells_list_in_square]
        # if len(cells_list_in_square)==0:
        #     cells_list_in_square=cells_list_in_cluster
        ###########################
            
        # print(centroid)
        
        ##########
        # centroid_exps,cell_exps,cell_edge_index=None,None,None

        # centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
        # # print( centroid_exps.shape,index)
        # normalized_counts = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) * 100
        # centroid_exps = np.log1p(normalized_counts)
       
        # cell_counts= np.stack(cells_list_in_cluster['counts'].to_numpy())
        # # print(cell_counts.shape)
        # normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
        # cell_exps = np.log1p(normalized_counts)
        ############
        
        # cell_edge_index= get_edge_index(cells_list_in_cluster,k=6)
        cell_edge_index= cluster_edge_index
      
        # return item
        if self.encoder_mode==True:
            emb_cells_in_cluster=torch.empty(0)
            cell_exps=np.array([])
            cell_edge_index=[]
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
    #     return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]'D:/DATA/Gene_Expression/crunch/Register'
    
class NeuronData_3(Dataset):
    def __init__(self,cluster_path= 'E:/DATA/crunch/tmp', emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed'
                 , augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I'],evel=False):
        self.augmentation = augmentation
        emb_dir=emb_folder
        # emb_dir=  
        self.encoder_mode= encoder_mode
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
        print(f'Type: {dataset_type}, group: {group}')
        centroids_dict={}
        valid_clusters_dict={}
        valid_cell_list_cluster_dict={}
        emb_cells_dict={}
        emb_centroids_dict={}
        lenngths=[]
        cluster_edge_index_dict={}
        cluster_index_cells_list_in_square_dict={}
        
        cluster_centroid_exps_dict={}
        cluster_cell_exps_dict={}
        cluster_cell_mask_dict={}
        sorted_indices_dict={}
        for name in NAMES:
            print(name)
            cluster_edge_index=[]
            cluster_index_cells_list_in_square=[]
            cluster_centroid_exps=[]
            cluster_cell_exps=[]
            cluster_cell_mask=[]
            
            
            # preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'{cluster_path}/cluster/{group}/cluster_data_split'
            elif train == False:
                cluster_dir=f'{cluster_path}/cluster/{group}/cluster_data'
                
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
            # with open(f'{cluster_dir}/{name}_kmeans.pkl','rb') as f:
            #     kmeans = pickle.load(f)
            
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/24/{group}/{name}.npy'))
            emb_centroids= torch.from_numpy(np.load(f'{emb_dir}/80/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids

            # centroids = kmeans.cluster_centers_ # n x [x,y]
            centroids = cell_list_cluster.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
            # Filter out invalid clusters (those with 'train' = -1)
            distances = cdist(centroids, centroids)

            # Initialize variables
            
            if group =='train':
                valid_cell_list_cluster= cell_list_cluster[cell_list_cluster[group] != -1]
                # len of emb_cells == len of cell_list_cluster
                emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
            else:
                valid_cell_list_cluster= cell_list_cluster
                
            valid_clusters = valid_cell_list_cluster['cluster'].unique()    
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]
            
            
            
          
            
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
            ''''''
            n_centroids = len(centroids)
            visited = [False] * n_centroids
            sorted_indices = []
            current = 0  # Start from the first centroid

            for _ in range(n_centroids):
                # Mark current centroid as visited
                visited[current] = True
                sorted_indices.append(current)
                
                # Find the next closest unvisited centroid
                unvisited_distances = [(i, distances[current, i]) for i in range(n_centroids) if not visited[i]]
                if unvisited_distances:
                    current = min(unvisited_distances, key=lambda x: x[1])[0]

            # Use the sorted indices to get sorted centroids
            sorted_indices_dict[name]=sorted_indices
            ''''''
               
            lenngths.append(len(centroids))
            # len of centroids == len of valid_cluster == len of emb_centroids
            # len of valid_cell_list_cluster == len of  emb_cells
            centroids_dict[name]            =centroids
            valid_clusters_dict[name]       =valid_clusters
            # valid_cell_list_cluster_dict[name]   =valid_cell_list_cluster
            emb_cells_dict[name]            =emb_cells
            emb_centroids_dict[name]        =emb_centroids
            #####################################################
            
            for i in range(len(valid_clusters)):
                cell_mask=(valid_cell_list_cluster['cluster'].to_numpy() == valid_clusters[i])
                cluster_cell_mask.append(cell_mask)
                
                cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == valid_clusters[i]]
                edge_index=get_edge_index(cells_list_in_cluster,k=6)
                cluster_edge_index.append(edge_index)
                x_center, y_center = centroids[i,1],centroids[i,2]
                if group == 'train':
                    half_side = int(256 / 2)
                    x_min, x_max = x_center - half_side, x_center + half_side
                    y_min, y_max = y_center - half_side, y_center + half_side
                    
                    index_cells_list_in_square =(
                    (cells_list_in_cluster['x'] >= x_min) & (cells_list_in_cluster['x'] <= x_max) &
                    (cells_list_in_cluster['y'] >= y_min) & (cells_list_in_cluster['y'] <= y_max)
                    )
                    # cluster_index_cells_list_in_square.append(index_cells_list_in_square)
                    
                    
                    ############
                    cells_list_in_square=cells_list_in_cluster[index_cells_list_in_square]
                    if len(cells_list_in_square)==0:
                        cells_list_in_square=cells_list_in_cluster
                    centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
                    normalized_counts = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) * 100
                    centroid_exps = np.log1p(normalized_counts)
                    cells_list_in_square=None
                    del cells_list_in_square
                    
                    cell_exps= np.stack(cells_list_in_cluster['counts'].to_numpy())
                    normalized_counts = cell_exps / cell_exps.sum(axis=1, keepdims=True) * 100
                    cell_exps = np.log1p(normalized_counts)
                    
                    cells_list_in_cluster=None
                    del cells_list_in_cluster
                else:
                    centroid_exps=np.empty(460)
                    cell_exps=np.empty(460)
                    
                
                
                # cell_exps= np.stack(cells_list_in_cluster['counts'].to_numpy())
                # normalized_counts = cell_exps / cell_exps.sum(axis=1, keepdims=True) * 100
                # cell_exps = np.log1p(normalized_counts)
                
                # cells_list_in_cluster=None
                # del cells_list_in_cluster
                #add to list
                cluster_centroid_exps.append(centroid_exps) 
                centroid_exps=None
                cluster_cell_exps.append(cell_exps)
                cell_exps=None
                del centroid_exps,cell_exps
             
            cluster_cell_mask_dict[name]=cluster_cell_mask    
            cluster_centroid_exps_dict[name]= cluster_centroid_exps
            cluster_cell_exps_dict[name]= cluster_cell_exps
            
            cluster_centroid_exps=None
            cluster_cell_exps=None
            valid_cell_list_cluster=None
            cluster_cell_mask=None
            del valid_cell_list_cluster,cluster_cell_mask
                
                ##############
                
            cluster_index_cells_list_in_square_dict[name]=cluster_index_cells_list_in_square
            cluster_edge_index_dict[name]= cluster_edge_index
            cluster_edge_index=None
            cluster_index_cells_list_in_square=None
                
            # cluster_exps_dict[name]= cluster_exps
            # valid_cell_exps_dict[name]= valid_cell_exps
        self.cluster_cell_mask_dict= cluster_cell_mask_dict
        self.cluster_cell_exps_dict = cluster_cell_exps_dict
        self.cluster_centroid_exps_dict= cluster_centroid_exps_dict
        # self.cluster_index_cells_list_in_square_dict=cluster_index_cells_list_in_square_dict
        self.cluster_edge_index_dict= cluster_edge_index_dict
        self.centroids_dict             =centroids_dict
        self.valid_clusters_dict        =valid_clusters_dict
        self.emb_cells_dict             =emb_cells_dict
        self.emb_centroids_dict         =emb_centroids_dict
        self.lengths                    =lenngths
        self.sorted_indices_dict        =sorted_indices_dict
        # self.valid_cell_list_cluster_dict    =valid_cell_list_cluster_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        
        cluster_cell_mask_dict=None
        cluster_cell_exps_dict=None
        cluster_centroid_exps_dict=None
        valid_cell_list_cluster=None
        cell_list_cluster   =None
        emb_cells           =None
        emb_centroids       =None
        # print(self.cumlen[-1])
        self.global_to_local = self._create_global_index_map()
    def __len__(self):
        return self.cumlen[-1]
    
    def _create_global_index_map(self):
        """Create a mapping from global index to (dataset index, local index)."""
        global_to_local = []
        for i, name in enumerate(self.id2name):
            local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
            global_to_local.extend(local_indices)
        return global_to_local
    
    def __getitem__(self, index):
        
     
        i, idx = self.global_to_local[index]
        idx= self.sorted_indices_dict[self.id2name[i]][idx]
        centroid= self.centroids_dict[self.id2name[i]][idx]
        cluster_id= self.valid_clusters_dict[self.id2name[i]][idx]
        
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][idx]
        cluster_edge_index= self.cluster_edge_index_dict[self.id2name[i]][idx]
        cell_exps=self.cluster_cell_exps_dict[self.id2name[i]][idx]
        centroid_exps= self.cluster_centroid_exps_dict[self.id2name[i]][idx]
        cluster_cell_mask= self.cluster_cell_mask_dict[self.id2name[i]][idx]
        emb_cells_in_cluster= self.emb_cells_dict[self.id2name[i]][cluster_cell_mask]
      
    
        
        cell_edge_index= cluster_edge_index
      
        # return item
        if self.encoder_mode==True:
            emb_cells_in_cluster=torch.empty(0)
            cell_exps=np.array([])
            cell_edge_index=[]
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
 
def build_batch_graph(batch,device,centroid_layer):
    """Build a graph for a batch of 400 clusters."""
    all_x = []  # Store node features
    all_edge_index = []  # Store edges
    cluster_centroids = []  # Store centroids for all clusters
    centroid_exps=[]
    cell_exps=[]
    all_centroids_emb=[]
    node_offset = 0  # To adjust indices for each cluster
    # print(batch.x.shape)
    centroid_exps = np.vstack(batch.centroid_exps)  # Get centroid expression data
    # print(centroid_exps.shape)
    # print(batch.x.shape, batch.emb_centroid)
    if batch.x.shape[0]!=0:
        cell_exps = np.vstack(batch.cell_exps)  # Get cell expression data
        # print(centroid_exps.shape, cell_exps.shape)
        exps= np.vstack((centroid_exps, cell_exps))
        for i in range(len(batch.edge_index)):
            # print(torch.tensor(batch.edge_index[i]).shape)
            all_edge_index.append(torch.tensor(batch.edge_index[i]) + node_offset + len(batch.edge_index))
            # print(torch.tensor(edge_index).shape,len(edge_index))
            node_offset += batch.cell_num[i]
        edge_index = torch.cat(all_edge_index, dim=0)
        # emb_centroids=batch.emb_centroid
        batch.emb_centroid= batch.emb_centroid.view(-1,1024)
        all_x = torch.cat([batch.emb_centroid, batch.x], dim=0)
    else:
        exps=centroid_exps
        all_edge_index = []
        # print(len(edge_index))
        # print(edge_index.size(0),batch.edge_index.size(0))
        edge_index=torch.empty(0, dtype=torch.long)
        all_x = batch.emb_centroid
    # print(all_x.shape,centroid_exps.shape)
    # print(edge_index.shape)
    # all_x = batch.x  # All node features (batch size x feature_dim)
    

    # Assuming centroid embeddings are stored as a separate tensor
    # emb_centroids = batch.emb_centroid  # Stack centroid embeddings from the batch
    cluster_centroids = torch.tensor(np.array(batch.cluster_centroid))  # Get centroid positions
    
   
    dist_matrix = torch.cdist(cluster_centroids, cluster_centroids)
    
    ###################
    # # Add edges for 6 nearest clusters
    # edge_index_centroid=torch.empty(0, dtype=torch.long, device=device)
    # for i in range(len(cluster_centroids)):
    #     try:
    #         neighbors = torch.topk(dist_matrix[i], k=min(6,len(cluster_centroids)-1), largest=False).indices
    #     except:
    #         print(len(cluster_centroids))
    #         break
    #     for neighbor in neighbors:
    #         # print(torch.tensor([[i], [neighbor]]).size())
    #         edge_to_add = torch.tensor([[i, neighbor]], dtype=torch.long).to(device)
    #         edge_index = torch.cat([edge_index, edge_to_add], dim=0)
    #         if centroid_layer==True:
    #             edge_index_centroid = torch.cat([edge_index_centroid, edge_to_add], dim=0)
    # # print(f'shape:    {exps.shape}')
    
    ###############
    k=min(6,len(cluster_centroids))
    neighbors = torch.topk(dist_matrix, k=k, largest=False).indices  # Top 6 neighbors

    # Add inter-cluster edges
    edge_index_centroid = []
    for i in range(len(cluster_centroids)):
        add= torch.stack([torch.full((k,), i), neighbors[i]])
        edge_index_centroid.append(add)
    edge_index_centroid = torch.cat(edge_index_centroid, dim=1)
    edge_index_centroid= edge_index_centroid.T
    edge_index = torch.cat([edge_index, edge_index_centroid], dim=0)
    if centroid_layer ==False:
        edge_index_centroid =None
    
    ##############################
    # print(all_x.shape,all_x.dtype)
    return Data(x=all_x.to(device), edge_index=edge_index.to(device), edge_index_centroid=edge_index_centroid),exps,centroid_exps



# def build_batch_graph(batch, device, centroid_layer):
#     """Build a graph for a batch of 400 clusters."""
#     all_edge_index = []  # Store edges
#     node_offset = 0  # To adjust indices for each cluster

#     # Combine centroid and cell expression data
#     centroid_exps = np.vstack(batch.centroid_exps)  # Centroid expression data
#     exps = torch.tensor(centroid_exps, device=device, dtype=torch.float32)

#     if len(batch.cell_exps) != 0:
#         cell_exps = np.vstack(batch.cell_exps)  # Cell expression data
#         exps = torch.cat([exps, torch.tensor(cell_exps, device=device)], dim=0)

#         # Adjust edge indices for each cluster
#         for edge_index in batch.edge_index:
#             all_edge_index.append(torch.tensor(edge_index, device=device) + node_offset)
#             node_offset += edge_index.shape[1]

#         edge_index = torch.cat(all_edge_index, dim=1)
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

#     # Load node features and centroids
#     all_x = batch.x.to(device)  # Node features
#     cluster_centroids = torch.tensor(batch.cluster_centroid, device=device, dtype=torch.float32)
#     emb_centroids = batch.emb_centroid.to(device)

#     # Compute distances and find nearest neighbors for cluster centroids
#     dist_matrix = torch.cdist(cluster_centroids, cluster_centroids)  # Pairwise distances
#     neighbors = torch.topk(dist_matrix, k=6, largest=False).indices  # Top 6 neighbors

#     # Add inter-cluster edges
#     edge_index_centroid = []
#     for i in range(len(cluster_centroids)):
#         edge_index_centroid.append(torch.stack([torch.full((min(6,len(cluster_centroids)),), i, device=device), neighbors[i]]))

#     if centroid_layer:
#         edge_index_centroid = torch.cat(edge_index_centroid, dim=1)
#         edge_index = torch.cat([edge_index, edge_index_centroid], dim=1)

#     return Data(x=all_x, edge_index=edge_index, emb_centroids=emb_centroids, edge_index_centroid=edge_index_centroid), exps, centroid_exps