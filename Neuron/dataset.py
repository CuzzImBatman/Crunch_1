import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
from scipy.spatial import distance
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import gc
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
            # edge_index.append((-1,i))
        # edge_index= np.array(edge_index)
        
        # edge_index+=1
        return edge_index


class NeuronData_3(Dataset):
    def __init__(self,cluster_path= 'E:/DATA/crunch/tmp/cluster', emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed'
                 , augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
                 ,evel=False
                 ,nolog1p= False
                 ,normalized=True
                 ,sorted_cluster=True):
        self.augmentation = augmentation
        emb_dir=emb_folder
        # emb_dir=  
        self.encoder_mode= encoder_mode
        NAMES = name_list
        group='evel'
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
        if evel==True:
            group= 'evel'
            dataset_type=-1
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
        cell_id_list_dict={}

        for name in NAMES:
            print(name)
            cluster_edge_index=[]
            cluster_index_cells_list_in_square=[]
            cluster_centroid_exps=[]
            cluster_cell_exps=[]
            cluster_cell_mask=[]
            cell_id_list=[]
            
            # preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'{cluster_path}/{group}/cluster_data_split'
            elif train == False:
                cluster_dir=f'{cluster_path}/{group}/cluster_data'
                
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
            # with open(f'{cluster_dir}/{name}_kmeans.pkl','rb') as f:
            #     kmeans = pickle.load(f)
            
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/24/{group}/{name}.npy'))
            emb_centroids= torch.from_numpy(np.load(f'{emb_dir}/80/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids

            # centroids = kmeans.cluster_centers_ # n x [x,y]
            centroids = cell_list_cluster.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
            # Filter out invalid clusters (those with 'train' = -1)
            
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
                    # emb_centroids           =emb_centroids[cluster_index]
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 1]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 1]
                    valid_clusters          =temp_cluster
                elif dataset_type ==0:
                    temp_cluster            =cell_list_cluster[cell_list_cluster[group] == 0]['cluster'].unique()
                    cluster_index           =[list(valid_clusters).index(cluster_id) for cluster_id in temp_cluster ]
                    # emb_centroids           =emb_centroids[cluster_index]
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 0]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 0]
                    valid_clusters          =temp_cluster
            cell_list_cluster=None
            centroids= centroids[valid_clusters]   
            emb_centroids= emb_centroids[valid_clusters]   
            print(len(valid_clusters),emb_centroids.shape)
            distances = cdist(centroids[:,1:], centroids[:,1:])

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
            if sorted_cluster== True:
                sorted_indices_dict[name]=sorted_indices
            else:
                sorted_indices_dict[name]= [i for i in range(n_centroids)]
            ''''''
               
            lenngths.append(len(centroids))
            # len of centroids == len of valid_cluster == len of emb_centroids
            # len of valid_cell_list_cluster == len of  emb_cells
            centroids_dict[name]            =centroids[:,1:]
            valid_clusters_dict[name]       =valid_clusters
            # valid_cell_list_cluster_dict[name]   =valid_cell_list_cluster
            emb_cells_dict[name]            =emb_cells
            emb_centroids_dict[name]        =emb_centroids
            #####################################################
            
            for i in range(len(valid_clusters)):
                # cell_mask=(valid_cell_list_cluster['cluster'].to_numpy() == valid_clusters[i])
                # cluster_cell_mask.append(cell_mask)
                cluster_cell_indices = np.where(valid_cell_list_cluster['cluster'].to_numpy() == valid_clusters[i])[0]
                cluster_cell_mask.append(cluster_cell_indices)
                
                cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == valid_clusters[i]]
                cell_ids= cells_list_in_cluster['cell_id'].to_numpy()
                cell_id_list.append(cell_ids)
                edge_index=get_edge_index(cells_list_in_cluster,k=6)
                cluster_edge_index.append(edge_index)
                x_center, y_center = centroids[i,1],centroids[i,2]
                cell_mask=None
                cell_ids=None
                edge_index=None
                if group == 'train':
                    half_side = int(80 / 2)
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
                    # cells_list_in_square=cells_list_in_cluster  ##TESTING
                    centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
                    if normalized==  True:
                        centroid_exps = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) 
                    if nolog1p == False:
                        centroid_exps = np.log1p(centroid_exps* 100)
                    
                    cells_list_in_square=None
                    del cells_list_in_square
                    
                    cell_exps= np.stack(cells_list_in_cluster['counts'].to_numpy())
                    if normalized==  True:
                        cell_exps = cell_exps / cell_exps.sum(axis=1, keepdims=True)
                    if nolog1p == False:
                        cell_exps = np.log1p(cell_exps * 100)
                    
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
                emb_cells=None
                emb_centroids=None
                del centroid_exps,cell_exps,emb_cells
             
            cell_id_list_dict[name]=cell_id_list
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
        self.cell_id_list_dict      = cell_id_list_dict 
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
        cell_ids= self.cell_id_list_dict[self.id2name[i]][idx]
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
            cell_num= len(emb_cells_in_cluster),
            cell_ids= cell_ids,
            slide_id= i
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
    cluster_centroids = torch.tensor(np.array(batch.cluster_centroid))  # Get centroid positions
    # print(cluster_centroids)
    dist_matrix = torch.cdist(cluster_centroids, cluster_centroids)
    k=min(6,len(cluster_centroids))
    neighbors = torch.topk(dist_matrix, k=k, largest=False).indices  # Top 6 neighbors

    if batch.x.shape[0]!=0:
        cell_exps = np.vstack(batch.cell_exps)  # Get cell expression data
        # print(centroid_exps.shape, cell_exps.shape)
        exps= np.vstack((centroid_exps, cell_exps))
        # print(len(batch.cell_num))
        for i in range(len(batch.edge_index)):
            # print(torch.tensor(batch.edge_index[i]).shape)        
            all_edge_index.append(torch.tensor(batch.edge_index[i]) + node_offset + len(batch.edge_index))
            add_edge_index=[]
            ''''''
            for j in range(batch.cell_num[i]):
                add_edge_index.append((i,j+ node_offset + len(batch.edge_index)))
                ####testing
                # for neighbor in neighbors[i]:
                #     add_edge_index.append((neighbor, j+ node_offset + len(batch.edge_index)))
            ''''''
            all_edge_index.append(torch.tensor(add_edge_index))
            node_offset += batch.cell_num[i]
        edge_index = torch.cat(all_edge_index, dim=0).to(torch.long)
        # print(edge_index)
        # emb_centroids=batch.emb_centroid
        # batch.emb_centroid= batch.emb_centroid.view(-1,1024)
        batch.emb_centroid= batch.emb_centroid.view(-1,batch.x.shape[1])
        all_x = torch.cat([batch.emb_centroid, batch.x], dim=0)
    else:
        exps=centroid_exps
        all_edge_index = []
        # print(len(edge_index))
        # print(edge_index.size(0),batch.edge_index.size(0))
        edge_index=torch.empty(0, dtype=torch.long)
        all_x = batch.emb_centroid
   
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
    # print(edge_index)
    ##############################
    # print(all_x.shape,all_x.dtype)
    return Data(x=all_x.to(device), edge_index=edge_index.to(device), edge_index_centroid=edge_index_centroid),exps,centroid_exps
class SuperNeuronData(Dataset):
    def __init__(self,cluster_path= 'E:/DATA/crunch/tmp/cluster', emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed', augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I'],evel=False,nolog1p= False):
        self.augmentation = augmentation
        
        # emb_dir=  
        self.encoder_mode= encoder_mode
        group='evel'
        NAMES = name_list
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
        if evel==True:
            group= 'evel'
            dataset_type=-1
        print(f'Type: {dataset_type}, group: {group}')
        super_centroids_dict={}
        valid_super_clusters_dict={}
        emb_cells_dict={}
        emb_centroids_dict={}
        emb_super_centroids_dict={}
        lenngths=[]
        sorted_indices_dict={}
      
        cluster_edge_index_list_dict={}
        cluster_exps_list_dict={}
        filter_cluster_list_dict={}
        filter_cell_list_dict={}
        super_cluster_exps_list_dict={}
        all_cell_id_list_dict={}
        for name in NAMES:
            print(f'Loading: {name}')
            cluster_edge_index_list=[]
            cluster_exps_list=[]
            super_cluster_exps_list=[]
            filter_cluster_list=[] # get centroid embedding
            filter_cell_list=[] #get cell embedding
            cell_id_list=[]
            all_cell_id_list=[]
            
            if split== True:
                cluster_dir=f'{cluster_path}/{group}/cluster_data_split'
            elif train == False:
                cluster_dir=f'{cluster_path}/{group}/cluster_data'
                
            with open(f'{cluster_dir}/{name}_clusters.pkl','rb') as f:
                cluster_list = pickle.load(f)
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
            
            emb_cells= torch.from_numpy(np.load(f'{emb_folder}/24/{group}/{name}.npy'))
            emb_centroids= torch.from_numpy(np.load(f'{emb_folder}/80/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids
            emb_super_centroids= torch.from_numpy(np.load(f'{emb_folder}/256/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids
            if len(emb_super_centroids.shape)>3:
                resize_transform = transforms.Resize((32, 32))

# Permute the tensor to change its shape from (N, H, W, C) to (N, C, H, W)
                emb_super_centroids = emb_super_centroids.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                emb_centroids = emb_centroids.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                emb_cells = emb_cells.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                # Apply the resize transform
                emb_super_centroids = resize_transform(emb_super_centroids)
                emb_centroids = resize_transform(emb_centroids)

                # Permute back to (N, H, W, C) if needed
                
            super_centroids = cluster_list.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
           
            valid_cluster_list= cluster_list
            valid_super_clusters = valid_cluster_list['super_cluster'].unique()    
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]

            # print(emb_cells.shape)
            # print(len(emb_cells), len( valid_cell_list_cluster))
            if dataset_type != None:
                if dataset_type ==1 :
                    temp_supter_cluster            =cluster_list[cluster_list[group] == 1]['super_cluster'].unique()
                    super_cluster_index           =[list(valid_super_clusters).index(cluster_id) for cluster_id in temp_supter_cluster ]
                 
                    filter_cluster          =valid_cluster_list[valid_cluster_list[group] == 1]['cluster'].unique()
                    valid_cell_list_cluster = cell_list_cluster[cell_list_cluster['cluster'].isin(set(filter_cluster))]
                    emb_cells               =emb_cells[valid_cell_list_cluster.index.to_numpy()]
                    valid_super_clusters    =temp_supter_cluster
                elif dataset_type ==0:
                    temp_supter_cluster            =cluster_list[cluster_list[group] == 0]['super_cluster'].unique()
                    super_cluster_index           =[list(valid_super_clusters).index(cluster_id) for cluster_id in temp_supter_cluster ]
                   
                    filter_cluster          =valid_cluster_list[valid_cluster_list[group] == 0]['cluster'].unique()
                    valid_cell_list_cluster = cell_list_cluster[cell_list_cluster['cluster'].isin(set(filter_cluster))]
                    emb_cells               =emb_cells[valid_cell_list_cluster.index.to_numpy()]
                    valid_super_clusters    =temp_supter_cluster
            cell_list_cluster=None
            super_centroids= super_centroids[valid_super_clusters]  
            emb_super_centroids= emb_super_centroids[valid_super_clusters]  
            valid_cell_list_cluster.reset_index(drop=True, inplace=True)

            gc.collect()
            ''''''
            # Use the sorted indices to get sorted centroids
            sorted_indices_dict[name]=self._sorted_indices(super_centroids)
            
            ''''''
               
            lenngths.append(len(super_centroids))
            
            # len of centroids == len of valid_cluster == len of emb_centroids
            # len of valid_cell_list_cluster == len of  emb_cells
            super_centroids_dict[name]            =super_centroids[:,1:]
            valid_super_clusters_dict[name]       =valid_super_clusters
            emb_cells_dict[name]            =emb_cells
            emb_centroids_dict[name]        =emb_centroids
            emb_super_centroids_dict[name]  =emb_super_centroids
            emb_cells=None
            emb_centroids=None
            emb_super_centroids=None
            #####################################################
            
            for i in range(len(valid_super_clusters)):


                cluster_list_in_super_cluster = valid_cluster_list[valid_cluster_list['super_cluster'] == valid_super_clusters[i]]
                filter_cluster          =cluster_list_in_super_cluster['cluster'].to_numpy()
                
                filter_cluster_list.append(filter_cluster)
                offset=0
                all_edge_index=[]
                all_cell_index= np.array([])
                all_centroid_exps=[]
                all_cell_exps=[]
                cell_id_list=[]
                # print(i)
                # cell_mask_all = np.zeros(len(valid_cell_list_cluster), dtype=bool)
                for cluster in filter_cluster:
                    cluster_cells= valid_cell_list_cluster[valid_cell_list_cluster['cluster']==cluster]
                    # print(i,i)
                    cell_ids= cluster_cells['cell_id'].to_numpy()
                    cell_id_list.append(cell_ids)
                    cell_list_in_square= self._get_points_in_square(dataframe= cluster_cells,r=int(80/2))
                    if len(cell_list_in_square)==0:
                        cell_list_in_square=cluster_cells
                    # cell_list_in_square=cluster_cells  #TESTINGGGG
    
                    centroid_exps=  np.array([np.sum(cell_list_in_square['counts'].to_numpy()/len(cell_list_in_square), axis=0)])
                    # centroid_exps= np.array([[0]*460])
                    all_centroid_exps.append(centroid_exps)
                    
                    cell_exps= np.stack(cluster_cells['counts'].to_numpy())
                    # cell_exps=np.array([[0]*460])
                    all_cell_exps.append(cell_exps)
                    
                    cell_index= cluster_cells.index
                    all_cell_index= np.concatenate([ all_cell_index, cell_index])
                    # all_cell_index=np.array([])
                    # print(all_cell_index.shape)
                    add= len(filter_cluster)+offset
                    edge_index=self._get_edge_index_(cluster_cells,k=6,add=add)
                    # edge_index=[]
                    for cell_i in range(len(cluster_cells)):
                        edge_index.append((list(filter_cluster).index(cluster), cell_i+ add))
          
                    offset = offset+ len(cluster_cells)
                    all_edge_index= all_edge_index+ edge_index
                    
                    cell_ids= None
                    cluster_cells=None
                    cell_list_in_square=None
                    centroid_exps=None
                    cell_exps=None
                    cell_index=None
                    edge_index=None
                    del cluster_cells,cell_list_in_square,cell_exps,centroid_exps,cell_index,edge_index
                    # gc.collect()
                # print(i,i,i)
                cell_id_list=np.hstack(cell_id_list)
                all_cell_id_list.append(cell_id_list)
                all_centroid_exps= np.vstack(all_centroid_exps)
                all_cell_exps= np.vstack(all_cell_exps)
                super_centroid_exps=np.array([np.sum(all_centroid_exps/len(all_centroid_exps), axis=0)])
                super_centroid_exps = super_centroid_exps / super_centroid_exps.sum(axis=1, keepdims=True)
                if nolog1p == False: 
                    super_centroid_exps = np.log1p(super_centroid_exps* 100)
                all_exps= np.vstack([all_centroid_exps,all_cell_exps])
                all_exps = all_exps / all_exps.sum(axis=1, keepdims=True) 
                if nolog1p == False:
                    all_exps = np.log1p(all_exps* 100)
                
                edge_index=self._get_edge_index_(dataframe=cluster_list_in_super_cluster,k=6,add=0)
                all_edge_index= edge_index + all_edge_index
                cluster_edge_index_list.append(all_edge_index)
                cluster_exps_list.append(all_exps)   
                filter_cell_list.append(all_cell_index)          
                super_cluster_exps_list.append(super_centroid_exps)
                cell_id_list=None
                super_centroid_exps=None
                all_cell_exps=None
                all_centroid_exps=None
                all_exps=None
                all_cell_index=None
                all_edge_index=None
                edge_index=None
                del all_exps, all_centroid_exps,super_centroid_exps,all_cell_index,all_edge_index,edge_index
                # gc.collect()
                
            gc.collect()
            all_cell_id_list_dict[name]= all_cell_id_list 
            cluster_edge_index_list_dict[name]=cluster_edge_index_list
            cluster_exps_list_dict[name]= cluster_exps_list
            filter_cluster_list_dict[name]= filter_cluster_list
            filter_cell_list_dict[name]= filter_cell_list
            super_cluster_exps_list_dict[name]=super_cluster_exps_list
            all_cell_id_list=None
            cluster_edge_index_list=None
            cluster_exps_list=None
            filter_cluster_list=None
            filter_cell_list=None
            super_cluster_exps_list=None
            valid_cell_list_cluster=None
            del cluster_edge_index_list,cluster_exps_list,filter_cluster_list,filter_cell_list,super_cluster_exps_list,valid_cell_list_cluster
        
        self.all_cell_id_list_dict              =all_cell_id_list_dict
        self.cluster_edge_index_list_dict       =cluster_edge_index_list_dict
        self.cluster_exps_list_dict             =cluster_exps_list_dict
        self.filter_cluster_list_dict           =filter_cluster_list_dict
        self.filter_cell_list_dict              =filter_cell_list_dict
        self.super_cluster_exps_list_dict       =super_cluster_exps_list_dict
        self.super_centroids_dict             =super_centroids_dict
        self.valid_super_clusters_dict        =valid_super_clusters_dict
        self.emb_cells_dict             =emb_cells_dict
        self.emb_centroids_dict         =emb_centroids_dict
        self.emb_super_centroids_dict         =emb_super_centroids_dict
        self.lengths                    =lenngths
        self.sorted_indices_dict        =sorted_indices_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        
        cluster_edge_index_list_dict=None
        cluster_exps_list_dict=None
        filter_cluster_list_dict=None
        valid_cell_list_cluster=None
        cell_list_cluster   =None
        emb_cells           =None
        emb_centroids       =None
        # print(self.cumlen[-1])
        gc.collect()
        self.global_to_local = self._create_global_index_map()
    def __len__(self):
        return self.cumlen[-1]
    def _get_edge_index_(self,dataframe, k=6,add=0):
    # """
        coordinates = dataframe[['x', 'y']].to_numpy()  # Shape: (n, 2)
        dist_matrix = distance.cdist(coordinates, coordinates, metric='euclidean')  # Shape: (n, n)
        n = len(coordinates)
        edge_index = []
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            
            for neighbor in neighbors:
                edge_index.append((i+add, neighbor+add))  # (source, target)
                edge_index.append((neighbor+add, i+add))
        return edge_index
    def _get_points_in_square(self,dataframe, r):
        half_side = r
        center= (dataframe['x'].mean(),dataframe['y'].mean())
        x_min, x_max = center[0] - half_side, center[0] + half_side
        y_min, y_max = center[1] - half_side, center[1] + half_side
        
        index_list_in_square =(
        (dataframe['x'] >= x_min) & (dataframe['x'] <= x_max) &
        (dataframe['y'] >= y_min) & (dataframe['y'] <= y_max)
        )
        list_in_square=dataframe[index_list_in_square]
        return list_in_square
    def _create_global_index_map(self):
        """Create a mapping from global index to (dataset index, local index)."""
        global_to_local = []
        for i, name in enumerate(self.id2name):
            local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
            global_to_local.extend(local_indices)
        return global_to_local
    def _sorted_indices(self,centroids):
        n_centroids = len(centroids)
        distances = cdist(centroids[:,1:], centroids[:,1:])
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
        return sorted_indices
    def __getitem__(self, index):
        
     
        i, idx = self.global_to_local[index]
        idx= self.sorted_indices_dict[self.id2name[i]][idx]
        super_centroid= self.super_centroids_dict[self.id2name[i]][idx]
        super_cluster_id= self.valid_super_clusters_dict[self.id2name[i]][idx]
        edge_index= self.cluster_edge_index_list_dict[self.id2name[i]][idx]
        exps= self.cluster_exps_list_dict[self.id2name[i]][idx]
        super_centroid_exps= self.super_cluster_exps_list_dict[self.id2name[i]][idx]
        cells_index_mask= self.filter_cell_list_dict[self.id2name[i]][idx]
        centroid_index_mask= self.filter_cluster_list_dict[self.id2name[i]][idx]

        # cell_ids= self.cell_id_list_dict[self.id2name[i]][idx]
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][centroid_index_mask]
        emb_cells= self.emb_cells_dict[self.id2name[i]][cells_index_mask]
        emb_super_centroid=self.emb_super_centroids_dict[self.id2name[i]][idx]
        
        centroid_num = len(centroid_index_mask)
        
        emb_x = torch.cat((emb_centroid, emb_cells), dim=0)
        emb_cells=None
        emb_centroid=None
        centroid_index_mask=None
        # super_centroid_exps=None
        # gc.collect()
    
        
      
        # return item
        
        return  Data(
            x= emb_x,  # Node features (including centroid and cells)
            edge_index=edge_index,  # Edge indices (intra-cluster and centroid-to-cell)
            emb_super_centroid=emb_super_centroid,  # Embedding for centroid
            super_centroid=super_centroid, # Centroid coordinates
            exps=exps,
            super_centroid_exps=super_centroid_exps,
            all_num= len(emb_x),
            centroid_num= centroid_num
            # cell_ids= cell_ids
        )
        # return emb_centroid, GE_centroid, centroid, emb_cells_in_cluster, cell_exps, cell_edge_index

       
def build_super_batch_graph(batch,device):
    """Build a graph for a batch of 400 clusters."""
    all_x = []  # Store node features
    all_edge_index = []  # Store edges
    super_centroid_exps=[]
    node_offset = 0  # To adjust indices for each cluster
    # print(batch.x.shape)
    super_centroid_exps = np.vstack(batch.super_centroid_exps)  # Get centroid expression data
   
    exps = np.vstack(batch.exps)  # Get cell expression data
    # print(centroid_exps.shape, cell_exps.shape)
    # print(super_centroid_exps.shape, exps.shape)
    exps= np.vstack((super_centroid_exps, exps))
    # print(len(batch.edge_index))
    centroid_index=[]
    # Assuming centroid embeddings are stored as a separate tensor
    # emb_centroids = batch.emb_centroid  # Stack centroid embeddings from the batch
    super_centroid = torch.tensor(np.array(batch.super_centroid))  # Get centroid positions
    dist_matrix = torch.cdist(super_centroid, super_centroid)
    k=min(6,len(super_centroid))
    neighbors = torch.topk(dist_matrix, k=k, largest=False).indices  # Top 6 neighbors
    
    for i in range(len(batch.edge_index)):
        # print(torch.tensor(batch.edge_index[i]).shape)        
        all_edge_index.append(torch.tensor(batch.edge_index[i]) + node_offset + len(batch.edge_index))
        add_edge_index=[]
        for j in range(batch.all_num[i]): #centroid_num
            add_edge_index.append((i,j+ node_offset + len(batch.edge_index)))
            if j < batch.centroid_num[i]:
                centroid_index.append(j+ node_offset + len(batch.edge_index))
            ####testing
            # for neighbor in neighbors[i]:
            #     add_edge_index.append((neighbor, j+ node_offset + len(batch.edge_index)))
        all_edge_index.append(torch.tensor(add_edge_index))
        node_offset += batch.all_num[i]
    edge_index = torch.cat(all_edge_index, dim=0).to(torch.long)
    # emb_centroids=batch.emb_centroid
    # print(batch.emb_super_centroid.shape)
    if len(batch.emb_super_centroid.shape)>1:
        batch.emb_super_centroid= batch.emb_super_centroid.view(-1,3,32,32)
    else:
        batch.emb_super_centroid= batch.emb_super_centroid.view(-1,batch.x.shape[1])
    all_x = torch.cat([batch.emb_super_centroid, batch.x], dim=0)
   
    

    
    # Add inter-cluster edges
    edge_index_centroid = []
    for i in range(len(super_centroid)):
        add= torch.stack([torch.full((k,), i), neighbors[i]])
        edge_index_centroid.append(add)
    edge_index_centroid = torch.cat(edge_index_centroid, dim=1)
    edge_index_centroid= edge_index_centroid.T
    edge_index = torch.cat([edge_index, edge_index_centroid], dim=0)

    
    ##############################
    # print(all_x.shape,all_x.dtype)
    return Data(x=all_x.to(device), edge_index=edge_index.to(device), edge_index_centroid=edge_index_centroid,centroid_index=centroid_index),exps,super_centroid_exps

    # @staticmethod
    # def partition(lst, n):
    #     division = len(lst) / float(n)
    #     return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

class SuperNeuronData_2(Dataset):
    def __init__(self,cluster_path= 'E:/DATA/crunch/tmp/cluster', emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed', augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I'],evel=False,nolog1p= False):
        self.augmentation = augmentation
        
        # emb_dir=  
        self.encoder_mode= encoder_mode
        group='evel'
        NAMES = name_list
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
        if evel==True:
            group= 'evel'
            dataset_type=-1
        print(f'Type: {dataset_type}, group: {group}')
        super_centroids_dict={}
        valid_super_clusters_dict={}
        emb_cells_dict={}
        emb_centroids_dict={}
        emb_super_centroids_dict={}
        lenngths=[]
        sorted_indices_dict={}
      
        cluster_edge_index_list_dict={}
        cluster_exps_list_dict={}
        filter_cluster_list_dict={}
        filter_cell_list_dict={}
        super_cluster_exps_list_dict={}
        all_cell_id_list_dict={}
        for name in NAMES:
            print(f'Loading: {name}')
            cluster_edge_index_list=[]
            cluster_exps_list=[]
            super_cluster_exps_list=[]
            filter_cluster_list=[] # get centroid embedding
            filter_cell_list=[] #get cell embedding
            cell_id_list=[]
            all_cell_id_list=[]
            
            if split== True:
                cluster_dir=f'{cluster_path}/{group}/cluster_data_split'
            elif train == False:
                cluster_dir=f'{cluster_path}/{group}/cluster_data'
                
            with open(f'{cluster_dir}/{name}_clusters.pkl','rb') as f:
                cluster_list = pickle.load(f)
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
            
            emb_cells= torch.from_numpy(np.load(f'{emb_folder}/24/{group}/{name}.npy'))
            emb_centroids= torch.from_numpy(np.load(f'{emb_folder}/80/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids
            emb_super_centroids= torch.from_numpy(np.load(f'{emb_folder}/256/{group}/{name}.npy')) # len of valid_clusters(['group'] != -1) = len of emb_centroids
            if len(emb_super_centroids.shape)>3:
                resize_transform = transforms.Resize((32, 32))

# Permute the tensor to change its shape from (N, H, W, C) to (N, C, H, W)
                emb_super_centroids = emb_super_centroids.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                emb_centroids = emb_centroids.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                emb_cells = emb_cells.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
                # Apply the resize transform
                emb_super_centroids = resize_transform(emb_super_centroids)
                emb_centroids = resize_transform(emb_centroids)

                # Permute back to (N, H, W, C) if needed
                
            super_centroids = cluster_list.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
           
            valid_cluster_list= cluster_list
            valid_super_clusters = valid_cluster_list['super_cluster'].unique()    
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]

            # print(emb_cells.shape)
            # print(len(emb_cells), len( valid_cell_list_cluster))
            if dataset_type != None:
                if dataset_type ==1 :
                    temp_supter_cluster            =cluster_list[cluster_list[group] == 1]['super_cluster'].unique()
                    super_cluster_index           =[list(valid_super_clusters).index(cluster_id) for cluster_id in temp_supter_cluster ]
                 
                    filter_cluster          =valid_cluster_list[valid_cluster_list[group] == 1]['cluster'].unique()
                    valid_cell_list_cluster = cell_list_cluster[cell_list_cluster['cluster'].isin(set(filter_cluster))]
                    emb_cells               =emb_cells[valid_cell_list_cluster.index.to_numpy()]
                    valid_super_clusters    =temp_supter_cluster
                elif dataset_type ==0:
                    temp_supter_cluster            =cluster_list[cluster_list[group] == 0]['super_cluster'].unique()
                    super_cluster_index           =[list(valid_super_clusters).index(cluster_id) for cluster_id in temp_supter_cluster ]
                   
                    filter_cluster          =valid_cluster_list[valid_cluster_list[group] == 0]['cluster'].unique()
                    valid_cell_list_cluster = cell_list_cluster[cell_list_cluster['cluster'].isin(set(filter_cluster))]
                    emb_cells               =emb_cells[valid_cell_list_cluster.index.to_numpy()]
                    valid_super_clusters    =temp_supter_cluster
            cell_list_cluster=None
            super_centroids= super_centroids[valid_super_clusters]  
            emb_super_centroids= emb_super_centroids[valid_super_clusters]  
            valid_cell_list_cluster.reset_index(drop=True, inplace=True)

            gc.collect()
            ''''''
            # Use the sorted indices to get sorted centroids
            sorted_indices_dict[name]=self._sorted_indices(super_centroids)
            
            ''''''
               
            lenngths.append(len(super_centroids))
            
            # len of centroids == len of valid_cluster == len of emb_centroids
            # len of valid_cell_list_cluster == len of  emb_cells
            super_centroids_dict[name]            =super_centroids[:,1:]
            valid_super_clusters_dict[name]       =valid_super_clusters
            emb_cells_dict[name]            =emb_cells
            # emb_centroids_dict[name]        =emb_centroids
            # emb_super_centroids_dict[name]  =emb_super_centroids
            emb_cells=None
            emb_centroids=torch.empty(emb_centroids.shape)
            emb_super_centroids=torch.empty(emb_super_centroids.shape)
            #####################################################
            
            for i in range(len(valid_super_clusters)):


                cluster_list_in_super_cluster = valid_cluster_list[valid_cluster_list['super_cluster'] == valid_super_clusters[i]]
                filter_cluster          =cluster_list_in_super_cluster['cluster'].to_numpy()
                
                filter_cluster_list.append(filter_cluster)
                offset=0
                all_edge_index=[]
                all_cell_index= np.array([])
                all_centroid_exps=[]
                all_cell_exps=[]
                cell_id_list=[]
                emb_centroids_list=[]
                # print(i)
                # cell_mask_all = np.zeros(len(valid_cell_list_cluster), dtype=bool)
                for cluster in filter_cluster:
                    cluster_cells= valid_cell_list_cluster[valid_cell_list_cluster['cluster']==cluster]
                    
                    emb_cluster_cells=emb_cells_dict[name][valid_cell_list_cluster['cluster']==cluster]
                    emb_cluster_cells = torch.mean(emb_cluster_cells, dim=0, keepdim=True)
                    emb_centroids_list.append(emb_cluster_cells)
                    emb_centroids[int(cluster),:]= emb_cluster_cells
                    emb_cluster_cells=None
                    
                    cell_ids= cluster_cells['cell_id'].to_numpy()
                    cell_id_list.append(cell_ids)
                    cell_list_in_square= self._get_points_in_square(dataframe= cluster_cells,r=int(80/2))
                    if len(cell_list_in_square)==0:
                        cell_list_in_square=cluster_cells
                    # cell_list_in_square=cluster_cells  #TESTINGGGG
    
                    centroid_exps=  np.array([np.sum(cell_list_in_square['counts'].to_numpy()/len(cell_list_in_square), axis=0)])
                    # centroid_exps= np.array([[0]*460])
                    all_centroid_exps.append(centroid_exps)
                    
                    cell_exps= np.stack(cluster_cells['counts'].to_numpy())
                    # cell_exps=np.array([[0]*460])
                    all_cell_exps.append(cell_exps)
                    
                    cell_index= cluster_cells.index
                    all_cell_index= np.concatenate([ all_cell_index, cell_index])
                    # all_cell_index=np.array([])
                    # print(all_cell_index.shape)
                    add= len(filter_cluster)+offset
                    edge_index=self._get_edge_index_(cluster_cells,k=6,add=add)
                    # edge_index=[]
                    for cell_i in range(len(cluster_cells)):
                        edge_index.append((list(filter_cluster).index(cluster), cell_i+ add))
          
                    offset = offset+ len(cluster_cells)
                    all_edge_index= all_edge_index+ edge_index
                    
                    cell_ids= None
                    cluster_cells=None
                    cell_list_in_square=None
                    centroid_exps=None
                    cell_exps=None
                    cell_index=None
                    edge_index=None
                    del cluster_cells,cell_list_in_square,cell_exps,centroid_exps,cell_index,edge_index
                    # gc.collect()
                # print(i,i,i)
                emb_super_centroids[i,:]= torch.mean(torch.stack(emb_centroids_list), dim=0)
                cell_id_list=np.hstack(cell_id_list)
                all_cell_id_list.append(cell_id_list)
                all_centroid_exps= np.vstack(all_centroid_exps)
                all_cell_exps= np.vstack(all_cell_exps)
                super_centroid_exps=np.array([np.sum(all_centroid_exps/len(all_centroid_exps), axis=0)])
                super_centroid_exps = super_centroid_exps / super_centroid_exps.sum(axis=1, keepdims=True)
                if nolog1p == False: 
                    super_centroid_exps = np.log1p(super_centroid_exps* 100)
                all_exps= np.vstack([all_centroid_exps,all_cell_exps])
                all_exps = all_exps / all_exps.sum(axis=1, keepdims=True) 
                if nolog1p == False:
                    all_exps = np.log1p(all_exps* 100)
                
                edge_index=self._get_edge_index_(dataframe=cluster_list_in_super_cluster,k=6,add=0)
                all_edge_index= edge_index + all_edge_index
                cluster_edge_index_list.append(all_edge_index)
                cluster_exps_list.append(all_exps)   
                filter_cell_list.append(all_cell_index)          
                super_cluster_exps_list.append(super_centroid_exps)
                cell_id_list=None
                super_centroid_exps=None
                all_cell_exps=None
                all_centroid_exps=None
                all_exps=None
                all_cell_index=None
                all_edge_index=None
                edge_index=None
                del all_exps, all_centroid_exps,super_centroid_exps,all_cell_index,all_edge_index,edge_index
                # gc.collect()
                
            gc.collect()
            all_cell_id_list_dict[name]= all_cell_id_list 
            cluster_edge_index_list_dict[name]=cluster_edge_index_list
            cluster_exps_list_dict[name]= cluster_exps_list
            filter_cluster_list_dict[name]= filter_cluster_list
            filter_cell_list_dict[name]= filter_cell_list
            super_cluster_exps_list_dict[name]=super_cluster_exps_list
            emb_centroids_dict[name]        =emb_centroids
            emb_super_centroids_dict[name]  =emb_super_centroids
            all_cell_id_list=None
            cluster_edge_index_list=None
            cluster_exps_list=None
            filter_cluster_list=None
            filter_cell_list=None
            super_cluster_exps_list=None
            valid_cell_list_cluster=None
            emb_centroids=None
            emb_super_centroids=None
            del cluster_edge_index_list,cluster_exps_list,filter_cluster_list,filter_cell_list,super_cluster_exps_list,valid_cell_list_cluster
        
        self.all_cell_id_list_dict              =all_cell_id_list_dict
        self.cluster_edge_index_list_dict       =cluster_edge_index_list_dict
        self.cluster_exps_list_dict             =cluster_exps_list_dict
        self.filter_cluster_list_dict           =filter_cluster_list_dict
        self.filter_cell_list_dict              =filter_cell_list_dict
        self.super_cluster_exps_list_dict       =super_cluster_exps_list_dict
        self.super_centroids_dict             =super_centroids_dict
        self.valid_super_clusters_dict        =valid_super_clusters_dict
        self.emb_cells_dict             =emb_cells_dict
        self.emb_centroids_dict         =emb_centroids_dict
        self.emb_super_centroids_dict         =emb_super_centroids_dict
        self.lengths                    =lenngths
        self.sorted_indices_dict        =sorted_indices_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        
        cluster_edge_index_list_dict=None
        cluster_exps_list_dict=None
        filter_cluster_list_dict=None
        valid_cell_list_cluster=None
        cell_list_cluster   =None
        emb_cells           =None
        emb_centroids       =None
        # print(self.cumlen[-1])
        gc.collect()
        self.global_to_local = self._create_global_index_map()
    def __len__(self):
        return self.cumlen[-1]
    def _get_edge_index_(self,dataframe, k=6,add=0):
    # """
        coordinates = dataframe[['x', 'y']].to_numpy()  # Shape: (n, 2)
        dist_matrix = distance.cdist(coordinates, coordinates, metric='euclidean')  # Shape: (n, n)
        n = len(coordinates)
        edge_index = []
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            
            for neighbor in neighbors:
                edge_index.append((i+add, neighbor+add))  # (source, target)
                edge_index.append((neighbor+add, i+add))
        return edge_index
    def _get_points_in_square(self,dataframe, r):
        half_side = r
        center= (dataframe['x'].mean(),dataframe['y'].mean())
        x_min, x_max = center[0] - half_side, center[0] + half_side
        y_min, y_max = center[1] - half_side, center[1] + half_side
        
        index_list_in_square =(
        (dataframe['x'] >= x_min) & (dataframe['x'] <= x_max) &
        (dataframe['y'] >= y_min) & (dataframe['y'] <= y_max)
        )
        list_in_square=dataframe[index_list_in_square]
        return list_in_square
    def _create_global_index_map(self):
        """Create a mapping from global index to (dataset index, local index)."""
        global_to_local = []
        for i, name in enumerate(self.id2name):
            local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
            global_to_local.extend(local_indices)
        return global_to_local
    def _sorted_indices(self,centroids):
        n_centroids = len(centroids)
        distances = cdist(centroids[:,1:], centroids[:,1:])
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
        return sorted_indices
    def __getitem__(self, index):
        
     
        i, idx = self.global_to_local[index]
        idx= self.sorted_indices_dict[self.id2name[i]][idx]
        super_centroid= self.super_centroids_dict[self.id2name[i]][idx]
        super_cluster_id= self.valid_super_clusters_dict[self.id2name[i]][idx]
        edge_index= self.cluster_edge_index_list_dict[self.id2name[i]][idx]
        exps= self.cluster_exps_list_dict[self.id2name[i]][idx]
        super_centroid_exps= self.super_cluster_exps_list_dict[self.id2name[i]][idx]
        cells_index_mask= self.filter_cell_list_dict[self.id2name[i]][idx]
        centroid_index_mask= self.filter_cluster_list_dict[self.id2name[i]][idx]

        # cell_ids= self.cell_id_list_dict[self.id2name[i]][idx]
        emb_centroid=self.emb_centroids_dict[self.id2name[i]][centroid_index_mask]
        emb_cells= self.emb_cells_dict[self.id2name[i]][cells_index_mask]
        emb_super_centroid=self.emb_super_centroids_dict[self.id2name[i]][idx]
        
        centroid_num = len(centroid_index_mask)
        
        emb_x = torch.cat((emb_centroid, emb_cells), dim=0)
        emb_cells=None
        emb_centroid=None
        centroid_index_mask=None
        # super_centroid_exps=None
        # gc.collect()
    
        
      
        # return item
        
        return  Data(
            x= emb_x,  # Node features (including centroid and cells)
            edge_index=edge_index,  # Edge indices (intra-cluster and centroid-to-cell)
            emb_super_centroid=emb_super_centroid,  # Embedding for centroid
            super_centroid=super_centroid, # Centroid coordinates
            exps=exps,
            super_centroid_exps=super_centroid_exps,
            all_num= len(emb_x),
            centroid_num= centroid_num
            # cell_ids= cell_ids
        )
        # return emb_centroid, GE_centroid, centroid, emb_cells_in_cluster, cell_exps, cell_edge_index

   