import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import torchvision.transforms as transforms
import glob
import spatialdata as sd

# import cv2
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms.functional as TF
import random
import pickle
from skimage.measure import regionprops
from tqdm import tqdm
import gc


class CLUSTER_BRAIN(torch.utils.data.Dataset):

    def __init__(self, emb_folder=f'D:/DATA/Gene_expression/Crunch/preprocessed', augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
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
        valid_cell_list_cluster_dict={}
        emb_cells_dict={}
        pos_dict={}
        exps_dict={}
        lenngths=[]
        for name in NAMES:
            preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'../cluster/train/cluster_data_split'
            elif train == False:
                cluster_dir=f'../cluster/validation'
                
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
           
            
            

            # Filter out invalid clusters (those with 'train' = -1)
            
            valid_cell_list_cluster= cell_list_cluster[cell_list_cluster[group] != -1]
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]
            
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/80/{group}/{name}.npy'))
            # print(emb_cells.shape[0]/16)
            
            # print(emb_cells.shape, len( valid_cell_list_cluster),len(filtered_cells_index),len(cell_list_cluster))
            # len of emb_cells == len of cell_list_cluster
            emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
            # print(emb_cells.shape)
            # print(len(emb_cells), len( valid_cell_list_cluster))
            cell_list_cluster=None
            if dataset_type != None:
                if dataset_type ==1 :
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 1]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 1]
                elif dataset_type ==0:
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 0]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 0]
            cell_counts= np.stack(valid_cell_list_cluster['counts'].to_numpy())
        # # print(cell_counts.shape)
            normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
            cell_exps = np.log1p(normalized_counts)
            exps_dict[name]   =cell_exps
            lenngths.append(len(emb_cells))
            pos= valid_cell_list_cluster[['x', 'y']].to_numpy()
            pos_dict[name]  = pos
            # len of valid_cell_list_cluster == len of  emb_cells
            # valid_cell_list_cluster_dict[name]   =valid_cell_list_cluster
            emb_cells_dict[name]            =emb_cells
            # valid_cell_list_cluster_dict=None
            # valid_cell_list_cluster=None
            
        self.exps_dict= exps_dict
        self.pos_dict= pos_dict
        self.emb_cells_dict             =emb_cells_dict
        self.lengths                    =lenngths
        # self.valid_cell_list_cluster_dict    =valid_cell_list_cluster_dict
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        print(lenngths, len(self.cumlen))
        valid_cell_list_cluster=None
        cell_list_cluster   =None
        emb_cells           =None

    def __getitem__(self, index):
        i = 0
        item = {}
        # print(index)
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        # print(index,i,len(self.loc_dict[self.id2name[i]]))
        emb_cell= self.emb_cells_dict[self.id2name[i]][idx]
        # print(len(self.valid_cell_list_cluster_dict[self.id2name[i]]),idx,self.id2name[i])
        # valid_cell_list_cluster= self.valid_cell_list_cluster_dict[self.id2name[i]].iloc[idx]
        exps= self.exps_dict[self.id2name[i]][idx]
        # print(center)
        # exps=  np.array([valid_cell_list_cluster['counts']])
        # # print( centroid_exps.shape,index)
        # normalized_counts = exps / exps.sum(axis=1, keepdims=True) * 100
        # exps = np.log1p(normalized_counts)
        item["feature"] = emb_cell
        # x,y=valid_cell_list_cluster[['x', 'y']]
        x,y= self.pos_dict[self.id2name[i]][idx]
        # x=int(x)
        # y=int(y)
        item["position"] = torch.Tensor((x,y))
        item["expression"] = exps
        item['id']=i-1
        return item

    def __len__(self):
        if len(self.cumlen)==1:
            return self.cumlen[0]
        else:
            print(self.cumlen)
            return self.cumlen[-1]
  
        
    def get_sdata(self, name):
        path= f'{self.dir}/{name}.zarr'
        # path = os.path.join()
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata