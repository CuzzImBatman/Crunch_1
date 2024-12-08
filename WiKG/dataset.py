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

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_features, exps):
        self.features = preprocessed_features
        self.exps = exps

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.exps[idx]
    
    

class WIKG_ONE(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, name=None,r=32 ,sr=False, aug=False, norm=False, fold=0):
        super(WIKG_ONE, self).__init__()
        self.dir = '../data'
        self.dir=f'F:/DATA/crunch_large/zip_server'

        self.r=int(r)
        # self.r = 200 // 4

        # sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.norm = norm

        
        # names= sample_names[:1]
        print('Loading sdata...')
        # self.sdata_dict = {i: self.get_sdata(i) for i in names}
        
        
        sdata= self.get_sdata(name)
        self.img= sdata['HE_registered'].to_numpy()
        nuc_img=sdata['HE_nuc_registered']
        self.cell_id_group=sdata['cell_id-group']
        self.anucleus= sdata['anucleus']
        del sdata
        gc.collect()

        
        # self.img_dict= {i: self.sdata_dict[i]['HE_original'].to_numpy()  for i in names}
        
        gene_list = self.anucleus.var['gene_symbols'].values
        
        print('Loading metadata...')
        # self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        # self.log1p_dict= {i: self.sdata_dict[i]['anucleus'].X for i in names}
        print(self.train)
        
        # center_dict={}
        # loc_dict={}
        # log1p_dict={}
        # lengths=[]
        
            # regions = nuc_img_dict[i]
        center_list=[]
        cell_id_list=[]
        train_length= len( self.anucleus.layers['counts'])
        
        partial_train_length=int(train_length*0.9)
        split_train_binary=[1] * partial_train_length + [0] * (train_length-partial_train_length)
        # split_train_binary=[1] * 62000 + [0] * (train_length-62000)
        random.shuffle(split_train_binary)
        
        split_path= "./train_split"
        os.makedirs(f"{split_path}", exist_ok=True)
        if self.train and os.path.exists(f'{split_path}/{name}_train.pkl')==False:
            with open(f'{split_path}/{name}_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(split_train_binary, f)
        elif os.path.exists(f'{split_path}/{name}_train.pkl')==True:
            with open(f'{split_path}/{name}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                split_train_binary = pickle.load(f)
                
    
        # if self.train:
        #     log1p_dict=self.anucleus.X[np.array(split_train_binary)==1]
        #     length= partial_train_length
        # else:
        #     log1p_dict=self.anucleus.X[np.array(split_train_binary)==0]
        #     length= train_length-partial_train_length
            
        cell_id_train = self.cell_id_group.obs[self.cell_id_group.obs['group'] == 'train']['cell_id'].to_numpy()
        cell_id_train = list(set(cell_id_train).intersection(set(self.anucleus.obs['cell_id'].unique())))

        cell_id_test=self.cell_id_group.obs[self.cell_id_group.obs['group'] == 'test']['cell_id']
        cell_id_validation=self.cell_id_group.obs[self.cell_id_group.obs['group'] == 'validation']['cell_id']


        ground_truth = self.anucleus.X[self.anucleus.obs['cell_id'].isin(cell_id_train),:]
        if self.train:
            log1p_dict=ground_truth[np.array(split_train_binary)==1]
            length= partial_train_length
        else:
            log1p_dict=ground_truth[np.array(split_train_binary)==0]
            length= train_length-partial_train_length
        max_len= max(max(cell_id_train),max(cell_id_test),max(cell_id_validation))
        bool_list= [0]*(max_len+1)
        for id in cell_id_train:
            bool_list[id]=1
        for id in cell_id_test:
            bool_list[id]=2
        for id in cell_id_validation:
            bool_list[id]=3
        print(len(split_train_binary),train_length,int(train_length*0.9))
            
        check_bin=0
        with open(f'./pre_load/{name}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
        for props in cell_list:
            cell_id= props['cell_id']
            # if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
            #     continue
            flag= True
            if cell_id >= len(bool_list) or bool_list[cell_id] !=1:
                flag =False
        
            if flag == False :
                continue
            # if flag == True   and self.train==False:
            #     continue
            if check_bin >=len(split_train_binary):
                break
            if (split_train_binary[check_bin] ==0 and self.train==True) or\
                (split_train_binary[check_bin] ==1 and self.train==False):
                check_bin+=1
                continue
            
            check_bin+=1
            
            centroid = props['center']
            center_list.append([int(centroid[1]), int(centroid[0])])
            cell_id_list.append(int(cell_id))
        
        print(len(center_list))
        self.center_list=center_list
        del nuc_img
        self.log1p_dict=log1p_dict
        self.length=length
       
        

        self.patch_dict = {}

    def __getitem__(self, index):
        
        item={}
        im= self.img
        exp = self.log1p_dict[index]
        center = self.center_list[index]
        # print(center)
        x_center, y_center = center
               
                # Calculate the crop boundaries
        minr, maxr = y_center - self.r, y_center + self.r
        minc, maxc = x_center - self.r, x_center + self.r

        # Ensure boundaries are within the image dimensions
        
        if (minr <0) or (minc <0) or (maxr <0) or (maxc <0):
            pad_top = max(0, -minr)
            minr = max(0, minr)

            pad_bottom = max(0, maxr - im.shape[1])
            maxr = min(maxr, im.shape[1])

            pad_left = max(0, -minc)
            minc = max(0, minc)

            pad_right = max(0, maxc - im.shape[2])
            maxc = min(maxc, im.shape[2])

        # Crop and pad the image if needed
        
            patch = np.pad(im[:, minr:maxr, minc:maxc],
                        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=0)
        else:
            # print( minr,maxr,minc,maxc, im.shape)
            patch = im[:, minr:maxr, minc:maxc]
        # patch = im[:, minr:maxr, minc:maxc]
        
        patch = Image.fromarray(np.transpose(patch,(2,1,0)))
        if patch.size !=(self.r*2,self.r*2):
            patch=patch.resize((self.r*2,self.r*2))
        # except:
            
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            item["image"] = patch
            item["position"] = torch.Tensor(center)
            item["expression"] = exp
            return patch,exp

        else:
            item["image"] = patch
            item["position"] = torch.Tensor(center)
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return patch,exp

    def __len__(self):
        return self.length
  
        
    def get_sdata(self, name):
        path= f'{self.dir}/{name}.zarr'
        # path = os.path.join()
        print(path)
        sdata = sd.read_zarr(path)
        return sdata

class Dummy(torch.utils.data.Dataset):
    def __init__(self, train=True):
        split_path= "./train_split"
        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
        names= sample_names
        lenghts=[]
        for i in names:
            split_path= "./train_split"
            split_train_binary=[]
            with open(f'{split_path}/{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                split_train_binary = pickle.load(f)
            if train:
                lenghts.append(int(len(split_train_binary)*0.9))
            else:
                lenghts.append(len(split_train_binary)-int(len(split_train_binary)*0.9))
        self.cumlen = np.cumsum( lenghts)

    def __getitem__(self, index):
        i = 0
        item = {}
        # print(index)
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        item['id']=i-1
        return item
    def __len__(self):
        return self.cumlen[-1]




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
        lenngths=[]
        pos_dict={}
        exps_dict={}
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
            # print(len(valid_cell_list_cluster))
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/80/{group}/{name}.npy'))
            # print(emb_cells.shape[0]/16)
            
            # print(emb_cells.shape, len( valid_cell_list_cluster),len(filtered_cells_index),len(cell_list_cluster))
            # len of emb_cells == len of cell_list_cluster
            emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
            # print(len(valid_cell_list_cluster),len(emb_cells))
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
            # print(len(valid_cell_list_cluster),len(emb_cells),name)
            cell_counts= np.stack(valid_cell_list_cluster['counts'].to_numpy())
        # # print(cell_counts.shape)
            normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
            cell_exps = np.log1p(normalized_counts)
            exps_dict[name]   =cell_exps
            lenngths.append(len(emb_cells))
            pos= valid_cell_list_cluster[['x', 'y']].to_numpy()
            pos_dict[name]  = pos
            # len of valid_cell_list_cluster == len of  emb_cells
            emb_cells_dict[name]            =emb_cells
            valid_cell_list_cluster_dict=None
            valid_cell_list_cluster=None
            
        self.exps_dict= exps_dict
        self.pos_dict= pos_dict
        self.emb_cells_dict             =emb_cells_dict
        self.lengths                    =lenngths
        self.cumlen =np.cumsum(self.lengths)
        self.id2name = dict(enumerate(NAMES))
        self.global_to_local = self._create_global_index_map()
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
        # while index >= self.cumlen[i]:
        #     i += 1
        # idx = index
        # if i > 0:
        #     idx = index - self.cumlen[i - 1]
            
        i, idx = self.global_to_local[index]
        item = {}
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
        return self.cumlen[-1]
   