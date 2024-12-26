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
# import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# import torchvision.transforms.functional as TF
import random
import pickle
# from skimage.measure import regionprops
# from tqdm import tqdm
import gc


class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None,name=None, ds=None,r=32 ,sr=False, aug=False, norm=False, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = '../data'
        self.r=r
        # self.r = 200 // 4

        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            # transforms.ColorJitter(0.2, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.norm = norm

        
        names= sample_names
        # names= sample_names[:1]
        print('Loading sdata...')
        # self.sdata_dict = {i: self.get_sdata(i) for i in names}
        img_dict={}
        nuc_img_dict={}
        anucleus_dict={}
        cell_id_group_dict={}
        
        for i in names:
            sdata= self.get_sdata(i)
            img_dict[i]= sdata['HE_registered'].to_numpy()
            nuc_img_dict[i]=sdata['HE_nuc_registered']
            cell_id_group_dict[i]=sdata['cell_id-group']
            anucleus_dict[i]= sdata['anucleus']
            sdata=None
            del sdata
            # gc.collect()

        
        # self.img_dict= {i: self.sdata_dict[i]['HE_original'].to_numpy()  for i in names}
        self.img_dict=img_dict
        # self.nuc_img_dict=nuc_img_dict
        self.cell_id_group_dict=cell_id_group_dict
        self.anucleus_dict=anucleus_dict
        gene_list = self.anucleus_dict[names[0]].var['gene_symbols'].values
        
        print('Loading metadata...')
        # self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        # self.log1p_dict= {i: self.sdata_dict[i]['anucleus'].X for i in names}
        print(self.train)
        
        center_dict={}
        loc_dict={}
        log1p_dict={}
        lengths=[]
        for i in names:
            # regions = nuc_img_dict[i]
            center_list=[]
            cell_id_list=[]
            train_length= len( self.anucleus_dict[i].layers['counts'])
            
            partial_train_length=int(train_length*0.9)
            split_train_binary=[1] * partial_train_length + [0] * (train_length-partial_train_length)
            # split_train_binary=[1] * 62000 + [0] * (train_length-62000)
            random.shuffle(split_train_binary)
            
            split_path= "./train_split"
            os.makedirs(f"{split_path}", exist_ok=True)
            if self.train and os.path.exists(f'{split_path}/{i}_train.pkl')==False:
                with open(f'{split_path}/{i}_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(split_train_binary, f)
            elif os.path.exists(f'{split_path}/{i}_train.pkl')==True:
                with open(f'{split_path}/{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                    split_train_binary = pickle.load(f)
                    
        
            # if self.train:
            #     log1p_dict[i]=self.anucleus_dict[i].X[np.array(split_train_binary)==1]
            #     lengths.append(partial_train_length)
            # else:
            #     log1p_dict[i]=self.anucleus_dict[i].X[np.array(split_train_binary)==0]
            #     lengths.append(train_length-partial_train_length)
            
            cell_id_train = self.cell_id_group_dict[i].obs[self.cell_id_group_dict[i].obs['group'] == 'train']['cell_id'].to_numpy()
            cell_id_train = list(set(cell_id_train).intersection(set(self.anucleus_dict[i].obs['cell_id'].unique())))

            ground_truth = self.anucleus_dict[i].X[self.anucleus_dict[i].obs['cell_id'].isin(cell_id_train),:]
            if self.train:
                log1p_dict[i]=ground_truth[np.array(split_train_binary)==1]
                lengths.append(partial_train_length)
            else:
                log1p_dict[i]=ground_truth[np.array(split_train_binary)==0]
                lengths.append(train_length-partial_train_length)
            bool_list= [0]*(max(cell_id_train)+1)
            for id in cell_id_train:
                bool_list[id]=1
            print(len(split_train_binary),train_length,int(train_length*0.9))
            
            check_bin=0
            with open(f'./pre_load/{i}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
            for props in cell_list:
                cell_id= props['cell_id']
                # print(cell_id)
                flag= True
                if cell_id >= len(bool_list) or bool_list[cell_id] ==0:
                    flag =False
                
                    
                    
                if flag == False :
                    continue
                # if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
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
            center_dict[i]=center_list
            loc_dict[i]=cell_id_list
        bool_list=None
        del  bool_list
        self.log1p_dict=log1p_dict
        self.lengths=lengths
        self.cumlen = np.cumsum(self.lengths)
       
        self.center_dict = center_dict
        self.loc_dict = loc_dict
        
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

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
        im= self.img_dict[self.id2name[i]]
        exp = self.log1p_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]
        
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
            patch = im[:, minr:maxr, minc:maxc]
        # patch = im[:, minr:maxr, minc:maxc]
        
        patch = Image.fromarray(np.transpose(patch,(2,1,0)))
        if patch.size !=(self.r*2,self.r*2):
            patch=patch.resize((self.r*2,self.r*2))
        # except:
        #     print( minr,maxr,minc,maxc, im.shape)
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            item["image"] = patch
            item["position"] = torch.Tensor(center)
            item["expression"] = exp
            item['id']=i-1
            return item

        else:
            item["image"] = patch
            item["position"] = torch.Tensor(center)
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            item['id']=i-1
            return item

    def __len__(self):
        return self.cumlen[-1]
  
        
    def get_sdata(self, name):
        path= f'{self.dir}/{name}.zarr'
        # path = os.path.join()
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

class MINI_DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, name=None,r=32 ,sr=False, aug=False, norm=False, fold=0):
        super(MINI_DATA_BRAIN, self).__init__()
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
        # gc.collect()

        
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

        ground_truth = self.anucleus.X[self.anucleus.obs['cell_id'].isin(cell_id_train),:]
        if self.train:
            log1p_dict=ground_truth[np.array(split_train_binary)==1]
            length= partial_train_length
        else:
            log1p_dict=ground_truth[np.array(split_train_binary)==0]
            length= train_length-partial_train_length
        bool_list= [0]*(max(cell_id_train)+1)
        for id in cell_id_train:
            bool_list[id]=1
        print(len(split_train_binary),train_length,int(train_length*0.9))
            
        check_bin=0
        with open(f'./pre_load/{name}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
        for props in cell_list:
            cell_id= props['cell_id']
            # if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
            #     continue
            flag= True
            if cell_id >= len(bool_list) or bool_list[cell_id] ==0:
                flag =False
        
            if flag == False: 
                continue
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
        
        # print(len(center_list))
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
            return item

        else:
            item["image"] = patch
            item["position"] = torch.Tensor(center)
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item

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

# class CLUSTER_BRAIN(torch.utils.data.Dataset):

#     def __init__(self,cluster_fold='E:/DATA/crunch/tmp', emb_folder=f'E:/DATA/crunch/tmp/preprocessed', augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
#                  name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']):
#         self.augmentation = augmentation
#         emb_dir=emb_folder
#         # emb_dir=  
#         self.encoder_mode= encoder_mode
#         NAMES = name_list
#         group='validation'
#         dataset_type= None
#         if train== False and split == True:
#                 dataset_type= 0 
#         elif train== True and split == True:
#                 dataset_type= 1
#         elif train== True and split == False:
#                 dataset_type= -1
#         if split ==True:
#             group='train'
#             try:
#                 NAMES.remove('DC1')
#             except:
#                 print('DC1 is not in list')
#         # NAMES= NAMES[:1]
#         print(f'Type: {dataset_type}')
#         valid_cell_list_cluster_dict={}
#         emb_cells_dict={}
#         lenngths=[]
#         pos_dict={}
#         exps_dict={}
#         cell_ids_dict={}
#         for name in NAMES:
#             preload_dir=f'../pre_load'
#             # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
#             #     cell_list_org= pickle.load(f)
#             if split== True:
#                 cluster_dir=f'{cluster_fold}/cluster/train/cluster_data_split'
#             elif train == False:
#                 cluster_dir=f'{cluster_fold}/cluster/evel/cluster_data'
                
#             with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
#                 cell_list_cluster = pickle.load(f)
           
            
            

#             # Filter out invalid clusters (those with 'train' = -1)
            
#             valid_cell_list_cluster= cell_list_cluster[cell_list_cluster[group] != -1]
#             # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]
#             # print(len(valid_cell_list_cluster))
#             emb_cells= torch.from_numpy(np.load(f'{emb_dir}/24/{group}/{name}.npy'))
#             # print(emb_cells.shape[0]/16)
            
#             # print(emb_cells.shape, len( valid_cell_list_cluster),len(filtered_cells_index),len(cell_list_cluster))
#             # len of emb_cells == len of cell_list_cluster
#             emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
#             # print(len(valid_cell_list_cluster),len(emb_cells))
#             # print(emb_cells.shape)
#             # print(len(emb_cells), len( valid_cell_list_cluster))
            
#             if dataset_type != None:
#                 if dataset_type ==1 :
#                     emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 1]
#                     valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 1]
#                 elif dataset_type ==0:
#                     emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 0]
#                     valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 0]
#             # print(len(valid_cell_list_cluster),len(emb_cells),name)
#             cell_list_cluster=None
#             cell_counts= np.stack(valid_cell_list_cluster['counts'].to_numpy())
#         # # print(cell_counts.shape)
#             normalized_counts = cell_counts / cell_counts.sum(axis=1, keepdims=True) * 100
#             cell_exps = np.log1p(normalized_counts)
#             exps_dict[name]   =cell_exps
#             lenngths.append(len(emb_cells))
#             pos= valid_cell_list_cluster[['x', 'y']].to_numpy()
#             pos_dict[name]  = pos
#             # len of valid_cell_list_cluster == len of  emb_cells
#             emb_cells_dict[name]            =emb_cells
            
#             cell_ids=valid_cell_list_cluster['cell_id'].to_numpy()
#             cell_ids_dict[name]=cell_ids
#             valid_cell_list_cluster=None
#         self.cell_ids_dict=cell_ids_dict
#         self.exps_dict= exps_dict
#         self.pos_dict= pos_dict
#         self.emb_cells_dict             =emb_cells_dict
#         self.lengths                    =lenngths
#         self.cumlen =np.cumsum(self.lengths)
#         self.id2name = dict(enumerate(NAMES))
#         self.global_to_local = self._create_global_index_map()
#     def _create_global_index_map(self):
#         """Create a mapping from global index to (dataset index, local index)."""
#         global_to_local = []
#         for i, name in enumerate(self.id2name):
#             local_indices = list(zip([i] * self.lengths[i], range(self.lengths[i])))
#             global_to_local.extend(local_indices)
#         return global_to_local
#     def __getitem__(self, index):
#         # i = 0
#         # 
#         # # print(index)
#         # while index >= self.cumlen[i]:
#         #     i += 1
#         # idx = index
#         # if i > 0:
#         #     idx = index - self.cumlen[i - 1]
#         i, idx = self.global_to_local[index]
#         emb_cell= self.emb_cells_dict[self.id2name[i]][idx]
#         exps= self.exps_dict[self.id2name[i]][idx]
#         cell_id= self.cell_ids_dict[self.id2name[i]][idx]
#         item = {}
       
#         item["feature"] = emb_cell
#         x,y= self.pos_dict[self.id2name[i]][idx]
#         # x=int(x)
#         # y=int(y)
#         item["position"] = torch.Tensor((x,y))
#         item["expression"] = exps
#         item['id']=i-1
#         item['cell_id']=cell_id
#         return item

#     def __len__(self):
#         return self.cumlen[-1]
  
        
#     def get_sdata(self, name):
#         path= f'{self.dir}/{name}.zarr'
#         # path = os.path.join()
#         # print(path)
#         sdata = sd.read_zarr(path)
#         return sdata
    
class CLUSTER_BRAIN(torch.utils.data.Dataset):

    def __init__(self,cluster_fold='E:/DATA/crunch/tmp/cluster', emb_folder=f'E:/DATA/crunch/tmp/preprocessed', augmentation=True,encoder_mode=False, random_seed=1234, train=True, split= False,
                 name_list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I'],centroid=False):
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
        print(f'Type: {dataset_type}')
        valid_cell_list_cluster_dict={}
        emb_cells_dict={}
        lenngths=[]
        pos_dict={}
        exps_dict={}
        cell_ids_dict={}
        cluster_id_cells_dict={}
        for name in NAMES:
            preload_dir=f'../pre_load'
            # with open(f'{preload_dir}/{name}_cells.pkl','rb') as f:
            #     cell_list_org= pickle.load(f)
            if split== True:
                cluster_dir=f'{cluster_fold}/train/cluster_data_split'
            elif train == False:
                cluster_dir=f'{cluster_fold}/evel/cluster_data'
                
            with open(f'{cluster_dir}/{name}_cells.pkl','rb') as f:
                cell_list_cluster = pickle.load(f)
           
            
            

            # Filter out invalid clusters (those with 'train' = -1)
            
            valid_cell_list_cluster= cell_list_cluster[cell_list_cluster[group] != -1]
            # filtered_cells_index = [i for i in range(len(cell_list_org)) if cell_list_org[i]['label'] == group]
            # print(len(valid_cell_list_cluster))
            emb_cells= torch.from_numpy(np.load(f'{emb_dir}/24/{group}/{name}.npy'))
            if centroid==True:
                emb_centroids= torch.from_numpy(np.load(f'{emb_dir}/80/{group}/{name}.npy'))
                centroids = cell_list_cluster.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
            # print(emb_cells.shape[0]/16)
            
            # print(emb_cells.shape, len( valid_cell_list_cluster),len(filtered_cells_index),len(cell_list_cluster))
            # len of emb_cells == len of cell_list_cluster
            if group =='train':
                emb_cells= emb_cells[cell_list_cluster[group] != -1] # len of emb_cells == len of valid_cell_list_cluster
            # print(len(valid_cell_list_cluster),len(emb_cells))
            # print(emb_cells.shape)
            # print(len(emb_cells), len( valid_cell_list_cluster))
            if dataset_type != None:
                if dataset_type ==1 :
                    valid_clusters           = cell_list_cluster[cell_list_cluster[group] == 1]['cluster'].unique()
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 1]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 1]
                elif dataset_type ==0:
                    valid_clusters           = cell_list_cluster[cell_list_cluster[group] == 0]['cluster'].unique()
                    emb_cells               =emb_cells[valid_cell_list_cluster[group].to_numpy() == 0]
                    valid_cell_list_cluster =valid_cell_list_cluster[valid_cell_list_cluster[group] == 0]
            # print(len(valid_cell_list_cluster),len(emb_cells),name)
            cell_list_cluster=None
            cell_exps= np.stack(valid_cell_list_cluster['counts'].to_numpy())
        # # print(cell_counts.shape)
            cell_exps = cell_exps / cell_exps.sum(axis=1, keepdims=True) * 100
            cell_exps = np.log1p(cell_exps)
            pos= valid_cell_list_cluster[['x', 'y']].to_numpy()
            if centroid==True:
                centroids= centroids[valid_clusters]   
                emb_centroids= emb_centroids[valid_clusters]
                all_centroids_exps=[]
            
                for i in range(len(valid_clusters)):
                    
                    cells_list_in_cluster = valid_cell_list_cluster[valid_cell_list_cluster['cluster'] == valid_clusters[i]]
                    x_center, y_center = centroids[i,1],centroids[i,2]
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
                        centroid_exps=  np.array([np.sum(cells_list_in_square['counts'].to_numpy()/len(cells_list_in_square), axis=0)])
                        centroid_exps = centroid_exps / centroid_exps.sum(axis=1, keepdims=True) 
                        centroid_exps = np.log1p(centroid_exps* 100)
                        all_centroids_exps.append(centroid_exps)
                        centroid_exps=None
                        cells_list_in_square=None
                        cells_list_in_cluster=None
                        del cells_list_in_square, cells_list_in_cluster, centroid_exps
                    else:
                        centroid_exps=np.empty(460)
                all_centroids_exps=np.vstack(all_centroids_exps)
                cell_exps=np.vstack([all_centroids_exps,cell_exps])
                centroids= centroids[:,1:]
                pos_dict[name]  = np.vstack([centroids,pos])
                emb_cells_dict[name]            =np.vstack([emb_centroids,emb_cells])
                centroid_index= -(valid_clusters+1)
                cell_ids=valid_cell_list_cluster['cell_id'].to_numpy()
                cluster_id_cells=valid_cell_list_cluster['cluster'].to_numpy()
                cluster_id_cells_dict[name]=np.hstack([centroid_index,cluster_id_cells])
                cell_ids_dict[name]=np.hstack([centroid_index,cell_ids])
            else:
                pos_dict[name]  = pos
                emb_cells_dict[name]            =emb_cells
                cell_ids_dict[name]=valid_cell_list_cluster['cell_id'].to_numpy()
                cluster_id_cells_dict=valid_cell_list_cluster['cluster'].to_numpy()
            exps_dict[name]   =cell_exps
            # len of valid_cell_list_cluster == len of  emb_cells
            lenngths.append(len(emb_cells_dict[name]   ))
            print(len(cell_ids_dict[name]), len(emb_cells_dict[name]), len(cell_exps))
            valid_cell_list_cluster=None
            all_centroids_exps=None
            emb_centroids=None
            emb_cells=None
            cell_ids=None
            cluster_id_cells=None
            cell_exps=None
        self.cluster_id_cells_dict=cluster_id_cells_dict
        self.cell_ids_dict=cell_ids_dict
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
        # 
        # # print(index)
        # while index >= self.cumlen[i]:
        #     i += 1
        # idx = index
        # if i > 0:
        #     idx = index - self.cumlen[i - 1]
        i, idx = self.global_to_local[index]
        emb_cell= self.emb_cells_dict[self.id2name[i]][idx]
        exps= self.exps_dict[self.id2name[i]][idx]
        cell_id= self.cell_ids_dict[self.id2name[i]][idx]
        cluster_id= self.cluster_id_cells_dict[self.id2name[i]][idx]
        item = {}
       
        item["feature"] = emb_cell
        x,y= self.pos_dict[self.id2name[i]][idx]
        # x=int(x)
        # y=int(y)
        item["position"] = torch.Tensor((x,y))
        item["expression"] = exps
        item['id']=i-1
        item['cell_id']=cell_id
        item['cluster_id']=cluster_id
        return item

    def __len__(self):
        return self.cumlen[-1]
  
        
    def get_sdata(self, name):
        path= f'{self.dir}/{name}.zarr'
        # path = os.path.join()
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata