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
    
    
class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None,name=None, ds=None,r=32 ,sr=False, aug=False, norm=False, device='cuda:0'):
        super(DATA_BRAIN, self).__init__()
        self.dir = '../data'
        self.dir = f'F:/Data/crunch_large/Zip_server/'
        self.r=r
        # self.r = 200 // 4
        self.device = device
        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.norm = norm

        
        names= sample_names[:1]
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
            # nuc_img_dict[i]=sdata['HE_nuc_registered']
            cell_id_group_dict[i]=sdata['cell_id-group']
            anucleus_dict[i]= sdata['anucleus']
            del sdata
            gc.collect()

        
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
            center_dict[i]=center_list
            loc_dict[i]=cell_id_list
        del nuc_img_dict    
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
        # if self.train:
        #     # item["image"] = patch
        #     # item["position"] = torch.Tensor(center)
        #     # item["expression"] = exp
        #     # item['id']=i-1
            

        # else:
        #     # item["image"] = patch
        #     # item["position"] = torch.Tensor(center)
        #     # item["expression"] = exp
        #     # item["center"] = torch.Tensor(center)
        #     # item['id']=i-1
        #     return patch
        return patch,exp

    def __len__(self):
        # return 16
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
        
            if flag == False :
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