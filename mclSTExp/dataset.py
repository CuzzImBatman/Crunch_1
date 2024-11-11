import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
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


class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = '../data'
        self.r = 128 // 4

        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.norm = norm

        
        names= sample_names
        # names= sample_names[:1]
        print('Loading sdata...')
        self.sdata_dict = {i: self.get_sdata(i) for i in names}
        self.img_dict= {i: self.sdata_dict[i]['HE_original'].to_numpy()  for i in names}
        gene_list = self.sdata_dict[names[0]]['anucleus'].var['gene_symbols'].values
        
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
            regions = regionprops(self.sdata_dict[i]['HE_nuc_original'][0, :, :].to_numpy())
            center_list=[]
            cell_id_list=[]
            train_length= len( self.sdata_dict[i]['anucleus'].layers['counts'])
            
            partial_train_length=int(train_length*0.9)
            split_train_binary=[1] * partial_train_length + [0] * (train_length-partial_train_length)
            # split_train_binary=[1] * 62000 + [0] * (train_length-62000)
            random.shuffle(split_train_binary)
            
            if self.train and os.path.exists(f'{i}_train.pkl')==False:
                with open(f'{i}_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(split_train_binary, f)
            elif os.path.exists(f'{i}_train.pkl')==True:
                with open(f'{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                    split_train_binary = pickle.load(f)
                    
        
            if self.train:
                log1p_dict[i]=self.sdata_dict[i]['anucleus'].X[np.array(split_train_binary)==1]
                lengths.append(partial_train_length)
            else:
                log1p_dict[i]=self.sdata_dict[i]['anucleus'].X[np.array(split_train_binary)==0]
                lengths.append(train_length-partial_train_length)
            
            cell_id_train = self.sdata_dict[i]['cell_id-group'].obs[self.sdata_dict[i]['cell_id-group'].obs['group'] == 'train']['cell_id'].to_numpy()
            cell_id_train = list(set(cell_id_train).intersection(set(self.sdata_dict[i]['anucleus'].obs['cell_id'].unique())))

            # ground_truth = self.sdata_dict[i]['anucleus'].layers['counts'][self.sdata_dict[i]['anucleus'].obs['cell_id'].isin(cell_id_train),:]
            
            print(len(split_train_binary),train_length,int(train_length*0.9))
            
            check_bin=0
            for props in tqdm(regions):
                cell_id= props.label
                # if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
                #     continue
                if check_bin >=len(split_train_binary):
                    break
                if (split_train_binary[check_bin] ==0 and self.train) or\
                    (split_train_binary[check_bin] ==1 and self.train==False):
                    check_bin+=1
                    continue
                
                check_bin+=1
                
                centroid = props.centroid
                center_list.append([int(centroid[1]), int(centroid[0])])
                cell_id_list.append(int(cell_id))
            # print(len(center_list))
            center_dict[i]=center_list
            loc_dict[i]=cell_id_list
            
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
        # pad_top = max(0, -minr)
        # minr = max(0, minr)

        # pad_bottom = max(0, maxr - im.shape[1])
        # maxr = min(maxr, im.shape[1])

        # pad_left = max(0, -minc)
        # minc = max(0, minc)

        # pad_right = max(0, maxc - im.shape[2])
        # maxc = min(maxc, im.shape[2])

        # # Crop and pad the image if needed
        # if pad_top + pad_bottom + pad_left + pad_right > 0:
        #     patch = np.pad(im[:, minr:maxr, minc:maxc],
        #                 ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        #                 mode='constant', constant_values=0)
        # else:
        #     patch = im[:, minr:maxr, minc:maxc]
        patch = im[:, minr:maxr, minc:maxc]
        patch = Image.fromarray(np.transpose(patch,(2,1,0)))
        
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
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

    # def get_meta(self, name, gene_list=None):
    #     meta = pd.read_csv('./data/10X/151507/10X_Visium_151507_meta.csv', index_col=0)
    #     return meta



