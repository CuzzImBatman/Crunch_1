from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import spatialdata_plot
import spatialdata as sd
import random
import glob
import os
import pickle

import scanpy as sc
import scprep as scp
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from skimage.measure import regionprops
from tqdm import tqdm
import tifffile
from graph_construction import calcADJ

Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViT_HER2ST(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(ViT_HER2ST, self).__init__()
        self.cnt_dir = r'./data/her2st/data/ST-cnts'
        self.img_dir = r'./data/her2st/data/ST-imgs'
        self.pos_dir = r'./data/her2st/data/ST-spotfiles'
        self.lbl_dir = r'./data/her2st/data/ST-pat/lbl'

        self.r = 224 // 4

        gene_list = list(np.load(r'./data/her_hvg_cut_1000.npy', allow_pickle=True))

        self.gene_list = gene_list

        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]

        self.train = train
        self.sr = sr

        samples = names[1:33]
        # samples = names[1:4]

        te_names = [samples[fold]]

        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.label = {i: None for i in self.names}

        self.lbl2id = {'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2, 'cancer in situ': 3,
                       'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1}

        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.patch_dict = {}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180), transforms.ToTensor()])
        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        label = self.label[self.id2name[i]]
        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None

        adj = self.adj_dict[name]

        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)

            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()
            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    patches[i] = patch.permute(2, 0, 1) #cxy
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        print(path)
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        df.set_index('id', inplace=True)
        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)





class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = '.D:/data/crunch'
        self.r = 128 // 4

        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_NI']

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180), transforms.ToTensor()])
        self.norm = norm

        names= sample_names[:1]
        print('Loading sdata...')
        self.sdata_dict = {i: self.get_sdata(i) for i in names}
        
        gene_list = self.sdata_dict[names[0]]['anucleus'].var['gene_symbols'].values
        
        print('Loading metadata...')

        self.gene_set = list(gene_list)

        self.log1p_dict= {i: self.sdata_dict[i]['anucleus'].X for i in names}
        
        
        center_dict={}
        loc_dict={}
        for i in names:
            regions = regionprops(self.sdata_dict[i]['HE_nuc_original'][0, :, :].to_numpy())
            center_list=[]
            cell_id_list=[]
            train_length= len( self.sdata_dict[i]['anucleus'].layers['counts'])
            
            split_train_binary=[1] * int(train_length*0.9) + [0] * (train_length-int(train_length*0.9))
            split_train_binary=[1] * 62000 + [0] * (train_length-62000)
            random.shuffle(split_train_binary)
            print(len(split_train_binary),train_length,int(train_length*0.9))
            with open('i.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(split_train_binary, f)

            # Getting back the objects:
            # with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
            #     obj0, obj1, obj2 = pickle.load(f)
            for props in tqdm(regions):
                cell_id= props.label
                
                if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
                    break
                
                if split_train_binary[cell_id] ==0:
                    continue
                
                
                centroid = props.centroid
                center_list.append([int(centroid[1]), int(centroid[0])]) #x y
                cell_id_list.append(int(cell_id))
            center_dict[i]=center_list
            loc_dict[i]=cell_id_list
            
            
        
        self.center_dict = center_dict
        self.loc_dict = loc_dict
        
        
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.center_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    # def __getitem__(self, index):
        # i = index
        # name = self.id2name[i]
        # im = Image.fromarray(np.transpose(self.sdata_dict[self.id2name[i]]['HE_original'].to_numpy(),(2,1,0))) #cxy
        
        # im = self.transforms(im) #yxc
        # im = im.permute(1, 0, 2) #xyc
       
        # exps = self.log1p_dict[self.id2name[i]]
        # centers = self.center_dict[self.id2name[i]]
        # # loc = self.loc_dict[self.id2name[i]]
        # positions = torch.Tensor(centers)
        # # patch_dim = 3 * self.r * self.r * 4

        # if self.id2name[i] in self.patch_dict:
        #     patches = self.patch_dict[self.id2name[i]]
        # else:
        #     patches = None
        # adj = self.adj_dict[name]

        # n_patches = len(centers)
        # exps = torch.Tensor(exps)
        # if patches is None:
        #     patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

        #     for i in range(n_patches):
        #         center = centers[i]
        #         x, y = center
        #         patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r),
        #                 :]
        #         patches[i] = patch.permute(2, 0, 1) # cxy
        #     self.patch_dict[name] = patches
        # if self.train:
        #     return patches, positions, exps, adj
        # else:
        #     return patches, positions, exps, torch.Tensor(centers), adj
    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.sdata_dict[self.id2name[i]]['HE_original'].to_numpy() #cyx
        
        # im = self.transforms(im)
        # im = im.permute(2, 1, 0)
       
        exps = self.log1p_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        # loc = self.loc_dict[self.id2name[i]]
        positions = torch.Tensor(centers)
        # patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        n_patches = len(centers)
        exps = torch.Tensor(exps)
        if patches is None:
            patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]
                x_center, y_center = center
               
                # Calculate the crop boundaries
                minr, maxr = y_center - self.r, y_center + self.r
                minc, maxc = x_center - self.r, x_center + self.r

                # Ensure boundaries are within the image dimensions
                pad_top = max(0, -minr)
                minr = max(0, minr)

                pad_bottom = max(0, maxr - im.shape[1])
                maxr = min(maxr, im.shape[1])

                pad_left = max(0, -minc)
                minc = max(0, minc)

                pad_right = max(0, maxc - im.shape[2])
                maxc = min(maxc, im.shape[2])

                # Crop and pad the image if needed
                if pad_top + pad_bottom + pad_left + pad_right > 0:
                    patch = np.pad(im[:, minr:maxr, minc:maxc],
                                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                                mode='constant', constant_values=0)
                else:
                    patch = im[:, minr:maxr, minc:maxc]
                patch=self.transforms(Image.fromarray(np.transpose(patch,(2,1,0))))
                patches[i] = torch.Tensor(np.array(patch))# cxy
            self.patch_dict[name] = patches
        if self.train:
            return patches, positions, exps, adj
        else:
            return patches, positions, exps, torch.Tensor(centers), adj
    def __len__(self):
        return len(self.log1p_dict)

    def get_sdata(self, name):
        path= f'C:/data/crunch/data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

    # def get_meta(self, name, gene_list=None):
    #     meta = pd.read_csv('./data/10X/151507/10X_Visium_151507_meta.csv', index_col=0)
    #     return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
