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
        self.dir = '.D:/data/crunch'
        self.r = 128 // 4

        sample_names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_NI']

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
        names= sample_names[:1]
        print('Loading sdata...')
        self.sdata_dict = {i: self.get_sdata(i) for i in names}
        self.img_dict= {i: self.sdata_dict[i]['HE_original'].to_numpy()  for i in names}
        gene_list = self.sdata_dict[names[0]]['anucleus'].var['gene_symbols'].values
        
        print('Loading metadata...')
        # self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        self.log1p_dict= {i: self.sdata_dict[i]['anucleus'].X for i in names}
        
        
        center_dict={}
        loc_dict={}
        lengths=[]
        for i in names:
            regions = regionprops(self.sdata_dict[i]['HE_nuc_original'][0, :, :].to_numpy())
            center_list=[]
            cell_id_list=[]
            train_length= len( self.sdata_dict[i]['anucleus'].layers['counts'])
            
            partial_train_length=int(train_length*0.9)
            lengths.append(partial_train_length)
            split_train_binary=[1] * partial_train_length + [0] * (train_length-partial_train_length)
            # split_train_binary=[1] * 62000 + [0] * (train_length-62000)
            random.shuffle(split_train_binary)
            
            cell_id_train = self.sdata_dict[i]['cell_id-group'].obs[self.sdata_dict[i]['cell_id-group'].obs['group'] == 'train']['cell_id'].to_numpy()
            cell_id_train = list(set(cell_id_train).intersection(set(self.sdata_dict[i]['anucleus'].obs['cell_id'].unique())))

            ## Get y from the anucleus data
            ground_truth = self.sdata_dict[i]['anucleus'].layers['counts'][self.sdata_dict[i]['anucleus'].obs['cell_id'].isin(cell_id_train),:]
            
            
            print(len(split_train_binary),train_length,int(train_length*0.9))
            with open('i.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(split_train_binary, f)

            # Getting back the objects:
            # with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
            #     obj0, obj1, obj2 = pickle.load(f)
            check_bin=0
            for props in tqdm(regions):
                cell_id= props.label
                # if cell_id >= len( self.sdata_dict[i]['anucleus'].layers['counts']):
                #     continue
                if check_bin >=len(split_train_binary):
                    break
                if split_train_binary[check_bin] ==0:
                    check_bin+=1
                    continue
                
                check_bin+=1
                
                centroid = props.centroid
                center_list.append([int(centroid[1]), int(centroid[0])])
                cell_id_list.append(int(cell_id))
            # print(len(center_list))
            center_dict[i]=center_list
            loc_dict[i]=cell_id_list
            
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
            return item

        else:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item

    def __len__(self):
        return self.cumlen[-1]

    def get_sdata(self, name):
        path= f'C:/data/crunch/data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

    # def get_meta(self, name, gene_list=None):
    #     meta = pd.read_csv('./data/10X/151507/10X_Visium_151507_meta.csv', index_col=0)
    #     return meta


class HERDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super().__init__()
        self.cnt_dir = 'E:/Projects/Gene_expression/data/Her2st/data/ST-cnts'
        self.img_dir = 'E:/Projects/Gene_expression/data/Her2st/data/ST-imgs'
        self.pos_dir = 'E:/Projects/Gene_expression/data/Her2st/data/ST-spotfiles'
        self.lbl_dir = 'E:/Projects/Gene_expression/data/Her2st/data/ST-pat'
        self.r = 224 // 2
        gene_list = list(np.load('E:/Projects/Gene_expression/data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()  # ['A1.tsv.gz', 'A2.tsv.gz', ...]
        names = [i[:2] for i in names]  # ['A1', 'A2', 'A3',..]

        self.train = train

        samples = names[1:33]  # ['A2' - 'G3'] len=32
        te_names = [samples[fold]]  # fold = 0 # A2
        tr_names = list(set(samples) - set(te_names))
        if train:
            names = tr_names
        else:
            names = te_names
            self.meta_dict = {i: self.get_meta(i) for i in names}
            self.names = te_names
            self.label = {i: None for i in self.names}
            self.lbl2id = {
                'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
                'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
            }
            if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
                idx = self.meta_dict[self.names[0]].index
                lbl = self.lbl_dict[self.names[0]]
                lbl = lbl.loc[idx, :]['label'].values
                self.label[self.names[0]] = lbl

        print("Loading imgs ...")
        self.img_dict = {i: self.get_img(i) for i in names}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        # self.exp_dict = {i: m([self.gene_set].values) for i, m in self.meta_dict.items()}
        # print(self.exp_dict)
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}

        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            return item

        else:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item

    def __len__(self):
        return self.cumlen[-1]

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        # print(pos)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
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

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'

        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name  # data/her2st/data/ST-imgs/D/D6
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + 'lbl' + '/' + name + '_labeled_coordinates.tsv'
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

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""

    def __init__(self, train=True, val=False, gene_list=None, ds=None, sr=False, fold=0):
        super(SKIN, self).__init__()

        self.dir = 'D:\dataset\CSCC_data\GSE144240_RAW/'
        self.r = 224 // 2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)
        test_names = ['P2_ST_rep2']

        gene_list = list(np.load('../data/skin_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        self.train = train
        self.sr = sr

        samples = names[:2]
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            return item
        else:
            patch = transforms.ToTensor()(patch)
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item


    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        path = glob.glob(self.dir + '*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = glob.glob(self.dir + '*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = glob.glob(self.dir + '*spot*' + name + '.tsv')[0]
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

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        # meta.to_csv(f"D:\dataset\CSCC_data\GSE144240_RAW/{name}_metainfo.csv")
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)




class TenxDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):

        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)
    def __len__(self):
        return len(self.barcode_tsv)
    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]
        image = self.whole_image[(v1 - 112):(v1 + 112), (v2 - 112):(v2 + 112)]
        image = self.transform(image)

        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()
        item['barcode'] = barcode
        item['position'] = torch.Tensor([v1, v2])

        return item

