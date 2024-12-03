import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Vector Quantizer
from torch.utils.tensorboard import SummaryWriter
import csv
import os



class DataGenerator(Dataset):
    def __init__(self, numpy_folder, augmentation=True, random_seed=1234, k=5, split= False,
                 list= ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']):
        self.augmentation = augmentation
        # loc_list = sorted([os.path.basename(name) for name in glob.glob(os.path.join(image_folder, '*'))])
        # random.seed(random_seed)
        # random.shuffle(loc_list)

        # partitions = self.partition(loc_list, k)
        # selected_loc_list = [i for m in m_list for i in partitions[m]]

        # self.image_list = [
        #     img for loc in selected_loc_list for img in glob.glob(os.path.join(image_folder, loc, '*.png'))
        # ]
        NAMES = list
        # NAMES= NAMES[:1]
        
        tensors_list=[         
                np.load(f'{numpy_folder}/{name}.npy') for name in NAMES
    ]
        # except:
        #     tensors_list=[         
        #          torch.load(f'{numpy_folder}/{name}.pt') for name in NAMES
        # ]
        tensors_list= np.vstack(tensors_list)
        # for num in tensors_list:
        #     num= transforms.ToTensor()(num)
        # tensors_list=torch.from_numpy(tensors_list)    
        self.tensors_list = tensors_list

    def __len__(self):
        return len(self.tensors_list)

    def __getitem__(self, index):
        tensor =transforms.ToTensor()(self.tensors_list[index])

        if self.augmentation:
            # Flip upside-down (vertically)
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])  # Flip along the height dimension

            # Flip left-to-right (horizontally)
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])

        # img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        return tensor

    @staticmethod
    def partition(lst, n):
        division = len(lst) / float(n)
        return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
