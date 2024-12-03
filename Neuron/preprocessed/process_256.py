import os
import spatialdata as sd
import pickle
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('../')
from model import ImageEncoder
from dataset import DataGenerator
from torch.utils.data import Dataset, DataLoader
group='train'
dir = f'D:/DATA/Gene_expression/Crunch/patches/256/{group}'
save_dir=f'D:/DATA/Gene_expression/Crunch/preprocessed/256/{group}'
os.makedirs(save_dir,exist_ok=True)
numpy_folder = dir

# dir=f'F:/DATA/crunch_large/zip_server'
NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder= ImageEncoder()
encoder.eval()
encoder=encoder.to(device)


local_index=0
for name in NAMES:
    dataset = DataGenerator(numpy_folder,list=[name])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    print(len(dataloader))
    emb_stack=[]
    with torch.no_grad():
        i=0
        for batch in dataloader:
            batch = batch.to(device)
            emb= encoder(batch) 
            emb_stack.append(emb.cpu())
            i+=1
            print(i)
    emb_stack= torch.cat(emb_stack, dim=0)
    emb_stack=emb_stack.numpy()
    np.save(f'{save_dir}/{name}.npy',emb_stack)
    
            
        
        
        
    
    
