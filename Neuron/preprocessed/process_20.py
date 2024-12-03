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
from Neuron.preprocessed.VQVAE import VQVAE,VQVAEDataGenerator
from dataset import DataGenerator
from torch.utils.data import Dataset, DataLoader

dir = f'D:/DATA/Gene_expression/Crunch/patches/20'
save_dir='D:/DATA/Gene_expression/Crunch/preprocessed/20'
tensor_folder = dir

# dir=f'F:/DATA/crunch_large/zip_server'
NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQVAE(image_size=20, latent_dim=16, num_embeddings=1024)
checkpoint= torch.load('../patches/checkpoint/encoder-epoch-0009-loss-49.5791.pth')
vqvae.load_state_dict(checkpoint['model_state_dict'])

encoder=vqvae.encoder
vq_layer= vqvae.vq_layer

encoder.eval()
vq_layer.eval()

encoder=encoder.to(device)
vq_layer=vq_layer.to(device)

dataset = VQVAEDataGenerator(tensor_folder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
vq_stack=[]
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        emb= encoder(batch)
        vq,_= vq_layer(emb)
        vq_stack.append(vq.cpu())
    
    vq_stack= torch.cat(vq_stack, dim=0)
    vq_stack=vq_stack.numpy()
local_index=0
for name in NAMES:
    
    train_stack=[]
    test_stack=[]
    validation_stack=[]
    pre_load_path= '../../pre_load'
    
    with open(f'{pre_load_path}/{name}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
    
    for i in range(len(cell_list)):
        if cell_list[i]['label']=='train':
            train_stack.append(vq_stack[i+local_index])
        elif cell_list[i]['label']=='test':
            test_stack.append(vq_stack[i+local_index])
        elif cell_list[i]['label']=='validation':
            validation_stack.append(vq_stack[i+local_index])
            
    os.makedirs(f'{save_dir}/train',exist_ok=True)
    os.makedirs(f'{save_dir}/test',exist_ok=True)
    os.makedirs(f'{save_dir}/validation',exist_ok=True)

    
    local_index+= len(cell_list)
    train_stack= np.vstack(train_stack)
    test_stack= np.vstack(test_stack)
    validation_stack= np.vstack(validation_stack)
    np.save(f'{save_dir}/train/{name}.npy',train_stack)
    np.save(f'{save_dir}/test/{name}.npy',test_stack)
    np.save(f'{save_dir}/validation/{name}.npy',validation_stack)
            
        
        
        
    
    
