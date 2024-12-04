import os
import pickle
import torchvision.transforms as transforms
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('../')
from dataset import DataGenerator
from torch.utils.data import Dataset, DataLoader
from model import ImageEncoder

dir = f'D:/DATA/Gene_expression/Crunch/patches/80'
save_dir='D:/DATA/Gene_expression/Crunch/preprocessed/80'
tensor_folder = dir

# dir=f'F:/DATA/crunch_large/zip_server'
NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder= ImageEncoder()
encoder.eval()
encoder=encoder.to(device)
emb_stack=[]
# for name in NAMES:
#     dataset = DataGenerator(tensor_folder,list=[name])
#     dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
#     vq_stack=[]
    
#     with torch.no_grad():
#         i=0
#         for batch in dataloader:
#             batch = batch.to(device)
#             emb= encoder(batch) 
#             emb_stack.append(emb.cpu())
#             i+=1
#             print(i)
#             # if i==2:
#             #     break
# emb_stack = torch.cat(emb_stack, dim=0)
# print(emb_stack.shape)
# emb_stack=emb_stack.numpy()
    
# local_index=0
# for name in NAMES:
    
#     train_stack=[]
#     test_stack=[]
#     validation_stack=[]
#     pre_load_path= '../../pre_load'
    
#     with open(f'{pre_load_path}/{name}_cells.pkl','rb') as f:
#                 cell_list= pickle.load(f)
    
#     for i in range(len(cell_list)):
#     # for i in range(32):
#         if cell_list[i]['label']=='train':
#             train_stack.append(emb_stack[i+local_index])
#         elif cell_list[i]['label']=='test':
#             test_stack.append(emb_stack[i+local_index])
#         elif cell_list[i]['label']=='validation':
#             validation_stack.append(emb_stack[i+local_index])
            
#     os.makedirs(f'{save_dir}/train',exist_ok=True)
#     os.makedirs(f'{save_dir}/test',exist_ok=True)
#     os.makedirs(f'{save_dir}/validation',exist_ok=True)
    
    
#     local_index+= len(cell_list)
#     train_stack= np.vstack(train_stack)
#     # print(train_stack.shape)
#     test_stack= np.vstack(test_stack)
#     validation_stack= np.vstack(validation_stack)
#     np.save(f'{save_dir}/train/{name}.npy',train_stack)
#     np.save(f'{save_dir}/test/{name}.npy',test_stack)
#     np.save(f'{save_dir}/validation/{name}.npy',validation_stack)
            
        
        
from collections import defaultdict

    
    
local_index = 0
label_map = defaultdict(list)

for name in NAMES[1:]:
    emb_stack=[]
    dataset = DataGenerator(tensor_folder,list=[name])
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    vq_stack=[]
    
    with torch.no_grad():
        i=0
        for batch in dataloader:
            batch = batch.to(device)
            emb= encoder(batch) 
            emb_stack.append(emb.cpu())
            i+=1
            print(i)
    
    emb_stack = torch.cat(emb_stack, dim=0)
    print(emb_stack.shape)
    emb_stack=emb_stack.numpy()
    pre_load_path = '../../pre_load'
    
    # Load cell list
    with open(f'{pre_load_path}/{name}_cells.pkl', 'rb') as f:
        cell_list = pickle.load(f)

    # Separate labels efficiently
    labels = [cell['label'] for cell in cell_list]
    # indices = np.arange(len(cell_list)) + local_index  # Compute global indices
    indices = np.arange(len(cell_list))
    # Use NumPy's boolean indexing for efficiency
    train_mask = np.array(labels) == 'train'
    test_mask = np.array(labels) == 'test'
    validation_mask = np.array(labels) == 'validation'
    
    # Use boolean masks to extract corresponding embeddings
    train_stack = emb_stack[indices[train_mask]]
    test_stack = emb_stack[indices[test_mask]]
    validation_stack = emb_stack[indices[validation_mask]]
    
    # Optionally, add to a dictionary for further use
    # label_map[name].append({
    #     'train': train_stack,
    #     'test': test_stack,
    #     'validation': validation_stack,
    # })
    
    # Update local index
    local_index += len(cell_list)
    os.makedirs(f'{save_dir}/train',exist_ok=True)
    os.makedirs(f'{save_dir}/test',exist_ok=True)
    os.makedirs(f'{save_dir}/validation',exist_ok=True)
    
    
    local_index+= len(cell_list)
    train_stack= np.vstack(train_stack)
    # print(train_stack.shape)
    test_stack= np.vstack(test_stack)
    validation_stack= np.vstack(validation_stack)
    np.save(f'{save_dir}/train/{name}.npy',train_stack)
    np.save(f'{save_dir}/test/{name}.npy',test_stack)
    np.save(f'{save_dir}/validation/{name}.npy',validation_stack)