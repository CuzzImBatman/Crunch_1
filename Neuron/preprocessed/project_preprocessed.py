import os
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('../')
from model_mclstexp import mclSTExp_Attention_Pretrain
from dataset import DataGenerator
from torch.utils.data import Dataset, DataLoader
group_type=['train','evel']
group_type=['train']

absolute_path='E:/DATA/crunch/tmp'
r=int(256/2)
model_path='C:\\DATA\Crunch\mclSTExp\\model_result_centroid\\24\\checkpoint_epoch_219.pth.tar'


# dir=f'F:/DATA/crunch_large/zip_server'
for group in group_type:
    dir = f'{absolute_path}/preprocessed/{r*2}/{group}'
    save_dir=f'{absolute_path}/projection/{r*2}/{group}'
    os.makedirs(save_dir,exist_ok=True)
    numpy_folder = dir
    NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path)
    args= checkpoint['args']
    model = mclSTExp_Attention_Pretrain(
                                encoder_name= None,
                               spot_dim=args.dim,
                               temperature=args.temperature,
                               image_dim=args.image_embedding_dim,
                               projection_dim=args.projection_dim,
                               heads_num=args.heads_num,
                               heads_dim=args.heads_dim,
                               head_layers=args.heads_layers,
                               dropout=args.dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    projection_head= model.image_projection
    projection_head.eval()
    projection_head=projection_head.to(device)


    local_index=0
    for name in NAMES:
        dataset = DataGenerator(numpy_folder=numpy_folder,list=[name],augmentation=False)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        print(len(dataloader))
        emb_stack=[]
        with torch.no_grad():
            i=0
            for batch in dataloader:
                batch = batch.to(device)
                emb= projection_head(batch) 
                emb_stack.append(emb.cpu())
                i+=1
                print(i,emb.shape)
        emb_stack= torch.cat(emb_stack, dim=0)
        print(emb_stack.shape)
        emb_stack=emb_stack.numpy()
        np.save(f'{save_dir}/{name}.npy',emb_stack)
        
            
        
        
        
    
    
