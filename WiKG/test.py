import sys
import os
import argparse
import csv
import numpy as np
from model import WiKG
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from dataset import DATA_BRAIN,Dummy,PreprocessedDataset
from pathlib import Path
import anndata
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import pickle
# torch.multiprocessing.set_start_method('spawn')
from utils import get_R

NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']


def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        pred = model(data)
        # print(label.shape,pred.shape)
        loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))

    return pred


@torch.no_grad()
def val_one_epoch(model, val_loader, device, data_type='val'):
    model.eval()
    labels = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='green')

    for i, (data, label) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        labels = torch.cat([labels, label], dim=0)
        preds = torch.cat([preds, output.detach()], dim=0)

    return preds.cpu(), labels.cpu()

def preprocess_dataset(original_dataset, encoder, device):
    """
    Preprocess the images using the encoder and return the features and labels.
    """
    encoder.eval()  # Set encoder to evaluation mode
    features = []
    exps = []

    dataloader = DataLoader(original_dataset, batch_size=1024, shuffle=False)
    print(len(dataloader))
    i=0
    with torch.no_grad():
        for images, exp in dataloader:
            images = images.to(device)
            
            features_batch = encoder(images)
            features.append(features_batch.cpu())
            exps.extend(exp)
            i=i+1
            print(i)
           

    # Concatenate features along the batch dimension
    features = torch.cat(features, dim=0)
    exps=np.stack(exps)
    return features, exps



def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=4096, help='patch_size')

    parser.add_argument('--embed_dim', type=int, default=1024, help="The dimension of instance-level representations")
    parser.add_argument('--patch_size', type=int, default=112, help='patch_size')
    parser.add_argument('--utils', type=str, default=None, help='utils path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=460)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_dir', default='./', help='path where to save')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')

    return parser.parse_args()

def save_checkpoint(epoch, model, optimizer,scheduler, args, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args
    }
    dir=f"{args.save_dir}/model_result/{args.patch_size}"
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, f"{dir}/{filename}")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(epoch, model, optimizer,scheduler,args):
    filename=f"checkpoint_epoch_{epoch}.pth.tar"
    dir=f"{args.save_dir}model_result/{args.patch_size}"
    checkpoint = torch.load(f"{dir}/{filename}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # args = checkpoint['args']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch + 1, args,model,scheduler,optimizer


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # utils_dir = args.utils
    # encoder= ImageEncoder().to(args.device)
    # train_dataset = DATA_BRAIN(train=True,r=int(args.patch_size/2), device=args.device)
    # features, exps = preprocess_dataset(train_dataset, encoder, args.device)
    # features_np = features.numpy()
    # np.savez('preprocessed_train.npz', features=features_np, exps=exps)
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # features, exps = preprocess_dataset(val_set, encoder, args.device)
    # features_np = features.numpy()
    
    # np.savez('preprocessed_val.npz', features=features_np, exps=exps)
    # data= np.load('./preprocess/preprocessed_train.npz')
    # print(features_np)
    # features = torch.from_numpy(data['features'])
    # exps=data['exps']
    # preprocessed_train_dataset = PreprocessedDataset(features, exps)

    # print(f'Using fold {args.fold}')
    # print(f'train: {len(preprocessed_train_dataset)}')
    # print(f'valid: {len(val_set)}')

    # dummy_dataset= Dummy(train=True)
    # batch_sampler = CustomBatchSampler(dummy_dataset, shuffle=True)
    # train_dataLoader = DataLoader(preprocessed_train_dataset, batch_size=args.batch_size, shuffle=batch_sampler,num_workers=args.n_workers,pin_memory=True)    
    
    model = WiKG(dim_in=args.embed_dim, dim_hidden=1024, topk=6, n_classes=args.n_classes, agg_type='bi-interaction', dropout=0.3, pool='mean').to(device)
    epoch =587
    filename=f"checkpoint_epoch_{epoch}.pth.tar"
    dir=f"{args.save_dir}model_result/{args.patch_size}"
    checkpoint = torch.load(f"{dir}/{filename}", map_location ='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    # data= np.load('./preprocess/preprocessed_val.npz')
    test_datasize=[]
    for i in NAMES:
        with open(f'./train_split/{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
            split_train_binary = pickle.load(f)
        
        test_datasize.append(len(split_train_binary)-sum(split_train_binary))
    data= np.load('D:/Downloads/crunch/WiKG/preprocessed_val.npz')
    hvg_pcc_list = []
    heg_pcc_list = []
    mse_list = []
    mae_list = []
    for i in range(len(test_datasize)):
        index_start = sum(test_datasize[:i])
        index_end = sum(test_datasize[:i + 1])
        features = torch.from_numpy(data['features'][index_start:index_end])
        exps=data['exps'][index_start:index_end]
        preprocessed_val_dataset = PreprocessedDataset(features, exps)
        val_loader = DataLoader(preprocessed_val_dataset, batch_size=4096, num_workers=args.n_workers, shuffle=False)
        output_dir = args.save_dir
        
        # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
        # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
        
        os.makedirs(output_dir, exist_ok=True)




        start_epoch= args.start_epoch
    
        print(start_epoch)
        print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
        pred, true = val_one_epoch(model=model, val_loader=val_loader, device=device, data_type='val')
        
        pred=pred.numpy()
        true= true.numpy()  
        tmp_list=[str(i) for i in range(460)]
        adata_true = anndata.AnnData(true)
        adata_pred = anndata.AnnData(pred)
        adata_pred.var_names = tmp_list
        adata_true.var_names = tmp_list
        gene_mean_expression = np.mean(adata_true.X, axis=0)
        top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
        top_50_genes_names = adata_true.var_names[top_50_genes_indices]
        top_50_genes_expression = adata_true[:, top_50_genes_names]
        top_50_genes_pred = adata_pred[:, top_50_genes_names]

        heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)
        hvg_pcc, hvg_p = get_R(adata_pred, adata_true)
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]
        
        heg_pcc_list.append(np.mean(heg_pcc))
        hvg_pcc_list.append(np.mean(hvg_pcc))

        mse = mean_squared_error(true, pred)
        mse_list.append(mse)
        print(f"Mean Squared Error (MSE) {NAMES[i]}: ", mse)
        mae = mean_absolute_error(true, pred)
        mae_list.append(mae)
        print(f"Mean Absolute Error (MAE) {NAMES[i]}: ", mae)
        print(f"avg heg pcc {NAMES[i]}: {np.mean(heg_pcc):.4f}")
        print(f"avg hvg pcc {NAMES[i]}: {np.mean(hvg_pcc):.4f}")

    print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
    print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
    print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
    print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
            
                

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)