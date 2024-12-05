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
from dataset import Dummy,CLUSTER_BRAIN
from pathlib import Path
from lr_scheduler import LR_Scheduler
from torch.utils.data import Sampler
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import ImageEncoder
import pandas as pd
# torch.multiprocessing.set_start_method('spawn')
class CustomBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.groups = defaultdict(list)

        # Group items by ID
        for idx, item in enumerate(dataset):
            self.groups[item['id']].append(idx)

    def __iter__(self):
        group_keys = list(self.groups.keys())
        
        # Shuffle group order if shuffle is enabled
        if self.shuffle:
            random.shuffle(group_keys)
        
        # For each group ID, shuffle items within the group if shuffle is enabled
        shuffled_indices = []
        for key in group_keys:
            indices = self.groups[key][:]
            if self.shuffle:
                random.shuffle(indices)  # Shuffle items within each group ID
            shuffled_indices.extend(indices)
        
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.dataset)




def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        label= batch['expression']
        label = np.array(label, dtype=np.float32)
        label = torch.from_numpy(label).squeeze(1)
        batch = batch
        label = label.to(device)
        pred = model(batch)
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

    for i, batch in enumerate(val_loader):
        label= batch['expression']
        label = np.array(label, dtype=np.float32)
        label = torch.from_numpy(label).squeeze(1)
        batch = batch
        label = label.to(device)
        output = model(batch)
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
    parser.add_argument('--patch_size', type=int, default=80, help='patch_size')
    parser.add_argument('--utils', type=str, default=None, help='utils path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=460)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')
    parser.add_argument('--embed_dir', type=str, default='/content/preprocessed')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--encoder_mode', type=bool, default=False)

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
    dir=f"{args.save_dir}/model_result/{args.patch_size}"
    checkpoint = torch.load(f"{dir}/{filename}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # args = checkpoint['args']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch + 1, args,model,scheduler,optimizer
import anndata
from utils import get_R

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    utils_dir = args.utils
    # encoder= ImageEncoder().to(args.device)
    # train_dataset = DATA_BRAIN(train=True,r=int(args.patch_size/2), device=args.device)
    # features, exps = preprocess_dataset(train_dataset, encoder, args.device)
    # features_np = features.numpy()
    # np.savez('preprocessed_train.npz', features=features_np, exps=exps)
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # features, exps = preprocess_dataset(val_set, encoder, args.device)
    # features_np = features.numpy()
    
    # np.savez('preprocessed_val.npz', features=features_np, exps=exps)
    # print(features_np)
    dir=args.embed_dir
    NAMES=['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    if args.demo== True:
        NAMES=NAMES[:2]
    train_dataset = CLUSTER_BRAIN(emb_folder=dir,train=True,split=True,name_list=NAMES)
    # print(f'Using fold {args.fold}')
    print(f'train: {len(train_dataset)}')
    # print(f'valid: {len(val_set)}')
    val_dataset = [CLUSTER_BRAIN(emb_folder=dir,train=False,split=True,name_list=[name] ) for name in NAMES]
    
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=3,pin_memory=True)
    val_loader = [DataLoader(v_set, batch_size=args.batch_size
                             , shuffle=False,num_workers=3,pin_memory=True )
                  for v_set in val_dataset]    
    if args.local ==True:
        print('local run')
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=False)
        val_loader = [DataLoader(v_set, batch_size=args.batch_size, shuffle=False,pin_memory=False )
                for v_set in val_dataset]    
    output_dir = args.save_dir
    
    
    model = WiKG(dim_in=args.embed_dim, dim_hidden=1024, topk=6, n_classes=args.n_classes, agg_type='bi-interaction', dropout=0.3, pool='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.epochs
                             ,base_lr=0.001
                             ,iter_per_epoch = len(train_dataLoader)
                             ,warmup_epochs= 10
                            ,warmup_lr= 0.0003
                            ,final_lr= 0.00001
                            ,constant_predictor_lr=False
)
    
    
    
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)


    print(f"Start training for {args.epochs} epochs")

    with open(f'{output_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{output_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

    
    start_epoch= args.start_epoch
    # start_epoch, args, model,scheduler, optimizer = load_checkpoint(463, model, optimizer,scheduler,args)
    for step in range(start_epoch*len(train_dataLoader)):
        scheduler.step()
    print(start_epoch)
    min_val_mse = 200.0
    min_val_mae = 200.0
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        checkpoint_filename = f"checkpoint_best_epoch_{epoch}.pth.tar"
        train_logits = train_one_epoch(model=model, train_loader=train_dataLoader, optimizer=optimizer,scheduler=scheduler, device=device, epoch=epoch + 1)
        if (epoch+1)%2 ==0: 
            hvg_pcc_list = []
            heg_pcc_list = []
            mse_list = []
            mae_list = []
            for index in range(len(val_loader)): 
                val_preds, val_labels = val_one_epoch(model=model
                                                      , val_loader=val_loader[index]
                                                      , device=device
                                                      
                                                     )
                mse=mean_squared_error(val_labels, val_preds)
                mae=mean_absolute_error(val_labels, val_preds)
                ###############
                tmp_list=[str(i) for i in range(460)]
                val_labels=val_labels.numpy()
                val_preds= val_preds.numpy() 
                adata_true = anndata.AnnData(val_labels)
                adata_pred = anndata.AnnData(val_preds)
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
                mse_list.append(mse)
                mae_list.append(mae)
            
                print(f'name: {NAMES[index]}')
                print('Val\t[epoch {}] mse:{}\tmae:{}\theg:{} \thevg:{}'.format(epoch + 1, mse, mae,np.mean(heg_pcc),np.mean(hvg_pcc)))
            # print(f"avg heg pcc : {np.mean(heg_pcc):.4f}")
            # print(f"avg hvg pcc: {np.mean(hvg_pcc):.4f}")
            print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
            print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
            print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
            print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
            min_val_mse = min(min_val_mse, np.mean(mse_list))
            min_val_mae = min(min_val_mae, np.mean(mae_list))
       
            if min_val_mae == mse and epoch>30:
                print('best mse found... save best acc weights...')
                
                save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)
            with open(f'{output_dir}/results.csv', 'a') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch+1, mse,mae])
        if (epoch+1)%4==0:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
            save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)

        
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)
