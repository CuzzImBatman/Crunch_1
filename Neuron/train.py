import sys
import os
import argparse
import csv
import numpy as np
from dataset import NeuronData,build_batch_graph
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import NeuronData
from lr_scheduler import LR_Scheduler
from torch.utils.data import Sampler
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
# torch.multiprocessing.set_start_method('spawn')
import anndata

from utils import get_R


def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # data = data.to(device)
        graph_data= build_batch_graph(data,device)
        # graph_data.cpu()
        # data.cpu()
        pred,label = model(graph_data)
        label = torch.from_numpy(label.astype(np.float32)).to(device)
        # print(label.shape,pred.shape)
      
        loss = F.mse_loss(pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))

    torch.cuda.empty_cache()
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

    for i, data in enumerate(val_loader):
        # data = data.to(device)
        graph_data= build_batch_graph(data,device)
        # data.cpu()
        # graph_data.cpu()
        output,label = model(graph_data)
        label = torch.from_numpy(label).to(device)
        labels = torch.cat([labels.cpu(), label.cpu()], dim=0)
        preds = torch.cat([preds.cpu(), output.detach().cpu()], dim=0)

    return preds.cpu(), labels.cpu()



def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=200, help='patch_size')

    parser.add_argument('--embed_dir', type=str, default='/content/preprocessed')
    # parser.add_argument('--patch_size', type=int, default=112, help='patch_size')
    parser.add_argument('--utils', type=str, default=None, help='utils path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=460)
    parser.add_argument('--fold', type=int, default=0)
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

    utils_dir = args.utils
    NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    # NAMES=NAMES[:1]
    dir=args.embed_dir
    # dir='D:/DATA/Gene_expression/Crunch/preprocessed'
    traindata= NeuronData(emb_folder=dir,train=True, split =True,name_list= NAMES)
    train_dataLoader =DataLoader(traindata, batch_size=args.batch_size, shuffle=False,pin_memory=False)    
    # print(len(train_dataLoader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # traindata[2127]
    model=GATModel()
    model= model.to('cuda')
    #------------------------
    

    # print(f'Using fold {args.fold}')
    # print(f'valid: {len(val_set)}')

    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.epochs
                             ,base_lr=0.001
                             ,iter_per_epoch = len(train_dataLoader)
                             ,warmup_epochs= 10
                            ,warmup_lr= 0.0003
                            ,final_lr= 0.00001
                            ,constant_predictor_lr=False
)
    
    val_set= NeuronData(emb_folder=dir,train=False, split =True,name_list= NAMES)
    val_loader =DataLoader(val_set, batch_size=100, shuffle=False,pin_memory=True)    
    output_dir = args.save_dir
    
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)


    print(f"Start training for {args.epochs} epochs")

    with open(f'{output_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{output_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

    min_val_mse = 200.0
    min_val_mae = 200.0
    start_epoch= args.start_epoch
    # start_epoch, args, model,scheduler, optimizer = load_checkpoint(463, model, optimizer,scheduler,args)
    for step in range(start_epoch*len(train_dataLoader)):
        scheduler.step()
    print(start_epoch)
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        checkpoint_filename = f"checkpoint_best_epoch_{epoch}.pth.tar"
        train_logits = train_one_epoch(model=model, train_loader=train_dataLoader, optimizer=optimizer,scheduler=scheduler, device=device, epoch=epoch + 1)
        if (epoch+1)%2 ==0: 
            val_preds, val_labels = val_one_epoch(model=model, val_loader=val_loader, device=device, data_type='val')
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
            
            # print(f"avg heg pcc : {np.mean(heg_pcc):.4f}")
            # print(f"avg hvg pcc: {np.mean(hvg_pcc):.4f}")
            
            
            print('Val\t[epoch {}] mse:{}\tmae:{}\theg:{} \thevg:{}'.format(epoch + 1, mse, mae,np.mean(heg_pcc),np.mean(hvg_pcc)))
        
        
            min_val_mse = max(min_val_mse, mse)
            min_val_mae = max(min_val_mae, mae)
       
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