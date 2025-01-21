import sys
import os
import argparse
import csv
import numpy as np
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel_3,GATModel_5,GATModel_TRANS
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import SuperNeuronData,build_super_batch_graph
from lr_scheduler import LR_Scheduler
from torch.utils.data import Sampler
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
# torch.multiprocessing.set_start_method('spawn')
import anndata
import pickle
from utils import get_R
from scipy.stats import pearsonr,spearmanr



@torch.no_grad()
def val_one_epoch(model, val_loader, device, centroid,demo=False, encoder_mode =False, data_type='val',choosen_mask=None):
    model.eval()
    labels = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='green')

    for i, data in enumerate(val_loader):
        # data = data.to(device)
        graph_data= build_super_batch_graph(data,device)
        # data.cpu()
        # graph_data.cpu()
        if encoder_mode ==True:
            centroid=0
        output,label,_,_,centroid_index,edge_index = model(graph_data)
        head= min(centroid,len(data))
        min_bound=0.23
        min_bound=0
        
        # choosen_mask = [ 36,  37, 100, 172, 375, 453]
        # output[:,choosen_mask]=0
        output[output < min_bound] = 0
        # output[(output < 0.19) & (output>0.1)] = 0.1
        # output[(output < min_bound) & (output >0.2)] = 0.2
        
        mask = np.ones(output.shape[0], dtype=bool)
        mask[centroid_index] = False
        mask[:head]=False
        ''''''
        edge_index=edge_index.T
        # min_bound=0.2
        centroid_index=torch.tensor(centroid_index, device=edge_index.device)
        # for centroid in centroid_index:
        #     # Find cells connected to the current centroid using edge_index
        #     is_connected = edge_index[0] == centroid
        #     connected_cells = edge_index[1][is_connected]

        #     # Create a mask for genes in the current centroid below the threshold
        #     # gene_mask = output[centroid] < min_bound

        #     # Zero out these genes for the centroid
        #     # output[centroid, gene_mask] = 0
        #     # print(centroid,is_connected)

        #     # Zero out the same genes for all connected cells
        #     # output[connected_cells, :] *= ~gene_mask
        #     output[connected_cells] = output[connected_cells]+ output[centroid]/(0.9*len(connected_cells))
        ''''''
        output[output < min_bound] = 0
        # mask = np.zeros(output.shape[0], dtype=bool)

        
        # print(centroid_index,head)
        output= output[mask]
        label= label[mask]
        # print(label[0])
        
        label = torch.from_numpy(label).to(device)
        labels = torch.cat([labels.cpu(), label.cpu()], dim=0)
        preds = torch.cat([preds.cpu(), output.detach().cpu()], dim=0)
        preds = torch.round(preds * 100) / 100
        # labels = torch.round(labels * 100) / 100

        # if i==3 and demo==True:
        #     break
    print(labels.shape,preds.shape)
    return preds.cpu(), labels.cpu()



def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=150, help='patch_size')

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
    parser.add_argument('--encoder_mode', default=False, type=bool, help='test encoder')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')
    parser.add_argument('--demo', default=False, type=bool, help='toy run')
    parser.add_argument('--input_dim', default=1024, type=int, help='input dimmension')

    return parser.parse_args()



def load_checkpoint(epoch, model,args):
    try:
        filename=f"checkpoint_epoch_{epoch}.pth.tar"
        dir=f"{args.save_dir}"
        checkpoint = torch.load(f"{dir}/{filename}")
    except:
        filename=f"checkpoint_epoch_best_{epoch}.pth.tar"
        dir=f"{args.save_dir}"
        checkpoint = torch.load(f"{dir}/{filename}")
    if args.encoder_mode== True:
        dir=f"{args.save_dir}"
    model.load_state_dict(checkpoint['model_state_dict'])
    # epoch = checkpoint['epoch']
    # args = checkpoint['args']
    print(f"Checkpoint loaded from epoch {epoch}")
    return model

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    print(f'demo: {args.demo== True}  encoder: {args.encoder_mode== True}')
    
    NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    
    if args.demo== True:
        NAMES=NAMES[:1]
    dir=args.embed_dir
    with open(f'E:/DATA/crunch/resources/indices_test_list_7.pkl','rb') as f:
        indices_test_list=pickle.load(f)
    model=GATModel_TRANS(input_dim=args.input_dim)
    model= model.to(device)

    start_epoch= args.start_epoch
    model = load_checkpoint(args.start_epoch, model,args=args)
    
    val_set= [SuperNeuronData(emb_folder=dir,train=False, split =True,name_list= [name],encoder_mode=args.encoder_mode) 
              for name in NAMES]
    val_loader =[DataLoader(set, batch_size=args.batch_size, shuffle=False,pin_memory=True)for set in val_set]    
    output_dir = args.save_dir
    
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)


    print(f"Start training for {args.epochs} epochs")

    

  
    
    
    print(start_epoch)
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    
    hvg_pcc_list = []
    hvg_pcc_rv_list = []
    heg_pcc_list = []
    mse_list = []
    mae_list = []
    
    for index in range(len(val_loader)): 
        val_preds, val_labels = val_one_epoch(model=model,demo=args.demo
                                                , val_loader=val_loader[index]
                                                , device=device, data_type='val'
                                                , centroid= args.batch_size
                                                ,encoder_mode=args.encoder_mode
                                                ,choosen_mask= indices_test_list[index][0])
        mse=mean_squared_error(val_labels, val_preds)
        mae=mean_absolute_error(val_labels, val_preds)
        ###############
        tmp_list=[str(i) for i in range(460)]
        # tmp_list=[str(i) for i in range(val_labels.shape[0])]
        val_labels=val_labels.numpy()
        val_preds= val_preds.numpy() 
        dim=0
        adata_true = anndata.AnnData(val_labels)
        adata_pred = anndata.AnnData(val_preds)
        adata_pred.var_names = tmp_list
        adata_true.var_names = tmp_list
        gene_mean_expression = np.mean(adata_true.X, axis=0)
        top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
        top_50_genes_names = adata_true.var_names[top_50_genes_indices]
        top_50_genes_expression = adata_true[:, top_50_genes_names]
        top_50_genes_pred = adata_pred[:, top_50_genes_names]

        heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression,dim=0)
        hvg_pcc, hvg_p = get_R(adata_pred, adata_true,dim=0)
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]
        
        heg_pcc_list.append(np.mean(heg_pcc))
        hvg_pcc_list.append(np.mean(hvg_pcc))
        mse_list.append(mse)
        mae_list.append(mae)
    
        print(f'name: {NAMES[index]}')
        print('Val] mse:{}\tmae:{}\theg:{} \thevg:{}'.format(mse, mae,np.mean(heg_pcc),np.mean(hvg_pcc)))
    
    print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
    print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
    print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
    print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
    # min_val_mse = min(min_val_mse, np.mean(mse_list))
    # min_val_mae = min(min_val_mae, np.mean(mae_list))
    

    
        

        
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)