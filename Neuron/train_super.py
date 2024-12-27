import sys
import os
import argparse
import csv
import numpy as np
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel,GATModel_Softmax,GATModel_3,TransConv
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

from utils import get_R


def train_one_epoch(model,args, train_loader, optimizer,scheduler, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')
    # optimizer.zero_grad()
    optimizer.zero_grad()
    accumulation_steps = 1  # Number of steps to accumulate gradients
    if args.nolog1p ==False:
        loss_function= F.mse_loss
    else:
        loss_function=F.cross_entropy
    for i, data in enumerate(train_loader):
        # graph_data= build_batch_graph(data,device,centroid_layer=centroid_layer)
        # data=None
        # pred,label,pred_c,label_c = model(graph_data)
        if i!=-1: #for testing manual
            
            # data = data.to(device)
            
            graph_data= build_super_batch_graph(data,device)
            data=None
            # print(graph_data.exps.shape)
            # excep?t:
                # print(data)
            # graph_data.cpu()
            # data.cpu()
            pred,label,pred_c,label_c,_ = model(graph_data)
            
            label = np.array(label, dtype=np.float32)
            label = torch.from_numpy(label)
            label= label.to(device)
            # print(label.shape,pred.shape)

            # print(type(label))
            if pred_c is not None:
                label_c = np.array(label_c, dtype=np.float32)
                label_c= torch.from_numpy(label_c)
                label_c= label_c.to(device)
                loss = loss_function(pred, label)+ loss_function(pred_c, label_c)
            else:
                loss = loss_function(pred, label)

            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()  # Perform optimizer step
                scheduler.step()  # Adjust learning rate
                optimizer.zero_grad()
            
            # optimizer.step()
            # scheduler.step()
            # optimizer.zero_grad()
            
            total_loss = (total_loss * i + loss.detach()) / (i + 1)
            if args.nolog1p== False:
                train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 4))
            else:
                pred= torch.log1p(pred*100)
                label= torch.log1p(label*100)
                mse_loss= F.mse_loss(pred, label)
                train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {} \tloss_mse{}'.format(epoch, optimizer.param_groups[0]["lr"]
                                                                                            ,round(total_loss.item(), 4)
                                                                                             ,round(mse_loss.item(), 4))
            # if i==3 and demo==True:
        #     break
    torch.cuda.empty_cache()
    return pred


@torch.no_grad()
def val_one_epoch(model, val_loader, device,args, centroid, data_type='val'):
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
        if args.encoder_mode ==True:
            centroid=0
        output,label,_,_,_= model(graph_data)
        head= min(centroid,len(data))
        output[output < 0] = 0
        output= output[head:]
        label= label[head:]
        
        label = torch.from_numpy(label).to(device)
        if args.nolog1p == True:
            output= torch.log1p(output*100)
            label= torch.log1p(label*100)
        labels = torch.cat([labels.cpu(), label.cpu()], dim=0)
        preds = torch.cat([preds.cpu(), output.detach().cpu()], dim=0)
        # if i==3 and demo==True:
        #     break
    print(labels.shape,preds.shape)
    return preds.cpu(), labels.cpu()



def parse():
    parser = argparse.ArgumentParser('Training for NNeuron')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=150, help='patch_size')

    parser.add_argument('--embed_dir', type=str, default='/content/preprocessed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=460)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_dir', default='./model_result', help='path where to save')
    parser.add_argument('--encoder_mode', default=False, type=bool, help='test encoder')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')
    parser.add_argument('--demo', default=False, type=bool, help='toy run')
    parser.add_argument('--local', default=False, type=bool, help='toy run')
    parser.add_argument('--centroid_layer', default=False, type=bool, help='add layer')
    parser.add_argument('--nolog1p', default=False, type=bool, help='no log1p in dataset')
    parser.add_argument('--partial', default=-1, type=int, help='leave-one-out training')
    parser.add_argument('--input_dim', default=1024, type=int, help='input dimmension')

    return parser.parse_args()

def save_checkpoint(epoch, model, optimizer,scheduler, args, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args
    }
    dir=f"{args.save_dir}"
    if args.encoder_mode== True:
        dir=f"{args.save_dir}/encoder"
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, f"{dir}/{filename}")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(epoch, model, optimizer,scheduler,args):
    filename=f"checkpoint_epoch_{epoch}.pth.tar"
    dir=f"{args.save_dir}"
    if args.encoder_mode== True:
        dir=f"{args.save_dir}/encoder"
    checkpoint = torch.load(f"{dir}/{filename}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # args = checkpoint['args']
    scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch + 1, args,model,scheduler,optimizer


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    print( f'demo: {args.demo== True}  encoder: {args.encoder_mode== True}')
    
    NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    # NAMES=NAMES[:1]
    
    # train_NAMES= NAMES[:3]+ NAMES[5:]
    train_NAMES= NAMES
    val_NAMES=NAMES
    if args.demo== True:
        NAMES=NAMES[:1]
        train_NAMES=NAMES
        val_NAMES=NAMES
    if args.partial>=0:
        # NAMES=NAMES[1]
        index=args.partial
        train_NAMES=NAMES[:index]+NAMES[index+1:]
        val_NAMES=[NAMES[index]]
    dir=args.embed_dir
    # dir='D:/DATA/Gene_expression/Crunch/preprocessed'
    if args.local == True:
        pin= False
    else:
        pin= True
    traindata= SuperNeuronData(emb_folder=dir
                            ,train=True
                            , split =True
                          ,name_list= train_NAMES
                          ,encoder_mode=args.encoder_mode
                          ,nolog1p= args.nolog1p
                          )
    train_dataLoader =DataLoader(traindata, batch_size=args.batch_size, shuffle=False,pin_memory=pin)    
    # print(len(train_dataLoader))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # traindata[2127]
    model=GATModel_3(centroid_layer=args.centroid_layer,input_dim=args.input_dim)
    model= model.to(device)
    #------------------------
    

    # print(f'Using fold {args.fold}')
    # print(f'valid: {len(val_set)}')

    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.epochs
                             ,base_lr=0.00018
                             ,iter_per_epoch = len(train_dataLoader)
                             ,warmup_epochs= 10
                            ,warmup_lr= 0.00015
                            ,final_lr= 0.00005
                            ,constant_predictor_lr=False
)
    
    val_set= [SuperNeuronData(emb_folder=dir
                           ,train=False
                           , split =True
                           ,name_list= [name]
                           ,nolog1p=args.nolog1p
                           ,encoder_mode=args.encoder_mode) 
              for name in val_NAMES]
    val_loader =[DataLoader(set, batch_size=args.batch_size, shuffle=False,pin_memory=pin)for set in val_set]    
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
    if start_epoch >0:
        start_epoch, args, model,scheduler, optimizer = load_checkpoint(start_epoch, model, optimizer,scheduler,args)
    # for step in range(start_epoch*len(train_dataLoader)):
    #     scheduler.step()
    print(start_epoch)
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        checkpoint_filename = f"checkpoint_best_epoch_{epoch}.pth.tar"
        train_logits = train_one_epoch(model=model, train_loader=train_dataLoader
                                       ,args=args
                                      , optimizer=optimizer
                                      ,scheduler=scheduler
                                      , device=device, epoch=epoch + 1
                                      )
        if (epoch+1)%8 ==0: 
            hvg_pcc_list = []
            heg_pcc_list = []
            mse_list = []
            mae_list = []
            for index in range(len(val_loader)): 
                val_preds, val_labels = val_one_epoch(model=model
                                                      , val_loader=val_loader[index]
                                                      , device=device, data_type='val'
                                                      , centroid= args.batch_size
                                                      ,args=args
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
            
                print(f'name: {val_NAMES[index]}')
                print('Val\t[epoch {}] mse:{}\tmae:{}\theg:{} \thevg:{}'.format(epoch + 1, mse, mae,np.mean(heg_pcc),np.mean(hvg_pcc)))
          
            print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
            print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
            print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
            print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
            min_val_mse = min(min_val_mse, np.mean(mse_list))
            min_val_mae = min(min_val_mae, np.mean(mae_list))
       
            if min_val_mae == np.mean(mse_list) and epoch>10:
                print('best mse found... save best acc weights...')
                
                save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)
            with open(f'{output_dir}/results.csv', 'a') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch+1,  np.mean(mse_list),np.mean(mae_list), np.mean(heg_pcc_list),np.mean(hvg_pcc_list)])
        if (epoch+1)%4==0:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
            save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)

        
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)