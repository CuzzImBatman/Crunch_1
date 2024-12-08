import sys
import os
import argparse
import csv
import numpy as np
from dataset import NeuronData,build_batch_graph
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel,GATModel_3
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import NeuronData,NeuronData_3
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


def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch,demo=False):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')
    # optimizer.zero_grad()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # data = data.to(device)
        graph_data= build_batch_graph(data,device)
        # graph_data.cpu()
        # data.cpu()
        pred,label = model(graph_data)
        label = np.array(label, dtype=np.float32)
        label = torch.from_numpy(label)
        label= label.to(device)
        # print(label.shape,pred.shape)

        # print(type(label))

        loss = F.mse_loss(pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))
        if i==3 and demo==True:
            break
    torch.cuda.empty_cache()
    return pred


@torch.no_grad()
def val_one_epoch(model, val_loader, device, centroid,demo=False, encoder_mode =False, data_type='val'):
    model.eval()
    preds = torch.tensor([], device=device)
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='green')

    for i, data in enumerate(val_loader):
        # data = data.to(device)
        graph_data= build_batch_graph(data,device,centroid_layer=False)
        # data.cpu()
        # graph_data.cpu()
        if encoder_mode ==True:
            centroid=0
        output,label,_,_ = model(graph_data)
        output= output[centroid:]
        
        preds = torch.cat([preds.cpu(), output.detach().cpu()], dim=0)
        if i==3 and demo==True:
            break
    print(preds.shape)
    return preds.cpu()



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
    

    return parser.parse_args()


def load_checkpoint(epoch, model, optimizer,scheduler,args):
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
    print(args.demo,args.encoder_mode, f'demo: {args.demo== True}  encoder: {args.encoder_mode== True}')
    
    utils_dir = args.utils
    NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    # NAMES=NAMES[:1]
    
    if args.demo== True:
        NAMES=NAMES[:1]
    dir=args.embed_dir
    # dir='D:/DATA/Gene_expression/Crunch/preprocessed'
    # traindata= NeuronData(emb_folder=dir,train=True, split =True,name_list= train_NAMES,encoder_mode=args.encoder_mode)
    # train_dataLoader =DataLoader(traindata, batch_size=args.batch_size, shuffle=False,pin_memory=False)    
    # print(len(train_dataLoader))
    # traindata[2127]
    model=GATModel_3()
    model= model.to(device)
    #------------------------
    

    # print(f'Using fold {args.fold}')
    # print(f'valid: {len(val_set)}')

    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.epochs
                             ,base_lr=0.001
                             ,iter_per_epoch = 1
                             ,warmup_epochs= 10
                            ,warmup_lr= 0.0003
                            ,final_lr= 0.00001
                            ,constant_predictor_lr=False
)
    
    val_set= [NeuronData_3(emb_folder=dir,train=False, split =False,name_list= [name],encoder_mode=args.encoder_mode) 
              for name in NAMES]
    val_loader =[DataLoader(set, batch_size=args.batch_size, shuffle=False,pin_memory=True)for set in val_set]    
    output_dir = args.save_dir
    
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)



    with open(f'{output_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{output_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

   
    start_epoch= args.start_epoch
    start_epoch, args, model,scheduler, optimizer = load_checkpoint(args.epochs, model, optimizer,scheduler,args)
    
    print(start_epoch)
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    
   
    for index in range(len(val_loader)): 
        val_preds = val_one_epoch(model=model,demo=args.demo
                                                , val_loader=val_loader[index]
                                                , device=device, data_type='val'
                                                , centroid= args.batch_size
                                                ,encoder_mode=args.encoder_mode)
        
    
        print(f'name: {NAMES[index]}')
    # print(f"avg heg pcc : {np.mean(heg_pcc):.4f}")
    # print(f"avg hvg pcc: {np.mean(hvg_pcc):.4f}")
    
    
        

        
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)