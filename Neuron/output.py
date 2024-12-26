import sys
import os
import argparse
import csv
import numpy as np
from dataset import NeuronData,build_batch_graph
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel_3
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import NeuronData_3
from torch.utils.data import Sampler
from collections import defaultdict
from tqdm import tqdm
# torch.multiprocessing.set_start_method('spawn')
import pandas as pd
import itertools
import spatialdata as sd


@torch.no_grad()
def val_one_epoch(model, val_loader, device, centroid,demo=False, encoder_mode =False, data_type='val'):
    model.eval()
    labels = torch.tensor([], device=device)
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
        head= min(centroid,len(data))
        output[output < 0] = 0
        output= output[head:]
        label= label[head:]
        
        label = torch.from_numpy(label).to(device)
        labels = torch.cat([labels.cpu(), label.cpu()], dim=0)
        preds = torch.cat([preds.cpu(), output.detach().cpu()], dim=0)
       
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

    model=GATModel_3()
    model= model.to(device)

    start_epoch= args.start_epoch
    model = load_checkpoint(args.start_epoch, model,args=args)
    
    val_set= [NeuronData_3(emb_folder=dir,train=False, split =True,name_list= [name],encoder_mode=args.encoder_mode) 
              for name in NAMES]
    val_loader =[DataLoader(set, batch_size=args.batch_size, shuffle=False,pin_memory=False)for set in val_set]    
    output_dir = args.save_dir
    
    # val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    # val_loader = DataLoader(val_set, batch_size=1024, num_workers=args.n_workers, shuffle=False)
    

    os.makedirs(output_dir,exist_ok=True)
    print(f"Start training for {args.epochs} epochs")
    data_path= 'F:/Data/crunch_large/submit/data'
    gene_names = sd.read_zarr(f'{data_path}/DC5.zarr')["anucleus"].var.index
    
    
    print(start_epoch)
    
    all_predictions=[]
    for index in range(len(val_loader)): 
        pred, _ = val_one_epoch(model=model,demo=args.demo
                                                , val_loader=val_loader[index]
                                                , device=device, data_type='val'
                                                , centroid= args.batch_size
                                                ,encoder_mode=args.encoder_mode)
        
        cell_ids = []
        for idx in range(len(val_set[index])):
            data = val_set[index][idx]  # Access the data at index `idx`
            cell_ids.extend(data.cell_ids.tolist())
        pred = np.round(pred, 2)
        prediction = pd.DataFrame(
            itertools.product(cell_ids, gene_names),
            columns=["cell_id", "gene"]
        )
        prediction["prediction"] = pred.numpy().ravel(order="C")
        prediction["slide_name"] = NAMES[index]  # Add the slide name as a new column
        
        all_predictions.append(prediction)

    # Combine predictions for all slides
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"Predictions saved to {output_dir}/predictions.csv")
        
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)