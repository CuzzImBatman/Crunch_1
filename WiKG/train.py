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
from dataset import DATA_BRAIN,Dummy
from pathlib import Path
from lr_scheduler import LR_Scheduler
from torch.utils.data import Sampler
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd

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




def train_one_epoch(model, train_loader, optimizer, device, epoch):
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


def cal_metrics(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    # accuracy
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())

    # macro-average area under the cureve (AUC) scores
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())

    # weighted f1-score
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')

    # quadratic weighted Kappa
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')

    # macro specificity 
    specificity_list = []
    for class_idx in range(num_classes):
        true_positive = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() == class_idx))
        true_negative = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() != class_idx))
        false_positive = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() == class_idx))
        false_negative = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() != class_idx))

        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)

    macro_specificity = np.mean(specificity_list)

    # confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    return accuracy, auc, f1, kappa, macro_specificity, confusion_mat



def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512, help='patch_size')

    parser.add_argument('--embed_dim', type=int, default=1024, help="The dimension of instance-level representations")
    parser.add_argument('--patch_size', type=int, default=100, help='patch_size')
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
    dir=f"{args.path_save}/model_result/{args.patch_size}"
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, f"{dir}/{filename}")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(epoch, model, optimizer,scheduler,args):
    filename=f"checkpoint_epoch_{epoch}.pth.tar"
    dir=f"{args.path_save}/model_result/{args.patch_size}"
    checkpoint = torch.load(f"{dir}/{filename}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
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

    train_dataset = DATA_BRAIN(train=True,r=int(args.patch_size/2), device=args.device)
        

    # print(f'Using fold {args.fold}')
    print(f'train: {len(train_dataset)}')
    # print(f'valid: {len(val_set)}')

    dummy_dataset= Dummy(train=True)
    batch_sampler = CustomBatchSampler(dummy_dataset, shuffle=True)
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=batch_sampler,num_workers=0,pin_memory=True)    
    
    
    model = WiKG(dim_in=args.embed_dim, dim_hidden=512, topk=6, n_classes=args.n_classes, agg_type='bi-interaction', dropout=0.3, pool='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=1e-5)
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.epochs
                             ,base_lr=0.2
                             ,iter_per_epoch = len(train_dataLoader)
                             ,warmup_epochs= 20
                            ,warmup_lr= 0.1
                            ,final_lr= 0.015
                            ,constant_predictor_lr=False
)
    
    
    output_dir = args.save_dir
    
    val_set = DATA_BRAIN(train=False,r=int(args.patch_size/2), device=args.device)
    val_loader = DataLoader(val_set, batch_size=50, num_workers=0, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)


    print(f"Start training for {args.epochs} epochs")

    with open(f'{output_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{output_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

    max_val_mse = 0.0
    max_val_auc = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_logits = train_one_epoch(model=model, train_loader=train_dataLoader, optimizer=optimizer, device=device, epoch=epoch + 1)
        if (epoch+1)%2 ==0: 
            val_preds, val_labels = val_one_epoch(model=model, val_loader=val_loader, device=device, data_type='val')
            mse=mean_squared_error(val_labels, val_preds)
            mae=mean_absolute_error(val_labels, val_preds)
            print('Val\t[epoch {}] mse:{}\tmae:{}'.format(epoch + 1, mse, mae))
        
        
            max_val_mse = max(max_val_mse, mse)
            max_val_mae = max(max_val_mae, mae)
       
            if max_val_mse == mse and epoch>30:
                print('best mse found... save best acc weights...')
                checkpoint_filename = f"checkpoint_best_epoch_{epoch}.pth.tar"
                save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)
            with open(f'{output_dir}/results.csv', 'a') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch+1, mse,mae])
        if (epoch+1)%4==0:
            save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)

        
            

if __name__ == '__main__':
    opt = parse()
    main(opt)