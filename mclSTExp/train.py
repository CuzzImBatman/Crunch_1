import argparse
import torch
import os
from dataset import DATA_BRAIN
from model import mclSTExp_Attention
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--max_epochs', type=int, default=90, help='')#90 
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--dim', type=int, default=460, help='spot_embedding dimension (# HVGs)')  
    parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
    parser.add_argument('--heads_num', type=int, default=8, help='attention heads num')
    parser.add_argument('--heads_dim', type=int, default=64, help='attention heads dim')
    parser.add_argument('--heads_layers', type=int, default=2, help='attention heads layer num')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    parser.add_argument('--dataset', type=str, default='crunch', help='dataset')  
    parser.add_argument('--encoder_name', type=str, default='densenet121', help='image encoder')
    parser.add_argument('--path_save', type=str, default='.', help='model saved path')
    parser.add_argument('--resume', type=str, default=False, help='resume training')

    args = parser.parse_args()
    return args
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random


class CustomBatchSampler:
    def __init__(self, dataset, shuffle=True):
        # Group items by ID
        self.groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            self.groups[item['id']].append(idx)
        
        # Create a list of groups (each group corresponds to one unique ID)
        self.group_keys = list(self.groups.keys())
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.group_keys)  # Shuffle the groups by ID

        # Yield batches of indices corresponding to each ID
        for key in self.group_keys:
            yield self.groups[key]

    def __len__(self):
        return len(self.group_keys)
# Create a simple dataset


def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position" or k == "id"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def load_data(args):
    
        print(f'load dataset: {args.dataset}')
        train_dataset = DATA_BRAIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = DATA_BRAIN(train=False, fold=args.fold)
        # return train_dataLoader
        return train_dataLoader, test_dataset



def save_model(args, model, test_dataset=None, examples=[]):
  
        os.makedirs(f"{args.path_save}/model_result/{args.dataset}", exist_ok=True)
        torch.save(model.state_dict(),
                   f"{args.path_save}/model_result/{args.dataset}/best_{args.fold}.pt")


def save_checkpoint(epoch, model, optimizer, args, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch + 1, args
def main():
    args = generate_args()
    
    args.fold = 0
    
    train_dataLoader,test_dataLoader = load_data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mclSTExp_Attention(encoder_name=args.encoder_name,
                                spot_dim=args.dim,
                                temperature=args.temperature,
                                image_dim=args.image_embedding_dim,
                                projection_dim=args.projection_dim,
                                heads_num=args.heads_num,
                                heads_dim=args.heads_dim,
                                head_layers=args.heads_layers,
                                dropout=args.dropout)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-3
    )
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, args = load_checkpoint(args, model, optimizer)
    
    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        train(model, train_dataLoader, optimizer, epoch)
        
        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
        save_checkpoint(epoch, model, optimizer, args, filename=checkpoint_filename)
    # for epoch in range(args.max_epochs):
    #     model.train()
    #     train(model, train_dataLoader, optimizer, epoch)

    
    save_model(args, model)
    print("Saved Model")


if __name__ == '__main__':
    main()
