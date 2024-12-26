import argparse
import torch
import os
from dataset import DATA_BRAIN,Dummy,CLUSTER_BRAIN
from model import mclSTExp_Attention, mclSTExp_Attention_Pretrain
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr
from torch.utils.data import Sampler
from collections import defaultdict
import random
import torch.nn as nn
from lr_scheduler import LR_Scheduler
def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--max_epochs', type=int, default=220, help='')#90 
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
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
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    parser.add_argument('--patch_size', type=int, default=100, help='patch_size')
    parser.add_argument('--test_model', type=str, default='64-99', help='patch_size(n)-epoch(e)')
    parser.add_argument('--embed_dir', type=str, default='/content/preprocessed')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--centroid', type=bool, default=False)

    args = parser.parse_args()
    return args



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




def train(model, train_dataLoader, optimizer,scheduler, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "feature" or k == "expression" or k == "position" or k == "id"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()
        count = batch["feature"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def load_data(args):
    
        # train_dataset = DATA_BRAIN(train=True,r=int(args.patch_size/2), fold=args.fold)
        # dummy_dataset= Dummy(train=True)
        # batch_sampler = CustomBatchSampler(dummy_dataset, shuffle=True)
        # # print('hellooooooooooooooooooooooooooo')
        # train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=batch_sampler,num_workers=3,pin_memory=True)
        NAMES=['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
        dir=args.embed_dir
        if args.demo == True:
            NAMES=NAMES[:2]
        train_dataset = CLUSTER_BRAIN(emb_folder=dir,train=True,split=True,name_list=NAMES,centroid=args.centroid)
        batch_sampler = CustomBatchSampler(train_dataset, shuffle=True)

        # dummy_dataset= Dummy(train=True)
        # batch_sampler = CustomBatchSampler(dummy_dataset, shuffle=True)
        # print('hellooooooooooooooooooooooooooo')
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=batch_sampler,num_workers=3,pin_memory=True)
        if args.local== True:
            print('local run')
            train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=batch_sampler,pin_memory=False)
        print(len(train_dataset))
        return train_dataLoader




def save_checkpoint(epoch, model, optimizer,scheduler, args, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args
    }
    dir=f"{args.path_save}"
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, f"{dir}/{filename}")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(epoch, model, optimizer,scheduler,args):
    filename=f"checkpoint_epoch_{epoch}.pth.tar"
    dir=f"{args.path_save}"
    checkpoint = torch.load(f"{dir}/{filename}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch + 1, args,model,scheduler,optimizer
# Apply the custom learning rate scheduler

def main():
    args = generate_args()
    print(args.resume)
    # device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mclSTExp_Attention_Pretrain(encoder_name=args.encoder_name,
                                spot_dim=args.dim,
                                temperature=args.temperature,
                                image_dim=args.image_embedding_dim,
                                projection_dim=args.projection_dim,
                                heads_num=args.heads_num,
                                heads_dim=args.heads_dim,
                                head_layers=args.heads_layers,
                                dropout=args.dropout)
    # model= nn.DataParallel(model,device_ids = [1, 3])
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    ratio = 512/args.batch_size
    train_dataLoader = load_data(args)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.3*ratio, weight_decay=1e-5
    )
    
    scheduler = LR_Scheduler(optimizer=optimizer
                             ,num_epochs=args.max_epochs
                             ,base_lr=0.3*ratio
                             ,iter_per_epoch = len(train_dataLoader)
                             ,warmup_epochs= 20
                            ,warmup_lr= 0.1*ratio
                            ,final_lr= 0.015
                            ,constant_predictor_lr=False
)
    start_epoch = 0
    # if args.resume ==True :
    # print('Resume')
    # start_epoch, args, model,scheduler, optimizer = load_checkpoint(114, model, optimizer,scheduler,args)
    print(f'start epoch: {start_epoch}, batch size: {args.batch_size}')
    
    # Training loop
    # scheduler = get_scheduler(scheduler_cfg=args.train.scheduler, optimizer=optimizer)
    # scheduler = lr_scheduler.LambdaLR(
    # optimizer, lr_lambda=lambda epoch: custom_lr_schedule(epoch, args.max_epochs))
    for step in range(start_epoch*len(train_dataLoader)):
        scheduler.step()
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        
        train(model, train_dataLoader, optimizer,scheduler , epoch)
        
        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
        if (epoch+1)%5 ==0:
            save_checkpoint(epoch, model, optimizer,scheduler, args, filename=checkpoint_filename)
    # for epoch in range(args.max_epochs):
    #     model.train()
    #     train(model, train_dataLoader, optimizer, epoch)

    
    # save_model(args, model)
    print("Saved Model")


if __name__ == '__main__':
    main()
