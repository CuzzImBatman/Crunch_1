from dataset import NeuronData,build_batch_graph
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from model import GATModel
traindata= NeuronData(train=True, split =True,name_list= ['DC5'])
train_dataLoader =DataLoader(traindata, batch_size=1000, shuffle=False,pin_memory=True)    
# print(len(train_dataLoader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# traindata[2127]
model=GATModel()
model= model.to('cuda')
model.eval()
for batch in train_dataLoader:
    # print(type(batch)) 
    batch = batch.to('cuda')
    # Check if each item in the batch is a Data object
    # print(type(batch[0]))
    batch_graph = build_batch_graph(batch,device)
    batch.cpu()
    # Pass to model
    output,exps = model(batch_graph)
    batch_graph.cpu()
