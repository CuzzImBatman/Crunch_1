import random

# import pytorch_lightning as pl
import torch
# from pytorch_lightning.loggers import CSVLogger
# from torch.utils.data import DataLoader

# from dataset import ViT_HER2ST, ViT_SKIN
# from predict import model_predict
# from utils import *
# from vis_model import THItoGene
print(torch.cuda.is_available())
from graph_construction import calcADJ
import numpy as np

coords=[
  [0.17736609, 0.30462937],
  [0.93492857, 0.28291749],
  [0.02717966, 0.70387086],
  [0.00247485, 0.89755543],
  [0.22384338, 0.01602379],
  [0.44486938, 0.2245041 ],
  [0.3372515 , 0.08104929],
  [0.323427  , 0.66932625],
  [0.91211763, 0.73752829],
  [0.31217372, 0.88060721],
  [0.23881759, 0.62384695],
  [0.65989771, 0.91218239],
  [0.94611499, 0.60775275],
  [0.8702169 , 0.21653096],
  [0.88532215, 0.32250498]
]
# print(calcADJ(coord=coords))

# adj=calcADJ(coord=coords)
# adj=np.where(adj > 0,2,0.1)
import sys
nodes=60000
# adj=0
# adj=torch.zeros((nodes, nodes))
# print(adj.element_size() * adj.numel())
print( calcADJ(coords
))