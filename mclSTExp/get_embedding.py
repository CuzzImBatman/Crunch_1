import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention_Pretrain
from dataset import CLUSTER_BRAIN
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
from train import generate_args
import pickle
import spatialdata as sd

NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']



def get_embeddings(model_path,args,model,r,save_path):
    # train_image_embeddings_dict=[]
    # train_spot_embeddings_dict=[]
    test_image_embeddings_dict = {}
    train_spot_embeddings_dict ={}
    for i in NAMES:
        train_spot_embeddings = []

        train_dataset= CLUSTER_BRAIN(train=True,split= True,name_list=[i], centroid=args.centroid)
        train_loader =DataLoader(train_dataset, batch_size=256, shuffle=False)
        checkpoint = torch.load(model_path)
        state_dict=checkpoint['model_state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')  # remove the prefix 'module.'
            new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
            new_state_dict[new_key] = state_dict[key]

        model.load_state_dict(new_state_dict)
        model.eval()
        model = model.to('cuda')
        print("Finished loading model")
        # train_image_embeddings = []
        # train_spot_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader):
                
                spot_feature = batch["expression"].cuda()
                x = batch["position"][:, 0].long().cuda()
                y = batch["position"][:, 1].long().cuda()
                centers_x = model.x_embed(x)
                centers_y = model.y_embed(y)
                spot_feature = spot_feature + centers_x + centers_y
                
                spot_features = spot_feature.unsqueeze(dim=0)
                spot_features = spot_features.to(torch.float32)
                train_spot_embedding = model.spot_encoder(spot_features)
                train_spot_embedding = model.spot_projection(train_spot_embedding).squeeze(dim=0)
                train_spot_embeddings.append(train_spot_embedding)
        train_spot_embeddings_dict[i]= torch.cat(train_spot_embeddings)
        # TiE= torch.Tensor(train_image_embeddings).cpu().numpy()
        # TsE= torch.Tensor(train_spot_embeddings).cpu().numpy()
        # np.save(save_path + "train_img_embeddings_" + str(NAMES[i]) + ".npy", TiE.T)
        # np.save(save_path + "train_spot_embeddings_" + str(NAMES[i]) + ".npy", TsE.T)

    for i in NAMES:
        test_image_embeddings = []

        test_dataset= CLUSTER_BRAIN(train=False,split= True,name_list=[i], centroid=args.centroid)
        test_loader =DataLoader(test_dataset, batch_size=256, shuffle=False)
        checkpoint = torch.load(model_path)
        state_dict=checkpoint['model_state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')  # remove the prefix 'module.'
            new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
            new_state_dict[new_key] = state_dict[key]

        model.load_state_dict(new_state_dict)
        model.eval()
        model = model.to('cuda')
        print("Finished loading model")
        # train_image_embeddings = []
        # train_spot_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                
               
                image_features = batch["feature"].cuda()
                image_embeddings = model.image_projection(image_features)
                test_image_embeddings.append(image_embeddings)

               
        test_image_embeddings_dict[i]= torch.cat(test_image_embeddings)        
        # train_image_embeddings_dict.append(train_image_embeddings)
        # train_spot_embeddings_dict.append(train_spot_embeddings)
        train_loader=None
        train_dataset=None
        del train_loader
        del train_dataset
        
        
    
               
        # TiE= torch.Tensor(test_image_embeddings).cpu().numpy()
        # np.save(save_path + "train_img_embeddings_" + str(NAMES[i]) + ".npy", TiE.T)
        # test_image_embeddings_dict.append(test_image_embeddings)
        test_loader=None
        test_dataset=None
        del test_loader
        del test_dataset

        
    
    
    return train_spot_embeddings_dict, test_image_embeddings_dict


# def find_matches(spot_embeddings, query_embeddings, top_k=1):
#     # find the closest matches
#     spot_embeddings = torch.tensor(spot_embeddings)
#     query_embeddings = torch.tensor(query_embeddings)
#     query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
#     spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
#     dot_similarity = query_embeddings @ spot_embeddings.T
#     print(dot_similarity.shape)
#     _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

#     return indices.cpu().numpy()


def save_embeddings(model_path, save_path, args,r):

    
    model = mclSTExp_Attention_Pretrain(encoder_name=args.encoder_name,
                               spot_dim=args.dim,
                               temperature=args.temperature,
                               image_dim=args.image_embedding_dim,
                               projection_dim=args.projection_dim,
                               heads_num=args.heads_num,
                               heads_dim=args.heads_dim,
                               head_layers=args.heads_layers,
                               dropout=args.dropout)

    train_spot_embeddings_all,test_image_embeddings_all = get_embeddings(model_path,args,model,r,save_path)
    for name in NAMES:
        train_spot_embeddings_all[name] = train_spot_embeddings_all[name].cpu().numpy()
        print("train spot_embeddings_all.shape", train_spot_embeddings_all[name].shape)
        test_image_embeddings_all[name] = test_image_embeddings_all[name].cpu().numpy()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + f"train_spot_embeddings_{name}.npy", train_spot_embeddings_all[name].T)
        np.save(save_path + f"test_image_embeddings_{name}.npy", test_image_embeddings_all[name].T)
    # for i in range(len(train_datasize)):
    #     index_start = sum(train_datasize[:i])
    #     index_end = sum(train_datasize[:i + 1])
    #     train_image_embeddings = train_img_embeddings_all[index_start:index_end]
    #     train_spot_embeddings = train_spot_embeddings_all[index_start:index_end]
    #     print("train image_embeddings.shape", train_image_embeddings.shape)
    #     print("train spot_embeddings.shape", train_spot_embeddings.shape)
    #     np.save(save_path + "train_img_embeddings_" + str(NAMES[i]) + ".npy", train_image_embeddings.T)
    #     np.save(save_path + "train_spot_embeddings_" + str(NAMES[i]) + ".npy", train_spot_embeddings.T)
    
    # for i in range(len(test_datasize)):
    #     index_start = sum(test_datasize[:i])
    #     index_end = sum(test_datasize[:i + 1])
    #     test_image_embeddings = test_img_embeddings_all[index_start:index_end]
    #     print("test image_embeddings.shape", test_image_embeddings.shape)
    #     np.save(save_path + "test_img_embeddings_" + str(NAMES[i]) + ".npy", test_image_embeddings.T)

def get_sdata(name):
        path= f'../data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

def main():

    args = generate_args()


    name_parse= args.test_model
    patch_size= int(name_parse.split('-')[0])
    epoch=name_parse.split('-')[1]
    MODEL_NAME= f'checkpoint_epoch_{epoch}.pth.tar'

    if args.centroid== True:
        model_path=f"./model_result_centroid/{patch_size}/{MODEL_NAME}"
        save_path=f"./model_result_centroid/{patch_size}/{epoch}/"
    else:
        model_path=f"./model_result/{patch_size}/{MODEL_NAME}"
        save_path=f"./model_result/{patch_size}/{epoch}/"
    save_embeddings(model_path=model_path,
                    save_path=save_path,
                    args=args
                    ,r=int(patch_size/2))

if __name__ == '__main__':
    main()

