import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention
from dataset import DATA_BRAIN
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
from train import generate_args
import pickle
import spatialdata as sd

model_path=f"./model_result/mcistexp/best_0.pt"
NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_NI']
SAVE_EMBEDDINGS = True

def build_loaders_inference():
    
    train_dataset= DATA_BRAIN(train=True)
    test_dataset = DATA_BRAIN(train=False)
        


    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    train_loader= DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(len(test_loader))
    print("Finished building loaders")
    return train_loader,test_loader


def get_embeddings(model_path,model):
    train_loader, test_loader = build_loaders_inference()
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to('cuda')
    print("Finished loading model")

    train_image_embeddings = []
    train_spot_embeddings = []
    test_image_embeddings = []
    test_spot_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            train_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y
            
            spot_features = spot_feature.unsqueeze(dim=0)
            train_spot_embedding = model.spot_encoder(spot_features)
            train_spot_embedding = model.spot_projection(train_spot_embedding).squeeze(dim=0)
            train_spot_embeddings.append(train_spot_embedding)
        for batch in tqdm(test_loader):
            
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y
            
            spot_features = spot_feature.unsqueeze(dim=0)
            test_spot_embedding = model.spot_encoder(spot_features)
            test_spot_embedding = model.spot_projection(test_spot_embedding).squeeze(dim=0)
            test_spot_embeddings.append(test_spot_embedding)
    return torch.cat(train_image_embeddings), torch.cat(train_spot_embeddings),torch.cat(test_image_embeddings), torch.cat(test_spot_embeddings)


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

def find_matches(spot_embeddings, query_embeddings, top_k=1, batch_size=1000):
    # Convert embeddings to tensors and normalize them
    spot_embeddings = torch.tensor(spot_embeddings).to(torch.float32)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    
    # Prepare a list to store the indices of top matches for each query
    all_indices = []
    
    # Process query embeddings in batches
    batch_size=256
    for i in range(0, len(query_embeddings), batch_size):
        # Ensure we do not go out of bounds
        batch_query_embeddings = torch.tensor(query_embeddings[i:min(i+batch_size, len(query_embeddings))]).to(torch.float32)
        batch_query_embeddings = F.normalize(batch_query_embeddings, p=2, dim=-1)

        # Compute dot product similarity
        dot_similarity = batch_query_embeddings @ spot_embeddings.T

        # Find the top-k matches for each query in the batch
        _, batch_indices = torch.topk(dot_similarity, k=top_k, dim=-1)
        
        # Append results to the final list
        all_indices.append(batch_indices.cpu().numpy())
    
    # Concatenate results for all batches into a single array
    return np.concatenate(all_indices, axis=0)
def save_embeddings(model_path, save_path, train_datasize, test_datasize):

    args = generate_args()
    model = mclSTExp_Attention(encoder_name=args.encoder_name,
                               spot_dim=args.dim,
                               temperature=args.temperature,
                               image_dim=args.image_embedding_dim,
                               projection_dim=args.projection_dim,
                               heads_num=args.heads_num,
                               heads_dim=args.heads_dim,
                               head_layers=args.heads_layers,
                               dropout=args.dropout)

    train_img_embeddings_all, train_spot_embeddings_all, \
    test_img_embeddings_all, test_spot_embeddings_all = get_embeddings(model_path,model)
    train_img_embeddings_all = train_img_embeddings_all.cpu().numpy()
    train_spot_embeddings_all = train_spot_embeddings_all.cpu().numpy()
    test_img_embeddings_all = test_img_embeddings_all.cpu().numpy()
    test_spot_embeddings_all = test_spot_embeddings_all.cpu().numpy()
    print("train img_embeddings_all.shape", train_img_embeddings_all.shape)
    print("train spot_embeddings_all.shape", train_img_embeddings_all.shape)
    print("test img_embeddings_all.shape", test_img_embeddings_all.shape)
    print("test spot_embeddings_all.shape", test_spot_embeddings_all.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(train_datasize)):
        index_start = sum(train_datasize[:i])
        index_end = sum(train_datasize[:i + 1])
        train_image_embeddings = train_img_embeddings_all[index_start:index_end]
        train_spot_embeddings = train_spot_embeddings_all[index_start:index_end]
        print("train image_embeddings.shape", train_image_embeddings.shape)
        print("train spot_embeddings.shape", train_spot_embeddings.shape)
        np.save(save_path + "train_img_embeddings_" + str(NAMES[i]) + ".npy", train_image_embeddings.T)
        np.save(save_path + "train_spot_embeddings_" + str(NAMES[i]) + ".npy", train_spot_embeddings.T)
    
    for i in range(len(test_datasize)):
        index_start = sum(test_datasize[:i])
        index_end = sum(test_datasize[:i + 1])
        test_image_embeddings = test_img_embeddings_all[index_start:index_end]
        test_spot_embeddings = test_spot_embeddings_all[index_start:index_end]
        print("test image_embeddings.shape", test_image_embeddings.shape)
        print("test spot_embeddings.shape", test_spot_embeddings.shape)
        np.save(save_path + "test_img_embeddings_" + str(NAMES[i]) + ".npy", test_image_embeddings.T)
        np.save(save_path + "test_spot_embeddings_" + str(NAMES[i]) + ".npy", test_spot_embeddings.T)

def get_sdata(name):
        path= f'C:/data/crunch/data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata
def get_expression():
    train_log1p_dict=[]
    test_log1p_dict=[]
    sdata=None
    for name in NAMES[:1]:
        sdata= get_sdata(name)
        
        with open(f'{name}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                split_train_binary = pickle.load(f)
        # train_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==1].T
        # test_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==0].T
        train_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==1].T)
        test_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==0].T)
    gene_list= sdata['anucleus'].var['gene_symbols'].values
    return train_log1p_dict,test_log1p_dict,gene_list 
    

names = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_NI']
# print(names)
# print(len(names))
#
NAMES=NAMES[:1]
train_datasize=[]
test_datasize=[]
for i in NAMES[:1]:
    with open(f'{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
        split_train_binary = pickle.load(f)
    train_datasize.append(sum(split_train_binary))
    test_datasize.append(len(split_train_binary)-sum(split_train_binary))
    

# datasize = [np.load(f"./data/preprocessed_expression_matrices/mcistexp/{name}/preprocessed_matrix.npy").shape[1] for
#             name in names]

if SAVE_EMBEDDINGS:
        fold=0
        save_embeddings(model_path=f"./model_result/mcistexp/best_{fold}.pt",
                        save_path=f"./embedding_result/mcistexp_result/embeddings_{fold}/",
                        train_datasize=train_datasize,
                        test_datasize=test_datasize)


# spot_expressions = [np.load(f"./data/preprocessed_expression_matrices/mcistexp/{name}/preprocessed_matrix.npy")
#                     for name in names]
train_spot_expressions, test_spot_expressions, gene_list=get_expression()


    
hvg_pcc_list = []
heg_pcc_list = []
mse_list = []
mae_list = []
fold=0
for test_name in NAMES[:1]:
    save_path = f"./embedding_result/mcistexp_result/embeddings_{fold}/"
    train_spot_embeddings = [np.load(save_path + f"train_spot_embeddings_{i}.npy") for i in NAMES]
    test_spot_embeddings = [np.load(save_path + f"test_spot_embeddings_{test_name}.npy")]

    image_embeddings = np.load(save_path + f"test_img_embeddings_{test_name}.npy")


    image_query = image_embeddings
    expression_gt = test_spot_expressions[NAMES.index(test_name)]
    spot_embeddings = train_spot_embeddings
    spot_expressions_rest = train_spot_expressions

    spot_key = np.concatenate(spot_embeddings, axis=1)
    # spot_key=spot_embeddings
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    method = "weighted"
    # save_path = f"./mcistexp_pred_att/{names[fold]}/"
    os.makedirs(save_path, exist_ok=True)
    if image_query.shape[1] != 256:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)

    indices = find_matches(spot_key, image_query, top_k=200)
    matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
    with open(f'indices.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(indices, f)
    

    for i in range(indices.shape[0]):
        a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
        # from sklearn.metrics.pairwise import cosine_similarity
        #
        # a = 1 - cosine_similarity(spot_key[indices[i, :], :], image_query[i, :].reshape(1, -1))
        reciprocal_of_square_a = np.reciprocal(a ** 2)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        weights = weights.flatten()
        matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0,
                                                        weights=weights)

    # np.save(save_path + "matched_spot_expression_pred_mclSTExp.npy", matched_spot_expression_pred.T)
    true = expression_gt
    pred = matched_spot_expression_pred


    adata_true = anndata.AnnData(true)
    adata_pred = anndata.AnnData(pred)

    adata_pred.var_names = gene_list
    adata_true.var_names = gene_list

    with open(f'adata_true.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(adata_true, f)
    with open(f'adata_pred.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(adata_pred, f)
    with open(f'true.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(true, f)
    with open(f'pred.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(pred, f)            
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

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true, pred)
    mse_list.append(mse)
    print("Mean Squared Error (MSE): ", mse)
    mae = mean_absolute_error(true, pred)
    mae_list.append(mae)
    print("Mean Absolute Error (MAE): ", mae)

    print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
    print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
    print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
    print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
