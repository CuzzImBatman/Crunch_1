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



NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']



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

def get_sdata(name):
        path= f'../data/{name}.zarr'

        # print(path)
        sdata = sd.read_zarr(path)
        return sdata
def get_expression():
    train_log1p_dict=[]
    test_log1p_dict=[]
    for name in NAMES:
        sdata= get_sdata(name)
        
        with open(f'./train_split/{name}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
                split_train_binary = pickle.load(f)
        # train_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==1].T
        # test_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==0].T
        train_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==1].T)
        test_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==0].T)
        
    gene_list= sdata['anucleus'].var['gene_symbols'].values
    sdata=None
    del sdata
    return train_log1p_dict,test_log1p_dict,gene_list 
    
def main():

    args = generate_args()


    name_parse= args.test_model
    patch_size= int(name_parse.split('-')[0])
    epoch=name_parse.split('-')[1]
    MODEL_NAME= f'checkpoint_epoch_{epoch}.pth.tar'
    # NAMES=NAMES[:1]
   
    train_spot_expressions, test_spot_expressions, gene_list=get_expression()

        
    hvg_pcc_list = []
    heg_pcc_list = []
    mse_list = []
    mae_list = []
    for test_name in NAMES:
        save_path = f"./model_result/{patch_size}/{epoch}/"
        train_spot_embeddings = np.load(save_path + f"train_spot_embeddings.npy")
        # test_spot_embeddings = [np.load(save_path + f"test_spot_embeddings_{test_name}.npy")]

        image_embeddings = np.load(save_path + f"test_img_embeddings_{test_name}.npy")


        image_query = image_embeddings
        expression_gt = test_spot_expressions[NAMES.index(test_name)]
        spot_embeddings = train_spot_embeddings
        spot_expressions_rest = train_spot_expressions

        # spot_key = np.concatenate(spot_embeddings, axis=1)
        spot_key=spot_embeddings
        expression_key = np.concatenate(spot_expressions_rest, axis=1)
        # expression_key=spot_expressions_rest
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

        indices = find_matches(spot_key, image_query, top_k=3000)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        with open(save_path+ f'indices_{test_name}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
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

  
if __name__ == '__main__':
    main() 
   
   
       




# datasize = [np.load(f"./data/preprocessed_expression_matrices/mcistexp/{name}/preprocessed_matrix.npy").shape[1] for
#             name in names]






