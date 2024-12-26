import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention
from dataset import CLUSTER_BRAIN
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
from train import generate_args
import pickle
import spatialdata as sd
import csv



NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']


# C:\DATA\Crunch\mclSTExp\model_result\80\219\test_image_embeddings_DC5.npy test_img_embeddings_DC5.npy
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

def find_matches(spot_embeddings, query_embeddings, top_k=1, batch_size=1000,device='cuda:0'):
    # Convert embeddings to tensors and normalize them
    spot_embeddings = torch.tensor(spot_embeddings, dtype=torch.float32, device=device)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    
    # Prepare a list to store the indices of top matches for each query
    all_indices = []
    
    # Process query embeddings in batches
    batch_size=256
    for i in range(0, len(query_embeddings), batch_size):
        # Ensure we do not go out of bounds
        batch_query_embeddings = torch.tensor(
            query_embeddings[i:min(i+batch_size, len(query_embeddings))],
            dtype=torch.float32,
            device=device
        )
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
        path= f'D:/data/crunch_large/data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata
# def get_expression():
#     train_log1p_dict=[]
#     test_log1p_dict=[]
#     for name in NAMES:
#         sdata= get_sdata(name)
        
#         with open(f'./train_split/{name}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
#                 split_train_binary = pickle.load(f)
#         # train_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==1].T
#         # test_log1p_dict[name]=sdata['anucleus'].X[np.array(split_train_binary)==0].T
#         train_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==1].T)
#         test_log1p_dict.append(sdata['anucleus'].X[np.array(split_train_binary)==0].T)
        
#     gene_list= sdata['anucleus'].var['gene_symbols'].values
#     sdata=None
#     del sdata
#     return train_log1p_dict,test_log1p_dict,gene_list 
def extract_expressions(dataset):
    """
    Extracts the "expression" field from all items in a dataset and returns as a NumPy array.

    Parameters:
    - dataset: The dataset object to extract expressions from.

    Returns:
    - np.ndarray: A NumPy array containing all the "expression" data from the dataset.
    """
    expressions = [item["expression"] for item in dataset]
    ids=[item["cell_id"] for item in dataset]
    cluster_ids=[item["cluster_id"] for item in dataset]
    return np.array(expressions),ids,cluster_ids
def main():

    args = generate_args()


    name_parse= args.test_model
    patch_size= int(name_parse.split('-')[0])
    epoch=name_parse.split('-')[1]
    MODEL_NAME= f'checkpoint_epoch_{epoch}.pth.tar'
    # NAMES=NAMES[:1]
    
    # train_spot_expressions, test_spot_expressions, gene_list=get_expression()
    train_spot_expressions=[]
    test_spot_expressions=[]
    test_ids=[]
    cluster_ids_list=[]
    for name in NAMES:
        train_set= CLUSTER_BRAIN(train=True, split = True,name_list=[name],centroid=args.centroid)
        exps,_,_=extract_expressions(train_set)
        train_spot_expressions.append(exps)
        print(train_spot_expressions[-1].shape)
        test_set= CLUSTER_BRAIN(train= False,split= True, name_list=[name],centroid=args.centroid)
        exps,ids,cluster_ids=extract_expressions(test_set)
        test_spot_expressions.append(exps)
        test_ids.append(ids)
        cluster_ids_list.append(cluster_ids)
        exps=None
        ids=None
        train_set=None
        test_set=None

    # train_datasize=[]
    # test_datasize=[]
    # for i in NAMES:
    #     with open(f'./train_split/{i}_train.pkl','rb') as f:  # Python 3: open(..., 'rb')
    #         split_train_binary = pickle.load(f)
        
    #     train_datasize.append(sum(split_train_binary))
    #     test_datasize.append(len(split_train_binary)-sum(split_train_binary))
            
    
    hvg_pcc_list = []
    heg_pcc_list = []
    mse_list = []
    mae_list = []
    k=1850
    if args.centroid== True:
        save_path = f"./model_result_centroid/{patch_size}/{epoch}/"
    else:
        save_path = f"./model_result/{patch_size}/{epoch}/"
    csv_file_path = os.path.join(save_path, "results.csv")
    train_spot_embeddings = [np.load(save_path + f"train_spot_embeddings_{name}.npy") for name in NAMES]
    if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        mode = "w"  # Overwrite if the file does not exist or is empty
    else:
        mode = "a"
    with open(csv_file_path, mode=mode, newline="") as file:
        for test_name in NAMES:
            

                # Open the CSV file for writing
            
            writer = csv.writer(file)
        
        # Write headers
            writer.writerow(["Test Name", "Drop out Slide", "MSE", "MAE", "Avg HEG PCC", "Avg HVG PCC"])
            index= NAMES.index(test_name)

            image_embeddings = np.load(save_path + f"test_image_embeddings_{test_name}.npy")

            image_query = image_embeddings
            expression_gt = test_spot_expressions[NAMES.index(test_name)]
            # spot_embeddings = train_spot_embeddings
            #########
            
            spot_embeddings=train_spot_embeddings
            spot_expressions_rest = train_spot_expressions
            # spot_embeddings= train_spot_embeddings[:index] + train_spot_embeddings[index+1:]
            # spot_expressions_rest= train_spot_expressions[:index] + train_spot_expressions[index+1:]
            ##########################
            spot_embeddings=np.concatenate(spot_embeddings, axis=1)
            # spot_key = np.concatenate(spot_embeddings, axis=1)
            spot_key=spot_embeddings
            expression_key = np.concatenate(spot_expressions_rest, axis=0)
            
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
            if os.path.exists(save_path+ f'indices_{test_name}.pkl'):
                print('loading indices')
                with open(save_path+ f'indices_{test_name}.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                    indices = pickle.load(f)
            else:
                indices = find_matches(spot_key, image_query, top_k=k)
                with open(save_path+ f'indices_{test_name}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump(indices, f)
            matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
            matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
            
            

            for i in range(indices.shape[0]):
                a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
                cluster_id = cluster_ids_list[index][i]
                indices_list= indices[i, :]
                if cluster_id > 0:
                    cluster_index= cluster_ids_list[index].index(-(cluster_id+1))
                    a_cluster= np.linalg.norm(spot_key[indices[cluster_index, :], :] - image_query[cluster_index, :], axis=1, ord=1)
                    k_smallest_indices_add = np.argsort(a_cluster)[:int(k/3)]
                    a_cluster[k_smallest_indices_add]= a_cluster[k_smallest_indices_add]/1.5
                    # k_smallest_indices= np.argsort(a)[:int(k/2)]
                    a= np.concatenate((a,a_cluster))
                    indices_list= np.concatenate((indices_list,indices[cluster_index, :]))
                    ''''''
                #     reciprocal_of_square_a = np.reciprocal(a ** 2)
                #     reciprocal_of_square_a_cluster = np.reciprocal(a_cluster ** 2)
                #     weights_cluster = reciprocal_of_square_a_cluster / np.sum(reciprocal_of_square_a_cluster)
                #     weights_cluster = weights_cluster.flatten()
                #     ratio=0.75
                # # reciprocal_of_square_a = np.reciprocal(a)
                #     weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
                #     weights = weights.flatten()
                #     matched_spot_expression_pred[i, :] = np.average(expression_key[indices_list, :], axis=0,
                #                                                     weights=weights)*ratio + np.average(expression_key[indices[cluster_index, :], :], axis=0,
                #                                                     weights=weights_cluster)*(1-ratio)
                #     continue
                    
                # from sklearn.metrics.pairwise import cosine_similarity
                # a = 1 - cosine_similarity(spot_key[indices[i, :], :], image_query[i, :].reshape(1, -1))
                reciprocal_of_square_a = np.reciprocal(a ** 2)
                # reciprocal_of_square_a = np.reciprocal(a)
                weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
                weights = weights.flatten()
                matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices_list, :], axis=0, weights=weights)
                matched_spot_expression_pred[i, :] = np.average(expression_key[indices_list, :], axis=0,
                                                                weights=weights)

            # np.save(save_path + "matched_spot_expression_pred_mclSTExp.npy", matched_spot_expression_pred.T)
            # for i in range(indices.shape[0]):
            #     a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
            #     cluster_id = cluster_ids_list[index][i]
            #     if cluster_id > 0:55
            #         cluster_index= cluster_ids_list[index].index(-(cluster_id+1))
            positive_mask =np.array( test_ids[index]) > 0
            negative_mask =np.array( test_ids[index])  < 0
            for mask in [negative_mask, positive_mask]:
               
          
                true = expression_gt[mask]
                pred = matched_spot_expression_pred[mask]
                min_bound=0.24
                max_bound=4.6052
                pred[pred >max_bound] = max_bound
                pred[pred<min_bound]=0
                adata_true = anndata.AnnData(true)
                adata_pred = anndata.AnnData(pred)
                gene_list=[str(i) for i in range(460)]
                adata_pred.var_names = gene_list
                adata_true.var_names = gene_list

                # with open(f'adata_true.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #             pickle.dump(adata_true, f)
                # with open(f'adata_pred.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #             pickle.dump(adata_pred, f)
                # with open(f'true.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #             pickle.dump(true, f)
                # with open(f'pred.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #             pickle.dump(pred, f)            
                gene_mean_expression = np.mean(adata_true.X, axis=0)
                top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
                top_50_genes_names = adata_true.var_names[top_50_genes_indices]
                top_50_genes_expression = adata_true[:, top_50_genes_names]
                top_50_genes_pred = adata_pred[:, top_50_genes_names]

                heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)
                hvg_pcc, hvg_p = get_R(adata_pred, adata_true)
                hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

                

                from sklearn.metrics import mean_squared_error, mean_absolute_error

                mse = mean_squared_error(true, pred)
                print(f"Mean Squared Error (MSE) {test_name}: ", mse)
                mae = mean_absolute_error(true, pred)
                print(f"Mean Absolute Error (MAE) {test_name}:", mae)
                if sum(mask)!=sum(np.array( test_ids[index])  < 0):
                    mse_list.append(mse)
                    mae_list.append(mae)
                    heg_pcc_list.append(np.mean(heg_pcc))
                    hvg_pcc_list.append(np.mean(hvg_pcc))
                
                    print(f"avg heg pcc {test_name}: {np.mean(heg_pcc):.4f}")
                    print(f"avg hvg pcc {test_name}: {np.mean(hvg_pcc):.4f}")   
                    writer.writerow([test_name, NAMES[index], mse, mae, np.mean(heg_pcc), np.mean(hvg_pcc)])   

        print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
        print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
        print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
        print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
        writer.writerow(["Overall Average", "", np.mean(mse_list), np.mean(mae_list),
                    np.mean(heg_pcc_list), np.mean(hvg_pcc_list)])

  
if __name__ == '__main__':
    main() 
   
   
       




# datasize = [np.load(f"./data/preprocessed_expression_matrices/mcistexp/{name}/preprocessed_matrix.npy").shape[1] for
#             name in names]






