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
import itertools
import pandas as pd
NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']

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
        
        path= f'F:/Data/crunch_large/submit/data/{name}.zarr'
        # print(path)
        sdata = sd.read_zarr(path)
        return sdata

def extract_expressions(dataset):
    """
    Extracts the "expression" field from all items in a dataset and returns as a NumPy array.

    Parameters:
    - dataset: The dataset object to extract expressions from.

    Returns:
    - np.ndarray: A NumPy array containing all the "expression" data from the dataset.
    """
    expressions = [item["expression"] for item in dataset]
    id=[item["cell_id"] for item in dataset]
    return np.array(expressions), id
def main():

    args = generate_args()


    name_parse= args.test_model
    patch_size= int(name_parse.split('-')[0])
    epoch=name_parse.split('-')[1]
    MODEL_NAME= f'checkpoint_epoch_{epoch}.pth.tar'
    sdata= get_sdata('DC5')
    gene_names = sdata["anucleus"].var.index

    train_spot_expressions=[]
    test_cell_ids=[]

    for name in NAMES:
        train_set= CLUSTER_BRAIN(train=True, split = True,name_list=[name])
        exps,_=extract_expressions(train_set)
        train_spot_expressions.append(exps)
        print(train_spot_expressions[-1].shape)
        test_set= CLUSTER_BRAIN(train= False,split= True, name_list=[name])
        _,cell_ids=extract_expressions(test_set)
        test_cell_ids.append(cell_ids)
        train_set=None
        test_set=None
        exps=None

 
    
   
    save_path = f"./model_result/{patch_size}/{epoch}/"
    csv_file_path = os.path.join(save_path, "output.csv")
    all_predictions=[]

    train_spot_embeddings = [np.load(save_path + f"train_spot_embeddings_{name}.npy") for name in NAMES]
    with open(csv_file_path, mode="w", newline="") as file:
        for test_name in NAMES:
            

                # Open the CSV file for writing
            
            writer = csv.writer(file)
        
        # Write headers
            writer.writerow(["Test Name", "Drop out Slide", "MSE", "MAE", "Avg HEG PCC", "Avg HVG PCC"])
            index= NAMES.index(test_name)

            image_embeddings = np.load(save_path + f"test_image_embeddings_{test_name}.npy")

            image_query = image_embeddings
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
            
            if spot_key.shape[1] != 256:
                spot_key = spot_key.T
                print("spot_key shape: ", spot_key.shape)
            if expression_key.shape[0] != spot_key.shape[0]:
                expression_key = expression_key.T
                print("expression_key shape: ", expression_key.shape)

            indices = find_matches(spot_key, image_query, top_k=1850)
            matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
            matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
            # with open(save_path+ f'indices_{test_name}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            #             pickle.dump(indices, f)
            

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
            pred = matched_spot_expression_pred
            cell_ids = test_cell_ids[index]
            
            pred = np.round(pred, 2)
            prediction = pd.DataFrame(
                itertools.product(cell_ids, gene_names),
                columns=["cell_id", "gene"]
            )
            prediction["prediction"] = pred.ravel(order="C")
            prediction["slide_name"] = NAMES[index]  # Add the slide name as a new column
            
            all_predictions.append(prediction)

        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions.to_csv(f"{csv_file_path}", index=False)
        print(f"Predictions saved to {csv_file_path}")    

  
if __name__ == '__main__':
    main() 
   
   
       




# datasize = [np.load(f"./data/preprocessed_expression_matrices/mcistexp/{name}/preprocessed_matrix.npy").shape[1] for
#             name in names]






