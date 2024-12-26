import os
import spatialdata as sd
import pickle
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
dir = f'D:/data/crunch_large/data/'
# dir=f'F:/DATA/crunch_large/zip_server'
NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
# names = sample_names[2:]
# with open(f'./pre_load/DC1_cells.pkl','wb') as f:
#     sdata = sd.read_zarr(f"{dir}/{NAMES[0]}.zarr")
#     cell_list=[]
#     for props in tqdm( regionprops(sdata['HE_nuc_registered'][0, :, :].to_numpy()) ):
#         cell_item={}
#         cell_item['cell_id']= props.label
#         centroid = props.centroid
#         cell_item['center']=[int(centroid[0]), int(centroid[1])]
#         cell_list.append(cell_item)
#     pickle.dump(cell_list,f)
for name in NAMES:
    pre_load_path= './pre_load'
    
    with open(f'./pre_load/{name}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
    sdata = sd.read_zarr(f"{dir}/{name}.zarr")
    cell_id_group=sdata['cell_id-group']
    
    # anucleus= sdata['anucleus']
    
    cell_id_train =  cell_id_group.obs[ cell_id_group.obs['group'] == 'train']['cell_id'].to_numpy()
    # cell_id_train = list(set(cell_id_train).intersection(set( anucleus.obs['cell_id'].unique())))

    cell_id_test= cell_id_group.obs[ cell_id_group.obs['group'] == 'test']['cell_id']
    cell_id_validation= cell_id_group.obs[ cell_id_group.obs['group'] == 'validation']['cell_id']
    max_len= max(max(cell_id_train),max(cell_id_test),max(cell_id_validation))
    bool_list= [0]*(max_len+1)
    
    for id in cell_id_train:
        bool_list[id]=1
    for id in cell_id_test:
        bool_list[id]=2
    for id in cell_id_validation:
        bool_list[id]=3
    ground_truth=np.array([])
    if name != 'DC1':
        anucleus= sdata['anucleus']
        cell_id_train = list(set(cell_id_train).intersection(set( anucleus.obs['cell_id'].unique())))
        ground_truth =  anucleus.layers['counts'][ anucleus.obs['cell_id'].isin(cell_id_train),:]
    del sdata
    r=10
    # im= sdata['HE_registered'].to_numpy()
    patches_list = []  # Initialize an empty list to store patches
    index=0
    for props in cell_list:
        cell_id= props['cell_id']
        centroid = props['center']
        if cell_id >= len(bool_list):
            props['label']='None'
            props['anucleus']=np.array([])
            continue
        elif bool_list[cell_id] ==1:
            if len(ground_truth)!=0:
                props['anucleus']=ground_truth[index]
                index+=1
            else:
                props['anucleus']=np.array([])
            props['label']='train'
        elif bool_list[cell_id] ==2:
            props['anucleus']=np.array([])
            props['label']='test'
        elif bool_list[cell_id] ==3:
            props['anucleus']=np.array([])
            props['label']='validation'
        else:
            props['anucleus']=np.array([])
            props['label']='None'
    # print(cell_list[0])       
    with open(f'./pre_load/{name}_cells.pkl','wb') as f:
            pickle.dump(cell_list,f)