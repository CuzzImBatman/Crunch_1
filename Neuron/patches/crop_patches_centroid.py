import os
import spatialdata as sd
import pickle
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.measure import regionprops
dir = f'D:/data/crunch_large/data'
dir=f'F:/DATA/crunch_large/submit/data'
NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
# names = sample_names[2:]
# with open(f'./pre_load/DC1_cells.pkl','wb') as f:
#     sdata = sd.read_zarr(f"{dir}/{NAMES[0]}.zarr")
#     cell_list=[]
#     for props in tqdm( regionprops(sdata['HE_nuc_registered'][0, :, :].to_numpy()) ):
#         cell_item={}
#         cell_item['cell_id']= props.label
#         centroid = props.centroid
#         cell_item['center']=[int(centroid[1]), int(centroid[0])]
#         cell_list.append(cell_item)
#     pickle.dump(cell_list,f)
r=int(80/2)
group_type=['train','evel']
group_type=['train']

absolute_path='E:/DATA/crunch/tmp'
absolute_path_old='D:\\DATA\\Gene_Expression\\crunch\\Register'
for group in group_type:
    for name in NAMES:
        try:
            cluster_path= f'{absolute_path}/cluster/{group}/cluster_data_split'
            with open(f'{cluster_path}/{name}_cells.pkl','rb') as f:
                    cell_locations = pickle.load(f)
            # with open(f'{cluster_path}/{name}_kmeans.pkl','rb') as f:
            #         kmeans= pickle.load(f)
        except:
            cluster_path= f'{absolute_path}/cluster/{group}/cluster_data'
            with open(f'{cluster_path}/{name}_cells.pkl','rb') as f:
                    cell_locations = pickle.load(f)
            # with open(f'{cluster_path}/{name}_kmeans.pkl','rb') as f:
            #             kmeans= pickle.load(f)
        centroids = cell_locations.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
        
        # try:
        #     cluster_path= f'{absolute_path_old}/cluster/{group}/cluster_data_split'
        #     with open(f'{cluster_path}/{name}_cells.pkl','rb') as f:
        #             cell_locations_old = pickle.load(f)
           
        # except:
        #     cluster_path= f'{absolute_path_old}/cluster/{group}/cluster_data'
        #     with open(f'{cluster_path}/{name}_cells.pkl','rb') as f:
        #             cell_locations_old = pickle.load(f)
        # cell_locations['cluster'] = cell_locations_old['cluster']  
        # cluster_path= f'{absolute_path}/cluster/{group}/cluster_data_split'
        # with open(f'{cluster_path}/{name}_cells.pkl','wb') as f:
        #             pickle.dump(cell_locations,f) 
        centroids = cell_locations.groupby('cluster')[['x', 'y']].mean().sort_index().reset_index().to_numpy()
        valid = all(prop['train'].nunique() == 1 for _, prop in cell_locations.groupby('cluster'))

        if valid:
            print("The dataframe satisfies the condition.")
        else:
            print("The dataframe does not satisfy the condition.")
        # print(cell_locations['cluster'])
        # centroids_old = kmeans.cluster_centers_

        print(len(centroids))
        print(centroids[0,1],centroids[0,2])
    # Filter out invalid clusters (those with 'train' = -1)
        # if group == 'train':
        #     valid_clusters = cell_locations[cell_locations['train'] != -1]['cluster'].unique()
        # else:
        #     valid_clusters= cell_locations['cluster'].unique()  
        valid_clusters= cell_locations['cluster'].unique() 
        valid_clusters = np.sort(valid_clusters)

        sdata = sd.read_zarr(f"{dir}/{name}.zarr")
    
        im= sdata['HE_original'].to_numpy()
        patches_list = []  # Initialize an empty list to store patches
        
        for cluster_id in valid_clusters:
            x_center, y_center = centroids[cluster_id,1],centroids[cluster_id,2]
            x_center= int(x_center)
            y_center= int(y_center)     
                    # Calculate the crop boundaries
            minr, maxr = y_center - r, y_center + r
            minc, maxc = x_center - r, x_center + r

            # Ensure boundaries are within the image dimensions
                
            if (minr <0) or (minc <0) or (maxr >im.shape[1]) or (maxc >im.shape[2]):
                pad_top = max(0, -minr)
                minr = max(0, minr)

                pad_bottom = max(0, maxr - im.shape[1])
                maxr = min(maxr, im.shape[1])

                pad_left = max(0, -minc)
                minc = max(0, minc)

                pad_right = max(0, maxc - im.shape[2])
                maxc = min(maxc, im.shape[2])

            # Crop and pad the image if needed
            
                patch = np.pad(im[:, minr:maxr, minc:maxc],
                            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)
            else:
                print(minr,maxr,minc,maxc)
                patch = im[:, minr:maxr, minc:maxc]
            # patch = im[:, minr:maxr, minc:maxc]
            
            patch = Image.fromarray(np.transpose(patch,(2,1,0)))
            if patch.size !=(r*2,r*2):
                patch=patch.resize((r*2,r*2))
            # except:
            #     print( minr,maxr,minc,maxc, im.shape)
            
            # patch = transforms.ToTensor()(patch)
            patches_list.append(patch)
        
        patches_list= np.stack(patches_list, axis=0)
        save_dir= f'{absolute_path}/patches/{r*2}'
        os.makedirs(f'{save_dir}/{group}',exist_ok=True)
        print(patches_list.shape)
        np.save(f'{save_dir}/{group}/{name}.npy',patches_list)
        
        print(f"Saved {patches_list.shape} patches to the list.")
        im=None
        patches_list=None
        sdata=None
        del im,patches_list
