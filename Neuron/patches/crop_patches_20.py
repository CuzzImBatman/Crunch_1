import os
import spatialdata as sd
import pickle
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
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
#         cell_item['center']=[int(centroid[1]), int(centroid[0])]
#         cell_list.append(cell_item)
#     pickle.dump(cell_list,f)
for name in NAMES[7:]:
    pre_load_path= '../../pre_load'
    
    with open(f'{pre_load_path}/{name}_cells.pkl','rb') as f:
                cell_list= pickle.load(f)
    sdata = sd.read_zarr(f"{dir}/{name}.zarr")
    r=40
    im= sdata['HE_registered'].to_numpy()
    patches_list = []  # Initialize an empty list to store patches
    for props in cell_list:
        cell_id= props['cell_id']
        centroid = props['center']
        
        x_center, y_center = centroid[1], centroid[0]
        x_center= int(x_center)
        y_center= int(y_center)     
                # Calculate the crop boundaries
        minr, maxr = y_center - r, y_center + r
        minc, maxc = x_center - r, x_center + r

        # Ensure boundaries are within the image dimensions
            
        if (minr <0) or (minc <0) or (maxr <0) or (maxc <0):
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
    patches_tensor= np.stack(patches_list)
    save_dir= f'D:/DATA/Gene_expression/Crunch/patches/{r*2}'
    os.makedirs(save_dir,exist_ok=True)
    np.save(f'{save_dir}/{name}.npy',patches_tensor)
    im=None
    patches_tensor=None
    sdata=None
    del im,patches_tensor
    print(f"Saved {len(patches_list)} patches to the list.")
