import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

NAMES = ['DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
for name in NAMES:
    with open(f'./pre_load/{name}_cells.pkl','rb') as f:
        cell_list= pickle.load(f)


    cell_locations = pd.DataFrame([{'x': c['center'][0], 'y': c['center'][1]} for c in cell_list])

    # Define the number of clusters
    n_clusters = 1100  # Adjust this based on your data and requirements

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(cell_locations[['x', 'y']])

    # Add the cluster labels to the DataFrame
    cell_locations['cluster'] = kmeans.labels_

    # Visualize the clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(
        cell_locations['x'], 
        cell_locations['y'], 
        c=cell_locations['cluster'], 
        s=0.5, 
        cmap='viridis',  # Continuous colormap
        alpha=0.5
    )
    plt.colorbar(label='Cluster ID')  # Add colorbar to interpret color mapping
    plt.title('Clusters Visualized with Continuous Colormap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    os.makedirs('./cluster_data',exist_ok= True )
    with open(f'./cluster_data/{name}_cell.pkl','wb') as f:
        pickle.dump(cell_locations,f)
    with open(f'./cluster_data/{name}_kmeans.pkl','wb') as f:
        pickle.dump(kmeans,f)
    
