import pandas as pd
import networkx as nx
import dask.dataframe as dd
import dask.distributed
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.sparse import save_npz

## COMBINE DATASETS TO CREATE SPARSE MATRIX (step 2)

cate_df = pd.read_csv('data/MINDsmall_train/news.tsv', delimiter='\t')
cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
cate_df = cate_df[['news_id', 'cat', 'subcat']]

behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv', delimiter='\t')
behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

# make graph from history
# behaviors_df = behaviors_df[['user_id', 'history']]
# behaviors_df['history'] = behaviors_df['history'].str.split(' ')
# # Explode the 'impressions' column into separate rows
# behaviors_df = behaviors_df.explode('history', ignore_index=True)
# behaviors_df = behaviors_df.rename(columns={'history':'news_id'})
# print(behaviors_df.head())

# make graph from impressions
behaviors_df = behaviors_df[['user_id', 'impressions']]
behaviors_df['impressions'] = behaviors_df['impressions'].str.split(' ')
# Explode the 'impressions' column into separate rows
behaviors_df = behaviors_df.explode('impressions', ignore_index=True)
# behaviors_df = behaviors_df[~behaviors_df['impressions'].str.endswith('0')]
behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda cell: cell[:-2])
behaviors_df = behaviors_df.rename(columns={'impressions':'news_id'})

# print(len(behaviors_df))
# print(behaviors_df.head())

agg_df = behaviors_df.groupby(['user_id', 'news_id']).size().reset_index(name='count')
pv_behaviors = agg_df.pivot(index='user_id', columns='news_id', values='count')
pv_behaviors = pv_behaviors.notnull().astype('int')

print(pv_behaviors.head())
print(len(pv_behaviors))

# # Convert the dataframe to a sparse user-news matrix
user_ids = behaviors_df['user_id'].astype('category').cat.codes
user_id_to_code = dict(zip(behaviors_df['user_id'], user_ids))
user_id_code_df = pd.DataFrame(list(user_id_to_code.items()), columns=['user_id', 'code'])

# Save the DataFrame as a CSV file
# user_id_code_df.to_csv('user_id_to_code_mapping.csv', index=False)
# print((len(user_ids), len(news_ids)))

############ PCA ####################

# from sklearn.decomposition import PCA

# # Fit PCA on your data
# pca = PCA().fit(pv_behaviors)

# # Calculate the cumulative sum of the explained variance ratio
# cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# # Find the number of components for a certain variance
# n_components_95 = np.where(cumulative_variance_ratio > 0.95)[0][0] + 1
# n_components_99 = np.where(cumulative_variance_ratio > 0.99)[0][0] + 1

# print(f"Number of components for 95% variance: {n_components_95}")
# print(f"Number of components for 99% variance: {n_components_99}")
# # print(user_id_code_df_pca.shape)

######### DISTANCE FUNCTIONS ###########

def pairwise_distance(A, B, p=2):
    """
    A: ndarray - m x n matrix, where each row is a vector of length n 
    B: ndarray - k x n matrix, where each row is a vector of length n 
    p: int - the order of the norm 

    Precondition: if A is m x n, then b must be k x n,
    i.e. the two inputs should agree in dimmension in the second component
    """
    a = A[:, None, :]
    b = B[None, :, :]
    
    return np.linalg.norm(a-b, axis=-1, ord=p)

# # # Compute the dot product for jaccard index

sparse_matrix = csr_matrix(pv_behaviors)
print(sparse_matrix.shape)
user_user_matrix = sparse_matrix.dot(sparse_matrix.T)
print(user_user_matrix.shape)

save_npz('all_recs.npz', user_user_matrix)