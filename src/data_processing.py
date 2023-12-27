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

'''
make dataset from history
'''
# behaviors_df = behaviors_df[['user_id', 'history']]
# behaviors_df['history'] = behaviors_df['history'].str.split(' ')
# # Explode the 'impressions' column into separate rows
# behaviors_df = behaviors_df.explode('history', ignore_index=True)
# behaviors_df = behaviors_df.rename(columns={'history':'news_id'})
# print(behaviors_df.head())

'''
make dataset from impressions
'''
behaviors_df = behaviors_df[['user_id', 'impressions']]
behaviors_df['impressions'] = behaviors_df['impressions'].str.split(' ')
# Explode the 'impressions' column into separate rows
behaviors_df = behaviors_df.explode('impressions', ignore_index=True)
# print(behaviors_df.shape) -- (5843442, 2)
# filter out the items the user didn't interact with
behaviors_df = behaviors_df[~behaviors_df['impressions'].str.endswith('0')]
# keep only the news id (without the 'click' indicator)
behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda cell: cell[:-2])
behaviors_df = behaviors_df.rename(columns={'impressions':'news_id'})
# disregard if user clicks multiple times on the same item
behaviors_df.drop_duplicates(inplace=True)

'''
create pivot table with users as rows and new items as columns
'''
agg_df = behaviors_df.groupby(['user_id', 'news_id']).size().reset_index(name='count')
pv_behaviors = agg_df.pivot(index='user_id', columns='news_id', values='count')
pv_behaviors = pv_behaviors.notnull().astype('int')
# shape: (50000, 7713)
# print(pv_behaviors.columns)
# print(pv_behaviors.head())
# print(pv_behaviors.index)

sparse_matrix = csr_matrix(pv_behaviors)
save_npz('pv_behaviors.npz', sparse_matrix)

'''
create user ids to index mapping
'''
# pv_behaviors.index.to_series().to_csv('user_ids.csv', index=False)