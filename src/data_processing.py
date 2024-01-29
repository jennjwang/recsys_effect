import pandas as pd
import networkx as nx
import dask.dataframe as dd
import dask.distributed
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.sparse import save_npz

## PIVOT DATASET TO CREATE SPARSE MATRIX (step 2)

# cate_df = pd.read_csv('data/MINDsmall_train/news.tsv', delimiter='\t')
# cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
# cate_df = cate_df[['news_id', 'cat', 'subcat']]

# behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv', delimiter='\t')
# behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

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
behaviors_df = pd.DataFrame({
    'user_id': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'impressions': [
        'item1 item2_0 item3',
        'item2_0 item3_0 item4_0',
        'item3 item4 item5',
        'item4_0 item5 item1',
        'item5 item1 item2'
    ]
})
behaviors_df = behaviors_df[['user_id', 'impressions']]
behaviors_df['impressions'] = behaviors_df['impressions'].str.split(' ')
# get the number of items recommended to users
behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda x: [item for item in x if not item.endswith('0')])
print(behaviors_df.head())
# behaviors_df['empty'] = behaviors_df['impressions'].apply(lambda x: x == [])
print('number of users', behaviors_df['user_id'].nunique())
print(behaviors_df.head())
# print('users without interaction: ', behaviors_df[behaviors_df['empty']]['user_id'].nunique())

# behaviors_df['impressions_len'] = behaviors_df['impressions'].apply(len)
# print(behaviors_df['impressions_len'].median()) # 24, the # of items users actually interacted with: 1
# print(behaviors_df['impressions_len'].mean()) # 37.22791213271833, the # of items users actually interacted with: 1.5
# print(behaviors_df['impressions_len'].min()) # 2, the # of items users actually interacted with: 1
# print(behaviors_df['impressions_len'].max()) # 299, the # of items users actually interacted with: 35

# Explode the 'impressions' column into separate rows
behaviors_df = behaviors_df.explode('impressions', ignore_index=True)
impression_counts = behaviors_df.groupby('impressions')['user_id'].nunique()
num_shared_impressions = (impression_counts > 1).sum()
print('num_shared', num_shared_impressions)

impression_users = behaviors_df.groupby('impressions')['user_id'].unique()

print(impression_users)

user_pairs = impression_users.apply(lambda x: list(combinations(x, 2)))

print(user_pairs)

# Flatten the list of user pairs and convert to a set to get the unique pairs
unique_user_pairs = set(pair for pairs_list in user_pairs for pair in pairs_list)
# Count the number of unique pairs
num_unique_pairs = len(unique_user_pairs)
print("num_unique", num_unique_pairs) # 69699419 / 1249975000 (from 50,000 C 2) = 0.0557

print(behaviors_df.shape) #(5843442, 2)
# filter out the items the user didn't interact with
# behaviors_df = behaviors_df[~behaviors_df['impressions'].str.endswith('0')]
# keep only the news id (without the 'click' indicator)
# behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda cell: cell[:-2])
behaviors_df = behaviors_df.rename(columns={'impressions':'news_id'})
# disregard if user clicks multiple times on the same item
print('dups', behaviors_df.duplicated().sum())
behaviors_df.drop_duplicates(inplace=True)

# print(len(behaviors_df))
print('news ids', behaviors_df['news_id'].nunique()) # 20288 total, 7713 that users clicked on

'''
create pivot table with users as rows and new items as columns
'''
agg_df = behaviors_df.groupby(['user_id', 'news_id']).size().reset_index(name='count')
pv_behaviors = agg_df.pivot(index='user_id', columns='news_id', values='count')
# print(pv_behaviors)
print(pv_behaviors.shape) # (50000, 7713)
print(pv_behaviors.max().max()) # 1

# pv_behaviors = pv_behaviors.notnull().astype('int')
# shape: (50000, 7713)
# print(pv_behaviors.columns)
# print(pv_behaviors.head())
# print(pv_behaviors.index)

# sparse_matrix = csr_matrix(pv_behaviors)
# save_npz('pv_behaviors.npz', sparse_matrix)

'''
create user ids to index mapping
'''
# pv_behaviors.index.to_series().to_csv('user_ids.csv', index=False)