import pandas as pd
import networkx as nx
import dask.dataframe as dd
import dask.distributed
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.sparse import save_npz

cate_df = pd.read_csv('MINDsmall_train/news.tsv', delimiter='\t')
cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
cate_df = cate_df[['news_id', 'cat', 'subcat']]

behaviors_df = pd.read_csv('MINDsmall_train/behaviors.tsv', delimiter='\t')
behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
behaviors_df = behaviors_df[['user_id', 'impressions']]

behaviors_df['impressions'] = behaviors_df['impressions'].str.split(' ')
# Explode the 'impressions' column into separate rows
behaviors_df = behaviors_df.explode('impressions', ignore_index=True)
behaviors_df = behaviors_df[~behaviors_df['impressions'].str.endswith('0')]
behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda cell: cell[:-2])
behaviors_df = behaviors_df.rename(columns={'impressions':'news_id'})
print(len(behaviors_df))
print(behaviors_df.head())

user_news_matrix = behaviors_df.pivot_table(index='user_id', columns='news_id', aggfunc='size', fill_value=0)
# print('pivoted')
# print(user_news_matrix)


# Convert the dataframe to a sparse user-news matrix
user_ids = behaviors_df['user_id'].astype('category').cat.codes
news_ids = behaviors_df['news_id'].astype('category').cat.codes
sparse_matrix = csr_matrix((np.ones(len(behaviors_df)), (user_ids, news_ids)))

# Compute the dot product
user_user_matrix = sparse_matrix.dot(sparse_matrix.T)



# save_npz('sparse_matrix.npz', user_user_matrix)

# print(user_user_matrix)

# dense_matrix = user_user_matrix.toarray()
# np.save('dense_matrix.npy', dense_matrix)

# G = nx.Graph()

# # Get user labels
# user_labels = behaviors_df['user_id'].astype('category').cat.categories

# # Iterate over non-zero entries and add edges
# rows, cols = user_user_matrix.nonzero()
# weights = user_user_matrix[rows, cols].A1 
# for i, j, w in zip(rows, cols, weights):
#     if i != j:  # Exclude self-loops
#         G.add_edge(user_labels[i], user_labels[j], weight=w)

# print('now visualization')
# G.remove_nodes_from(list(nx.isolates(G)))
# nx.draw(G, with_labels=True)
# plt.show()

# print(dense_matrix)
# print(len(dense_matrix))
# print(len(dense_matrix[0]))



# behaviors_df = behaviors_df.sample(80000)

# shared_news_df = behaviors_df.merge(behaviors_df, on='news_id')

# shared_news_df = shared_news_df[shared_news_df['user_id_x'] != shared_news_df['user_id_y']]
# shared_news_df = shared_news_df.groupby(['user_id_x', 'user_id_y']).size().reset_index(name='count')

# # Sort the 'user_id_x' and 'user_id_y' columns separately
# sorted_user_id_x = shared_news_df['user_id_x'].where(shared_news_df['user_id_x'] <= shared_news_df['user_id_y'], shared_news_df['user_id_y'])
# sorted_user_id_y = shared_news_df['user_id_y'].where(shared_news_df['user_id_x'] <= shared_news_df['user_id_y'], shared_news_df['user_id_x'])

# # Update the DataFrame with the sorted columns
# shared_news_df['user_id_x'] = sorted_user_id_x
# shared_news_df['user_id_y'] = sorted_user_id_y

# print(len(shared_news_df)) # 67129765 comparisons

# # # Remove duplicate rows, keeping only one occurrence of each pair
# shared_news_df = shared_news_df.drop_duplicates(subset=['user_id_x', 'user_id_y'])

# print(len(shared_news_df))

# # print(shared_news_df[shared_news_df['count'] > 1].head())
# # print(cate_df.head())

# # df = shared_news_df.merge(cate_df, on='news_id')
# # print(df.head())

# # user_news_count = df['user_id'].value_counts()
# # print(user_news_count.index)

# # # Create a graph
# G = nx.Graph()

# # Add nodes for each user
# G.add_nodes_from(shared_news_df['user_id_x'])

# for _, row in shared_news_df.iterrows():
#     G.add_edge(row['user_id_x'], row['user_id_y'], weight=row['count'])


# # shared_news_df = dd.from_pandas(shared_news_df, npartitions=10)

# # cluster = dask.distributed.LocalCluster(dashboard_address=':8888')
# # client = dask.distributed.Client(cluster)

# # # Convert the Dask DataFrame to a Pandas DataFrame
# # pandas_df = shared_news_df.compute()

# # # Create a NetworkX graph in parallel
# # def create_graph(data):
# #     G = nx.Graph()
# #     for _, row in data.iterrows():
# #         G.add_edge(row['user_id_x'], row['user_id_y'], weight=row['count'])
# #     return G

# # # Split the Pandas DataFrame into smaller chunks and process them in parallel
# # chunked_data = np.array_split(pandas_df, len(cluster.workers))
# # graphs = client.map(create_graph, chunked_data)
# # graphs = client.gather(graphs)

# # # Merge the graphs into one final graph
# # final_graph = nx.compose_all(graphs)

# # # Shutdown the Dask cluster
# # client.close()
# # cluster.close()

# print("Graph Nodes:", G.nodes())
# print("Graph Edges:", G.edges(data=True))

# edge_data = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

# # Create a DataFrame from the edge data
# edge_df = pd.DataFrame(edge_data, columns=['User1', 'User2', 'Count'])

# # Save the DataFrame as a CSV file
# edge_df.to_csv('user_shared_news.csv', index=False)