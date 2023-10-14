import itertools
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# df = pd.read_csv('user_shared_news.csv')
# df = df.sample(5000)
# print(df['User1'].nunique())
df = pd.read_csv('small_graph.csv')

# # distribution of count
# # plt.hist(df['Count'], bins=10, color='blue', alpha=0.7)
# # plt.xlabel('Count')
# # plt.ylabel('Frequency')
# # plt.title('Distribution of Count Values')
# # plt.grid(True)
# # plt.show()

G = nx.Graph()

# Iterate through the DataFrame and add edges with weights to the graph
for _, row in df.iterrows():
    user1 = row['Source']
    user2 = row['Target']
    count = row['Weight']
    G.add_edge(user1, user2, weight=count)

# print('now visualization')
# G.remove_nodes_from(list(nx.isolates(G)))
# nx.draw(G, with_labels=True)
# plt.show()
print('now community detection')

# # compute the best partition
# communities_generator = nx.community.louvain_communities(G)
# partition = {node: i for i, comm in enumerate(communities_generator) for node in comm}
# communities_list = [comm for comm in communities_generator]

# print(len(communities_list))

# modularity_score = nx.community.modularity(G, communities_list)
# print("Modularity:", modularity_score)

# # draw the graph
# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.viridis 
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=10,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()

# print('done with louvain')

communities_generator = nx.community.girvan_newman(G)
print('done with gn')
# # TODO: how to decide num partitions? 18 categories
num_partitions = 2
communities_list = [c for c in next(communities_generator)]
# for communities in itertools.islice(communities_generator, None):
#     communities_list = [comm for comm in communities_generator]
#     print(len(communities_list))

# community_assignment = {}
# for idx, partition in enumerate(communities):
#     for node in partition:
#         community_assignment[node] = idx

# communities_list = [comm for comm in communities_generator]

modularity_score = nx.community.modularity(G, communities_list)
print("Modularity:", modularity_score)


# pos = nx.spring_layout(G)
# # node_colors = [community_assignment[node] for node in G.nodes()]
# cmap = cm.get_cmap('viridis', num_partitions)
# # nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, node_size=10)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()
