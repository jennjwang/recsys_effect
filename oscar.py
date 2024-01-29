
import networkx as nx
import functools
import numpy as np
import pandas as pd
from scipy.sparse import csr_array, triu, save_npz, load_npz
# import graph_tool.all as gt
# from graph_tool.all import *
import datetime
import Graph_Sampling

upper = load_npz('user_user_matrix.npz')

def calc_modularity(threshold, user_user_matrix):

    mask = user_user_matrix.data >= threshold
    filtered_rows, filtered_cols = user_user_matrix.nonzero()
    filtered_rows_m = filtered_rows[mask]
    filtered_cols_m = filtered_cols[mask]
    filtered_data = user_user_matrix.data[mask]

    percentage = filtered_data.size / \
        ((user_user_matrix.shape[0] * user_user_matrix.shape[1]) / 2)
    # print(percentage)
    print(filtered_data.size)

    print(f'making graph with threshold {threshold}')
    print("time:-", datetime.datetime.now())
    # Create a graph from the filtered data
    G = nx.Graph()
    # edge_data = []

    # Iterate over the filtered data and add edges
    for i, j, w in zip(filtered_rows_m, filtered_cols_m, filtered_data):
        if i != j:  # Exclude self-loops
            G.add_edge(i, j, weight=w)
            # edge_data.append((i, j, w))

    print("starting communtiy detection (nested model)")
    print("time:-", datetime.datetime.now())

    # state = minimize_nested_blockmodel_dl(G)

    # print("starting newman modularity detection (nested model)")
    # print("time:-", datetime.datetime.now())
    # mod_state = minimize_blockmodel_dl(G, state=gt.ModularityState)

    # print("entropy")
    # print("time:-", datetime.datetime.now())
    # entropy = state.entropy()

    # print("modularity")
    # print("time:-", datetime.datetime.now())
    # modularity_score = mod_state.modularity()

    # # df_edges = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Weight'])

    # print('now community detection')

    # # compute the best partition
    # communities_generator = nx.community.louvain_communities(G)
    # partition = {node: i for i, comm in enumerate(
    #     communities_generator) for node in comm}
    # communities_list = [comm for comm in communities_generator]

    # # return communities_list

    # modularity_score = nx.community.modularity(G, communities_list)
    # coverage, performance = nx.community.partition_quality(G, communities_list)
    # # size = sum(1 for community in communities_list if len(community) < 50)
    # # small_percentage = size / len(communities_list)

    # print("modularity:", modularity_score)
    # print("coverage:", coverage)
    # print("performance:", performance)
    # print("entropy", entropy)
    # print("percentage", percentage)

    # community_sizes = [len(community)/51281 for community in communities_list]
    # print(community_sizes)
    # print('total:', sum(community_sizes)/51281)
    # print('size percentage:', small_percentage)

    # return entropy, percentage

    calc_modularity(threshold=10, user_user_matrix=upper)