import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

user_user_matrix = load_npz('sparse_matrix.npz')

non_zero_values = user_user_matrix.data
# clipped_data = non_zero_values

def calc_modularity(threshold):

    mask = user_user_matrix.data >= threshold
    clipped_data = user_user_matrix.data[mask]

    # print(max(clipped_data))
    # print(min(clipped_data))
    # print(len(np.unique(clipped_data)))

    unique_values, counts = np.unique(clipped_data, return_counts=True)

    # Normalize the frequencies
    normalized_frequencies = counts / sum(counts)

    # for value, freq, count in zip(unique_values, normalized_frequencies, counts):
    #     print(f"Value: {value}, Normalized Frequency: {freq:.4f}, Actual Frequency: {count}")

    # # Plot the bar chart
    # plt.bar(unique_values, normalized_frequencies, edgecolor='black')
    # plt.title('Normalized Frequencies of Unique Values')
    # plt.xlabel('Value')
    # plt.ylabel('Normalized Frequency')
    # plt.show()

    filtered_rows, filtered_cols = user_user_matrix.nonzero()
    filtered_rows = filtered_rows[mask]
    filtered_cols = filtered_cols[mask]
    filtered_data = user_user_matrix.data[mask]

    print(f'making graph with threshold {threshold}')
    # Create a graph from the filtered data
    G = nx.Graph()
    # edge_data = []

    # Iterate over the filtered data and add edges
    for i, j, w in zip(filtered_rows, filtered_cols, filtered_data):
        if i != j:  # Exclude self-loops
            G.add_edge(i, j, weight=w)
            # edge_data.append((i, j, w))

    # df_edges = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Weight'])

    print('now community detection')

    # compute the best partition
    communities_generator = nx.community.louvain_communities(G)
    partition = {node: i for i, comm in enumerate(communities_generator) for node in comm}
    communities_list = [comm for comm in communities_generator]

    modularity_score = nx.community.modularity(G, communities_list)
    coverage, performance = nx.community.partition_quality(G, communities_list)

    print("modularity:", modularity_score)
    print("coverage:", coverage)
    print("performance:", performance)

    return modularity_score, coverage, performance

mod = []
cov = []
perf = []

thrs = range(2, 11)
for i in thrs:
    mod_score, coverage, performance = calc_modularity(threshold=i)
    mod.append(mod_score)
    cov.append(coverage)
    perf.append(performance)
    

plt.plot(thrs, mod, label='y = modularity', color='blue')
plt.plot(thrs, cov, label='y = coverage', color='red')
plt.plot(thrs, perf, label='y = performance', color='green')

plt.title("Threshold vs Metrics")
plt.xlabel("threshold")
plt.legend()

plt.show()

# pos = nx.spring_layout(G)  # Use spring layout for positioning
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, width=0.5)
# plt.title("User-User Graph")
# plt.show()
# df_edges.to_csv('small_graph.csv', index=False)

# print(G.edges())
# cate_df = pd.read_csv('MINDlarge_train/news.tsv', delimiter='\t')
# cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

# # 285 subcategories
# print(len(set(cate_df['subcat'])))

# # 18 categories in large
# print(len(set(cate_df['cat'])))

# subcat_distribution = cate_df.groupby('cat')['subcat'].value_counts()

# # Convert the result to a DataFrame and reset the index to make it more readable
# subcat_distribution_df = subcat_distribution.reset_index(name='count')

# sorted_df = subcat_distribution_df.sort_values(by='count', ascending=False)

# # Reset the index for the sorted DataFrame
# sorted_df.reset_index(drop=True, inplace=True)

# print(sorted_df)

# # 101526 users
# print(len(cate_df))

