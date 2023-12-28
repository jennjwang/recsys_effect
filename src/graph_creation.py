import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv

# CREATE A GRAPH BASED ON MODULARITY (step 3)

user_user_matrix = load_npz('data/jaccard_similarity.npz')
non_zero_values = user_user_matrix.data

print(len(non_zero_values))

def graph_coverage(clipped_data):

    print(max(clipped_data))
    print(min(clipped_data))
    print(len(np.unique(clipped_data)))


    unique_values, counts = np.unique(clipped_data, return_counts=True)

    # Normalize the frequencies
    normalized_frequencies = counts / sum(counts)

    for value, freq, count in zip(unique_values, normalized_frequencies, counts):
        print(f"Value: {value}, Normalized Frequency: {freq:.4f}, Actual Frequency: {count}")

    # Plot the bar chart
    plt.bar(unique_values, normalized_frequencies, edgecolor='black')
    plt.title('Normalized Frequencies of Unique Values')
    plt.xlabel('Value')
    plt.ylabel('Normalized Frequency')
    plt.show()

graph_coverage(non_zero_values)

def calc_modularity(threshold):

    mask = user_user_matrix.data >= threshold
    # print(mask)

    filtered_rows, filtered_cols = user_user_matrix.nonzero()
    # print(filtered_cols)
    # print(filtered_rows)

    filtered_rows = filtered_rows[mask]
    filtered_cols = filtered_cols[mask]
    filtered_data = user_user_matrix.data[mask]
    # print(filtered_data)
    print(len(filtered_data))
    print(len(non_zero_values))

    percentage = len(filtered_data) / len(non_zero_values)
    print(percentage)

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

    # return communities_list

    modularity_score = nx.community.modularity(G, communities_list)
    coverage, performance = nx.community.partition_quality(G, communities_list)
    # size = sum(1 for community in communities_list if len(community) < 50)
    # small_percentage = size / len(communities_list)
    percentage = len(filtered_data) / len(non_zero_values)

    print("modularity:", modularity_score)
    print("coverage:", coverage)
    print("performance:", performance)
    print("percentage", )

    # community_sizes = [len(community)/51281 for community in communities_list]
    # print(community_sizes)
    # print('total:', sum(community_sizes)/51281)
    # print('size percentage:', small_percentage)

    return modularity_score, coverage, performance, percentage

mod = []
cov = []
perf = []
perc = []

# communities = calc_modularity(threshold=2)

# file_path = 'communities.csv'

# # Open the CSV file in write mode
# with open(file_path, 'w', newline='') as csvfile:
#     # Create a CSV writer
#     csvwriter = csv.writer(csvfile)
    
#     # Write each community as a row in the CSV file
#     for community in communities:
#         csvwriter.writerow(community)

thrs = range(2, 11)
for i in thrs:
    calc_modularity(threshold=i)
    mod_score, coverage, performance, percentage = calc_modularity(threshold=i)
    mod.append(mod_score)
    cov.append(coverage)
    perf.append(performance)
    perc.append(percentage)
    

plt.plot(thrs, mod, label='y = modularity', color='blue')
plt.plot(thrs, cov, label='y = coverage', color='red')
plt.plot(thrs, perf, label='y = performance', color='green')
plt.plot(thrs, perc, label='y = percentage of users', color='orange')

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
