import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

# TO UNDERSTAND THE MAKEUP OF THE COMMUNITIES (step 5)

mapping = pd.read_csv("data/user_id_to_code_mapping.csv")

map_dict = {}

# Iterate through each row in the DataFrame
for index, row in mapping.iterrows():
    key = row['code']
    value = row['user_id']
    map_dict[key] = value
    

print(map_dict)


with open('data/communities.csv', newline='') as file:
    reader = csv.reader(file)
    comms = list(reader)


a = [len(l) for l in comms]

plt.bar(x=range(len(comms)), height=a)
plt.show()

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
behaviors_df = behaviors_df[~behaviors_df['impressions'].str.endswith('0')]
behaviors_df['impressions'] = behaviors_df['impressions'].apply(lambda cell: cell[:-2])
behaviors_df = behaviors_df.rename(columns={'impressions':'news_id'})
behaviors_df = behaviors_df.merge(cate_df, on='news_id')


flattened_data = [(value, index) for index, sublist in enumerate(comms) for value in sublist]

# Create a DataFrame from the flattened data
comms_map = pd.DataFrame(flattened_data, columns=['elm', 'comm_ind'])

comms_map['user_id'] = comms_map['elm'].apply(lambda e: map_dict[int(e)])
print(comms_map.head())
print(len(comms_map))

print(behaviors_df.head())

print(len(behaviors_df))
behaviors_df = behaviors_df.merge(comms_map, on='user_id')
print(len(behaviors_df))
print(behaviors_df.head())


grouped = behaviors_df.groupby(['cat', 'comm_ind']).size().unstack(fill_value=0)

# Calculate the percentage
percentage = grouped.div(grouped.sum(axis=1), axis=0) * 100

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF', '#FF551a', '#55FF1a', '#FF1a55']

### Plot stacked category for communities
# # Plot the percentage
# percentage.plot(kind='bar', stacked=True, color=colors)

# # Set the labels and title
# plt.xlabel('Community')
# plt.ylabel('Percentage')
# plt.title('Percentage of News Category for each community')

# # Show the legend
# plt.legend(title='Category', loc='upper left', bbox_to_anchor=(1, 1))

# # Show the plot
# plt.show()

# df = behaviors_df
# communities = df['comm_ind'].unique()

# print(len(communities))

# n_rows = 4  # For example, 2 rows
# n = 5
# n_cols = int(np.ceil(n / n_rows))

# # Create a figure
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
# axs = axs.flatten()

# unique_cats = df['cat'].unique()
# print(unique_cats)
# # colors_dict = {cat: color for cat, color in zip(unique_cats, plt.cm.tab200.colors)}

# n = 0
# # 3, 4, 6, 11, 12 
# for i in range(14):
#     if i not in [3, 4, 6, 11, 12]:
#         continue
#     n += 1
#     community = i
#     # Filter the data for the current community
#     community_data = df[df['comm_ind'] == community]
    
#     # Group by 'cat', then count the occurrences
#     grouped = community_data.groupby('subcat').size()

#     print(grouped.index)

#     # community_colors = [colors_dict[cat] for cat in grouped.index]
    
#     # Plot the data
#     grouped.plot(kind='bar', ax=axs[n])

#     if len(community_data) != 0:
    
#         # Set the labels and title
#         axs[n].set_xlabel('Category')
#         axs[n].set_ylabel('Count')
#         axs[n].set_title(f'Category Distribution for Community {community}')

# # Adjust the layout
# plt.tight_layout()

# # Show the plot
# plt.show()