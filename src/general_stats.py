import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##### GET GENERAL STATS (step 1) ####
cate_df = pd.read_csv('data/MINDsmall_train/news.tsv', delimiter='\t')
cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

# 51281 unique news items
# print(cate_df['news_id'].nunique())

# # 285 subcategories
# print(len(set(cate_df['subcat'])))

# # 18 categories in large
# print(len(set(cate_df['cat'])))

subcat_distribution = cate_df['cat'].value_counts()
print(subcat_distribution)
subcat_distribution = subcat_distribution.reset_index(name='count')

# # Create a bar plot
# plt.figure(figsize=(10, 6))
# plt.bar(subcat_distribution['cat'], subcat_distribution['count'])
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.title('Category Distribution')
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# plt.show()

# 101526 news items
# 51281 


# # Convert the result to a DataFrame and reset the index to make it more readable
# subcat_distribution_df = subcat_distribution.reset_index(name='count')

# sorted_df = subcat_distribution_df.sort_values(by='count', ascending=False)

# # Reset the index for the sorted DataFrame
# sorted_df.reset_index(drop=True, inplace=True)

# print(sorted_df)

behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv', delimiter='\t')
behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

# 50000
print('users', behaviors_df['user_id'].nunique())
# 156964
print('impressions sessions', behaviors_df['impression_id'].nunique())

