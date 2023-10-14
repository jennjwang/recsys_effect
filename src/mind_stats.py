import pandas as pd

##### GET GENERAL STATS ####
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

