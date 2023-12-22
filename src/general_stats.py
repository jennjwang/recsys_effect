import pandas as pd
import matplotlib.pyplot as plt

##### GET GENERAL STATS (step 1) ####
cate_df = pd.read_csv('data/MINDsmall_train/news.tsv', delimiter='\t')
cate_df.columns = ['news_id', 'cat', 'subcat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

# # 285 subcategories
# print(len(set(cate_df['subcat'])))

# # 18 categories in large
# print(len(set(cate_df['cat'])))

subcat_distribution = cate_df['cat'].value_counts()
print(subcat_distribution)
subcat_distribution = subcat_distribution.reset_index(name='count')

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(subcat_distribution['cat'], subcat_distribution['count'])
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

# # Convert the result to a DataFrame and reset the index to make it more readable
# subcat_distribution_df = subcat_distribution.reset_index(name='count')

# sorted_df = subcat_distribution_df.sort_values(by='count', ascending=False)

# # Reset the index for the sorted DataFrame
# sorted_df.reset_index(drop=True, inplace=True)

# print(sorted_df)

# 101526 users
# 51281 
print(len(cate_df))

