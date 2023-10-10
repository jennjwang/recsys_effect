import pandas as pd
import json

with open("pinterest_iccv/pinterest.subset_iccv_board_pins.json", "r") as json_file:
    board_data = json.load(json_file)

board_df = pd.DataFrame(board_data)
column_names = ['_id', 'board_id', 'board_url', 'cate_id']
board_df.columns = column_names
# print(board_df.head())


with open("pinterest_iccv/pinterest.subset_iccv_board_cate.json", "r") as json_file:
    cate_data = json.load(json_file)

cate_df = pd.DataFrame(cate_data)
column_names = ['_id', 'board_id', 'board_url', 'cate_id']
cate_df.columns = column_names
# print(cate_df.head())

print(len(set(cate_df['cate_id'])))
# 468 unique categories

# users 46000
print(len(cate_df))
