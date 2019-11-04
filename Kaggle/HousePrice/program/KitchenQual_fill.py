# 填充测试集中的KitchenQual 的值
import pandas as pd
import os
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')

res = train_df[train_df.KitchenAbvGr == 1].groupby('KitchenQual').count()
print(res)