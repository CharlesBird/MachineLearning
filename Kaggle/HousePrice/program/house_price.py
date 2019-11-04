import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

prices = pd.DataFrame({'price': train_df['SalePrice'], 'log(price+1)': np.log1p(train_df['SalePrice'])})
ps = prices.hist()
plt.plot()
plt.show()

# y_train = np.log1p(train_df.pop('SalePrice'))
# all_df = pd.concat((train_df, test_df), axis=0)
# print(all_df.shape)
# print(y_train.head())

# explore = train_df.describe(include='all').T
# print(explore)

