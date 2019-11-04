import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')

key = 'Neighborhood'
# fields = ['MSSubClass', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
# for key in fields:

train_df.loc[(train_df[key].isnull()), key] = 'None'

plt.subplots(figsize=(20, 12))
plt.scatter(train_df[key], train_df.SalePrice)

plt.grid(b=True, which='major', axis='y')
plt.ylabel('价格')
plt.title('%s对房价影响' % key)

data = pd.concat([train_df['SalePrice'], train_df[key]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=key, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()