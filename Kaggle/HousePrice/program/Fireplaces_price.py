import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')

# plt.scatter(train_df.Fireplaces, train_df.SalePrice)
# plt.grid(b=True, which='major', axis='y')
# plt.ylabel('价格')
# plt.title('火炉对房价影响')

data = pd.concat([train_df['SalePrice'], train_df['Fireplaces']], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='Fireplaces', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()
"""
有关联性
"""