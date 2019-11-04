import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
train_df['HasPool'] = train_df.PoolArea.map(lambda x: 1 if x > 0 else 0)

# plt.scatter(train_df.HasPool, train_df.SalePrice)
# plt.grid(b=True, which='major', axis='y')
# plt.ylabel('价格')
# plt.title('有无游泳池对房价影响')

data = pd.concat([train_df['SalePrice'], train_df['HasPool']], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='HasPool', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()
"""
关联不是很强，可以列入
"""