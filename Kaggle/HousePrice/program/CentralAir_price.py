import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')

train_df['HasCentralAir'] = train_df.CentralAir.map(lambda x: 1 if x == 'Y' else 0)
# plt.scatter(train_df.HasCentralAir, train_df.SalePrice)
# plt.grid(b=True, which='major', axis='y')
# plt.ylabel('价格')
# plt.title('有无中央空调对房价影响')

data = pd.concat([train_df['SalePrice'], train_df['HasCentralAir']], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='HasCentralAir', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()