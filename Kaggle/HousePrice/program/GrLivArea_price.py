import pandas as pd
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')

# plt.scatter(train_df.LotArea, train_df.SalePrice)
plt.scatter(train_df.GrLivArea, train_df.SalePrice)
# plt.grid(b=True, which='major', axis='y')
plt.ylabel('价格')
plt.title('房子面积对房价的影响')
plt.show()