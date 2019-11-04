import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_train = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
sns.distplot(data_train['SalePrice'])
plt.show()