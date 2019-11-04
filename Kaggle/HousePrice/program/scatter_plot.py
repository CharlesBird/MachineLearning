import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], height=2.5)
plt.show()
