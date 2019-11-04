import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

k = 10
plt.subplots(figsize=(12, 9))
corrmat = train_df.corr()
cols = corrmat.nlargest(k, 'GarageCars')['GarageCars'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.3f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()