import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

data = pd.concat([train_df['SalePrice'], train_df['Neighborhood']], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='Neighborhood', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()