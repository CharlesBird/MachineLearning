import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

corrmat = train_df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1.0, square=True)
plt.show()