import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy
import os

train_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
test_df = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

sns.distplot(train_df['SalePrice'], fit=norm)
fig = plt.figure()
scipy.stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()