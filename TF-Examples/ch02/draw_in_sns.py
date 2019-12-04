import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv("data.csv", header=0, names=['x', 'y'])

sns.scatterplot('x', 'y', data=points)
plt.show()