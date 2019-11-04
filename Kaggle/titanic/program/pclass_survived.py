import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

survived_0 = data.Pclass[data.Survived == 0].value_counts()
survived_1 = data.Pclass[data.Survived == 1].value_counts()
df = pd.DataFrame({'未获救': survived_0, '获救': survived_1})
p = df.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(15, 8))
plt.title('乘客等级获救情况')
plt.xlabel('乘客等级')
plt.ylabel('人数')

plt.show()