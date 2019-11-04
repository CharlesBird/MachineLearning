import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data['Name_lenth'] = data.Name.apply(len)
print(data.Name_lenth.value_counts())
plt.scatter(data.Survived, data.Name_lenth)
plt.grid(b=True, which='major', axis='y')
plt.ylabel('名字长度')
plt.title('按名字长度获救情况（1为获救）')
plt.show()