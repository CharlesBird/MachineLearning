import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8, 6), dpi=80)
fig.set(alpha=0.8, facecolor='gray')

plt.subplot2grid((2, 3), (0, 0))
data.Survived.value_counts().plot(kind='bar')
plt.title('获救情况（1为获救）')
plt.ylabel('人数')

plt.subplot2grid((2, 3), (0, 1))
data.Pclass.value_counts().plot(kind='bar')
plt.title('乘客仓位分布')
plt.ylabel('人数')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data.Survived, data.Age)
plt.grid(b=True, which='major', axis='y')
plt.ylabel('年龄')
plt.title('按年龄获救情况（1为获救）')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data.Age[data.Pclass == 1].plot(kind='kde')
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 3].plot(kind='kde')
plt.grid(b=True, which='major')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.title('各等级的乘客年龄分布')
plt.legend(('头等舱', '2等舱', '3等舱'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data.Embarked.value_counts().plot(kind='bar')
plt.title('登船口岸上船人数')
plt.ylabel('人数')

plt.show()