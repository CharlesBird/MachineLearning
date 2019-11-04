import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# fig = plt.figure()
# fig.set(alpha=0.2, facecolor='gray')

survived_0 = data.Embarked[data.Survived == 0].value_counts()
survived_1 = data.Embarked[data.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived_1, '未获救': survived_0})
p = df.plot(kind='bar', stacked=True, color=['blue', 'red'])
plt.title('各港口乘客获救情况')
plt.xlabel('登陆港口')
plt.ylabel('人数')

plt.show()