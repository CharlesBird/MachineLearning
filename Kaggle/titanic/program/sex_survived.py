import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

survived_m = data.Survived[data.Sex == 'male'].value_counts()
survived_f = data.Survived[data.Sex == 'female'].value_counts()
df = pd.DataFrame({'男性': survived_m, '女性': survived_f})
p = df.plot(kind='bar', stacked=True, color=['blue', 'red'])
plt.title('按性别看获救情况')
plt.xlabel('是否获救')
plt.ylabel('人数')
plt.show()