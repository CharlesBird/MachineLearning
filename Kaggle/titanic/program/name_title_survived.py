import pandas as pd
import matplotlib.pyplot as plt
import os
import re
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data['Title'] = data.Name.map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
survived_0 = data.Title[data.Survived == 0].value_counts()
survived_1 = data.Title[data.Survived == 1].value_counts()
df = pd.DataFrame({'未获救': survived_0, '获救': survived_1})
p = df.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(15, 8))
plt.title('乘客称谓获救情况')
plt.xlabel('乘客称谓')
plt.ylabel('人数')

plt.show()