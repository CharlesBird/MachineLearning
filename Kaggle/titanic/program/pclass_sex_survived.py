import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
fig.set(alpha=0.65, facecolor='gray')

plt.title('根据舱等级和性别获救情况')

ax1 = fig.add_subplot(141)
data.Survived[data.Sex == 'female'][data.Pclass != 3].value_counts().plot(kind='bar', label='女性高级舱', color='red')
ax1.set_xticklabels(['获救', '未获救'], rotation=0)
ax1.legend(['女性/高级舱'], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data.Survived[data.Sex == 'female'][data.Pclass == 3].value_counts().plot(kind='bar', label='女性低级舱', color='pink')
ax2.set_xticklabels(['未获救', '获救'], rotation=0)
ax2.legend(['女性/低级舱'], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data.Survived[data.Sex == 'male'][data.Pclass != 3].value_counts().plot(kind='bar', label='男性高级舱', color='blue')
ax3.set_xticklabels(['未获救', '获救'], rotation=0)
ax3.legend(['男性/高级舱'], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data.Survived[data.Sex == 'male'][data.Pclass == 3].value_counts().plot(kind='bar', label='男性低级舱', color='lightblue')
ax4.set_xticklabels(['未获救', '获救'], rotation=0)
ax4.legend(['男性/低级舱'], loc='best')

plt.show()