import pandas as pd
import re
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
data_test = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv' )
print('--------查看缺失值-----------')
data.info()
print('--------数值型数据-----------')
print(data.describe())
print('--------存活数量-----------')
print(data.Survived.value_counts())
print('--------仓位人数-----------')
print(data.Pclass.value_counts())
print('--------港口登陆人数-----------')
print(data.Embarked.value_counts())
print('--------男女人数-----------')
print(data.Sex.value_counts())
print('--------间接亲属-----------')
print(data.SibSp.value_counts())
print('--------直接亲属-----------')
print(data.Parch.value_counts())
print('--------称呼统计-----------')
data_train_test = data.append(data_test, sort=False)
data_train_test['Title'] = data_train_test.Name.map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
print(data_train_test.Title.value_counts())
print('--------查看Test数据缺失值-----------')
data_test.info()
print('--------查看Test数值型数据-----------')
print(data_test.describe())
a = data.Fare.groupby(by=data_test['Pclass']).mean()
print(a, a.get([3]).values[0])