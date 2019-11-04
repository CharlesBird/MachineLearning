import pandas as pd
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv' )
sibsp = data.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(sibsp.count()['PassengerId'])
print(df)

parch = data.groupby(['Parch', 'Survived'])
df = pd.DataFrame(parch.count()['PassengerId'])
print(df)

print(data.Cabin.value_counts())