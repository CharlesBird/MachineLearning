import pandas as pd
import re
import os
data_train = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
        'YearBuilt', 'YearRemodAdd', 'PoolArea', 'Fireplaces', 'CentralAir', 'KitchenAbvGr',
        'MSSubClass', 'MSZoning', 'Neighborhood', 'OverallCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond',
        'BsmtFinSF1', 'Heating', 'HeatingQC', '2ndFlrSF', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenQual',
        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
        'MoSold', 'SaleType', 'SaleCondition']
X_train = data_train[cols]
X_train.info()

X_test = data_test[cols]
X_test.info()

train_des = X_train.describe()
print(train_des)
test_des = X_test.describe()
print(test_des)