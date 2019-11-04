import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
data_train = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/test.csv')

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
        'YearBuilt', 'YearRemodAdd', 'PoolArea', 'Fireplaces', 'CentralAir', 'KitchenAbvGr',
        'MSSubClass', 'MSZoning', 'Neighborhood', 'OverallCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond',
        'BsmtFinSF1', 'Heating', 'HeatingQC', '2ndFlrSF', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenQual',
        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
        'MoSold', 'SaleType', 'SaleCondition']

X_train = data_train[cols].copy()
X_test = data_test[cols].copy()


def set_PoolArea(df):
    # 有无泳池设置
    df['HasPool'] = df.PoolArea.map(lambda x: 1 if x > 0 else 0)
    return df


def set_missing_number_data(df):
    # 只有测试集存在缺失
    # GarageCars,GarageArea,TotalBsmtSF,BsmtFinSF1,BsmtFullBath
    val = dict.fromkeys(['GarageCars', 'GarageArea', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFullBath'], 0)
    df = df.fillna(value=val)
    return df

def set_missing_object_data(df):
    # 'BsmtQual', 'BsmtCond', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'
    val = dict.fromkeys(['BsmtQual', 'BsmtCond', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'], 'None')
    # 只有测试集缺失 {'KitchenQual': 'TA','SaleType': 'WD'}
    # 根据分析选择TA或者Gd，根据数据查看赋值WD
    val.update({'KitchenQual': 'TA', 'SaleType': 'WD'})
    df = df.fillna(value=val)
    return df

def set_missing_GarageYrBlt(df):
    # 车库建造日期，空值默认为1800
    # 有个错乱值2207，猜测是2007
    df = df.fillna(value={'GarageYrBlt': 1800})
    df.loc[df.GarageYrBlt == 2207, 'GarageYrBlt'] = 2007
    return df

def set_MSSubClass_level(df):
    # 销售住宅类型等级划分
    df.loc[(df.MSSubClass == 60.) | (df.MSSubClass == 120.), 'MSSubClassLevel'] = 'high'
    df.loc[(df.MSSubClass == 30.) | (df.MSSubClass == 80.), 'MSSubClassLevel'] = 'low'
    df.loc[df.MSSubClassLevel.isnull(), 'MSSubClassLevel'] = 'middle'
    return df

def set_missing_MSZoning(df):
    # 测试集中存在null，销售的一般分区分类
    # 根据MSSubClass 20，30，70分别匹配RL,C (all),RM
    df.loc[(df.MSZoning.isnull()) & (df.MSSubClass == 20.), 'MSZoning'] = 'RL'
    df.loc[(df.MSZoning.isnull()) & (df.MSSubClass == 30.), 'MSZoning'] = 'C (all)'
    df.loc[(df.MSZoning.isnull()) & (df.MSSubClass == 70.), 'MSZoning'] = 'RM'
    return df

def set_Neighborhood_level(df):
    # 城市范围内的位置划分等级，别墅，内环，中环，外环，老镇
    level_dict = {}
    level_dict.update(dict.fromkeys(['NoRidge', 'NridgHt', 'StoneBr'], 'villa'))
    level_dict.update(dict.fromkeys(['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'Somerst', 'Timber', 'Veenker'], 'inner'))
    level_dict.update(dict.fromkeys(['Blueste', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'SawyerW'], 'middle'))
    level_dict.update(dict.fromkeys(['SWISU', 'Sawyer', 'Edwards', 'BrkSide', 'OldTown'], 'oldtown'))
    level_dict.update(dict.fromkeys(['BrDale', 'IDOTRR', 'MeadowV'], 'outer'))
    df['NeighborhoodLevel'] = df.Neighborhood.map(level_dict)
    return df

def common_process(df):
    df = set_PoolArea(df)
    df = set_MSSubClass_level(df)
    df = set_Neighborhood_level(df)
    df = set_missing_GarageYrBlt(df)
    df = set_missing_object_data(df)

    dummies_CentralAir = pd.get_dummies(df['CentralAir'], prefix='CentralAir', dtype='int64')
    dummies_MSZoning = pd.get_dummies(df['MSZoning'], prefix='MSZoning', dtype='int64')
    dummies_MSSubClassLevel = pd.get_dummies(df['MSSubClassLevel'], prefix='MSSubClassLevel', dtype='int64')
    dummies_NeighborhoodLevel = pd.get_dummies(df['NeighborhoodLevel'], prefix='NeighborhoodLevel', dtype='int64')
    dummies_ExterQual = pd.get_dummies(df['ExterQual'], prefix='ExterQual', dtype='int64')
    dummies_Foundation = pd.get_dummies(df['Foundation'], prefix='Foundation', dtype='int64')
    dummies_BsmtQual = pd.get_dummies(df['BsmtQual'], prefix='BsmtQual', dtype='int64')
    dummies_BsmtCond = pd.get_dummies(df['BsmtCond'], prefix='BsmtCond', dtype='int64')
    # dummies_Heating = pd.get_dummies(df['Heating'], prefix='Heating', dtype='int64')
    dummies_HeatingQC = pd.get_dummies(df['HeatingQC'], prefix='HeatingQC', dtype='int64')
    dummies_KitchenQual = pd.get_dummies(df['KitchenQual'], prefix='KitchenQual', dtype='int64')
    dummies_GarageType = pd.get_dummies(df['GarageType'], prefix='GarageType', dtype='int64')
    dummies_GarageFinish = pd.get_dummies(df['GarageFinish'], prefix='GarageFinish', dtype='int64')
    # dummies_GarageQual = pd.get_dummies(df['GarageQual'], prefix='GarageQual', dtype='int64')
    dummies_GarageCond = pd.get_dummies(df['GarageCond'], prefix='GarageCond', dtype='int64')
    dummies_PavedDrive = pd.get_dummies(df['PavedDrive'], prefix='PavedDrive', dtype='int64')
    dummies_SaleType = pd.get_dummies(df['SaleType'], prefix='SaleType', dtype='int64')
    dummies_SaleCondition = pd.get_dummies(df['SaleCondition'], prefix='SaleCondition', dtype='int64')
    df = pd.concat([df, dummies_CentralAir, dummies_MSZoning, dummies_MSSubClassLevel, dummies_NeighborhoodLevel, dummies_ExterQual,
                    dummies_Foundation, dummies_BsmtQual, dummies_BsmtCond, dummies_HeatingQC, dummies_KitchenQual,
                    dummies_GarageType, dummies_GarageFinish, dummies_GarageCond, dummies_PavedDrive, dummies_SaleType,
                    dummies_SaleCondition], axis=1)
    df.drop(['MSZoning', 'CentralAir', 'MSSubClassLevel', 'NeighborhoodLevel', 'ExterQual', 'Foundation',
             'Neighborhood', 'BsmtQual', 'BsmtCond', 'Heating', 'HeatingQC', 'KitchenQual', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition', 'PoolArea', 'MSSubClass'], axis=1, inplace=True)
    return df


def get_part_data_scaler(X_train, X_test):
    # 数据标准化
    scale_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
                  'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'KitchenAbvGr', 'OverallCond',
                  'BsmtFinSF1', '2ndFlrSF', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'GarageYrBlt', 'MoSold']
    scaled_cols = [future+'_scaled' for future in scale_cols]
    scale_train_values = X_train[scale_cols].values
    scale_test_values = X_test[scale_cols].values
    standardScaler = StandardScaler()
    standardScaler.fit(scale_train_values)
    scaled_train_values = standardScaler.transform(scale_train_values)
    scaled_test_values = standardScaler.transform(scale_test_values)

    df_train_scaled = pd.DataFrame(scaled_train_values, columns=scaled_cols)
    df_test_scaled = pd.DataFrame(scaled_test_values, columns=scaled_cols)

    X_train_scaled = pd.concat([X_train, df_train_scaled], axis=1)
    X_test_scaled = pd.concat([X_test, df_test_scaled], axis=1)

    X_train_scaled.drop(scale_cols, axis=1, inplace=True)
    X_test_scaled.drop(scale_cols, axis=1, inplace=True)
    return X_train_scaled, X_test_scaled


def get_all_data_scaler(X_train, X_test):
    scale_train_values = X_train.values
    scale_test_values = X_test.values
    standardScaler = StandardScaler()
    standardScaler.fit(scale_train_values)
    scaled_train_values = standardScaler.transform(scale_train_values)
    scaled_test_values = standardScaler.transform(scale_test_values)

    X_train_scaled = pd.DataFrame(scaled_train_values, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaled_test_values, columns=X_test.columns)
    return X_train_scaled, X_test_scaled


def data_train_process():
    global X_train

    data_res = common_process(X_train)
    return data_res


def data_test_process():
    global X_test

    data_res = set_missing_MSZoning(X_test)
    data_res = set_missing_number_data(data_res)
    data_res = common_process(data_res)
    return data_res


def process_main():
    X_train_df = data_train_process()
    X_test_df = data_test_process()
    X_train_scaled, X_test_scaled = get_part_data_scaler(X_train_df, X_test_df)
    # X_train_scaled, X_test_scaled = get_all_data_scaler(X_train_df, X_test_df)
    X_train_scaled = pd.concat([data_train['SalePrice'], X_train_scaled], axis=1)
    X_test_scaled = pd.concat([data_test['Id'], X_test_scaled], axis=1)
    return X_train_scaled, X_test_scaled


if __name__ == '__main__':
    train_res, test_res = process_main()
    train_res.info()
    test_res.info()
    print(train_res.shape, test_res.shape)
