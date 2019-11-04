import pandas as pd
import os
data_train = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')


def get_Neighborhood_data(df):
    # 城市范围内的位置划分等级，别墅，内环，中环，外环，古镇
    avg_group_datas = df.groupby('Neighborhood')['SalePrice'].mean()
    keys = avg_group_datas.index
    avg_valus = avg_group_datas.values
    avg_dict = dict(zip(keys, avg_valus))
    mid_group_datas = df.groupby('Neighborhood')['SalePrice'].median()
    keys = mid_group_datas.index
    mid_valus = mid_group_datas.values
    mid_dict = dict(zip(keys, mid_valus))
    min_group_datas = df.groupby('Neighborhood')['SalePrice'].min()
    keys = min_group_datas.index
    min_valus = min_group_datas.values
    min_dict = dict(zip(keys, min_valus))
    max_group_datas = df.groupby('Neighborhood')['SalePrice'].max()
    keys = max_group_datas.index
    max_valus = max_group_datas.values
    max_dict = dict(zip(keys, max_valus))
    for key in keys:
        print(key, avg_dict[key], mid_dict[key], min_dict[key], max_dict[key])
    return df


if __name__ == '__main__':
    get_Neighborhood_data(data_train)