import pandas as pd
import re
import os
data_train = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/test.csv')
print('--------查看缺失值-----------')
data_train.info()
print('--------数值型数据-----------')
print(data_train.describe())


print('--------测试数据查看缺失值-----------')
data_test.info()
print('--------测试数据数值型数据-----------')
print(data_test.describe())

print(data_train.shape)
print(data_test.shape)

# 参考方案
# CNN
# Torch
# Std + PCA + svm/knn
# PCA + rfc
# https://www.kaggle.com/mkashif/dimensionality-reduction-and-viz-of-mnist-dataset
