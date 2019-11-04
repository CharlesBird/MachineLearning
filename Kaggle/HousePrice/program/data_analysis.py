import pandas as pd
import re
import os
data = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/test.csv')
print('--------查看缺失值-----------')
data.info()
print('--------数值型数据-----------')
print(data.describe())

# 峰度：峰度（Kurtosis）是描述某变量所有取值分布形态陡缓程度的统计量。
# 它是和正态分布相比较的。
# 1、Kurtosis=0 与正态分布的陡缓程度相同。
# 2、Kurtosis>0 比正态分布的高峰更加陡峭——尖顶峰
# 3、Kurtosis<0 比正态分布的高峰来得平台——平顶峰计算公式：β = M_4 /σ^4 偏度：
# 偏度：偏度（Skewness）是描述某变量取值分布对称性的统计量。
# Skewness=0 分布形态与正态分布偏度相同
# Skewness>0 正偏差数值较大，为正偏或右偏。长尾巴拖在右边。
# Skewness<0 负偏差数值较大，为负偏或左偏。长尾巴拖在左边。 计算公式： S= (X^ - M_0)/δ Skewness 越大，分布形态偏移程度越大。
print('--------偏度，峰度-----------')
print("偏度: %f" % data['SalePrice'].skew())
print("峰度: %f" % data['SalePrice'].kurt())

print('--------测试数据查看缺失值-----------')
data_test.info()
print('--------测试数据数值型数据-----------')
print(data_test.describe())
