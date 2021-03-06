# ReLU 函数导数

import numpy as np


def derivative(x):  # ReLU 函数的导数
    d = np.array(x, copy=True)  # 用于保存梯度的张量
    d[x < 0] = 0  # 元素为负的导数为0
    d[x >= 0] = 1  # 元素为正的元素导数为1
    return d