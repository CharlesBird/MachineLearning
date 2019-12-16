# LeakyReLU 函数导数
import numpy as np


# 其中p 为LeakyReLU 的负半段斜率
def derivative(x, p):
    dx = np.ones_like(x)  # 创建梯度张量
    dx[x < 0] = p  # 元素为负的导数为p
    return dx