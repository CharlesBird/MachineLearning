# 激活函数导数
# Sigmoid 函数导数
import numpy as np


def sigmoid(x):  # sigmoid 函数
    return 1 / (1 + np.exp(-x))


def derivative(x):  # sigmoid 导数的计算
    return sigmoid(x) * (1 - sigmoid(x))