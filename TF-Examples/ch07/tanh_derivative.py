# Tanh 函数梯度
import numpy as np

def sigmoid(x):  # sigmoid 函数实现
    return 1 / (1 + np.exp(-x))


def tanh(x):  # tanh 函数实现
    return 2 * sigmoid(2 * x) - 1


def derivative(x):  # tanh 导数实现
    return 1 - tanh(x) ** 2