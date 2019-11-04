import numpy as np


def accuracy_score(y_true, y_predict):
    """计算y_true与y_predict之间的准确率"""
    assert len(y_predict) == len(y_true), "the size of y_predict is equal to the size of y_true."

    return np.sum(y_predict == y_true) / len(y_true)