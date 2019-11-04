import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练集X计算出均值和标准差"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:, i]) for i in np.arange(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in np.arange(X.shape[1])])
        return self

    def transform(self, X):
        """根据均值和标准差将数据集X进行均值标准差归一化"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform!"
        assert X.shape[1] == len(self.mean_), "The size of X must be equal to the size of mean_ and scale_"

        resX = np.empty(shape=X.shape, dtype='float')
        for col in np.arange(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX