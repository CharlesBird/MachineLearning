import numpy as np


class LinearRegression:

    def __int__(self):
        """构造方法"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_nomal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be equal to the size of y_train"

        X_b = np.vstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"

        X_b = np.vstack(np.ones((len(X_predict), 1), 1), X_predict)
        return X_b.dot(self._theta)