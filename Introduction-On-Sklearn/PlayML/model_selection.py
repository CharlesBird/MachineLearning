import numpy as np


def train_test_split(X, y, test_radio=0.2, seed=None):
    """将数据X和y按照test_radio分割为X_trian，y_train，X_test，y_test"""
    assert X.shape[0] == y.shape[0], "the size of X is equal to the size of y."
    assert 0.0 <= test_radio < 1.0, "test_radio must be valid"

    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(shuffle_indexes) * test_radio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test