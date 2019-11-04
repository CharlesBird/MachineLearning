import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split

data_train = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/train.csv')


def find_SVC_hyper_param(X, y):
    param_grid = [
        {
            'C': [1000, 100, 10, 1],
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            # 'degree': [3, 4, 5],
            'gamma': ['auto', 1, 0.1, 0.01, 0.001],
            # 'tol': [1e-3, 1e-4, 1e-5],
            # 'verbose': [True, False],
            # 'random_state': [0, 100, 200, 500]
        },
    ]
    grid_search = GridSearchCV(SVC(), param_grid, n_jobs=-1, verbose=1, cv=5)
    grid_search.fit(X, y)
    print('best_SVC_reg_score', grid_search.best_score_)
    # 0.8268260827212238
    print('best_SVC_reg_param', grid_search.best_params_)
    # best_rid_reg_param {'alphas': (0.1, 1.0, 10.0)}
    return grid_search.best_estimator_


if __name__ == '__main__':
    train_np = data_train.values
    # for i in list(range(1000, 45000, 5000)):
    #     y = train_np[:i, 0]
    #     X = train_np[:i, 1:]
    #     find_SVC_hyper_param(X, y)

    y = train_np[:, 0]
    X = train_np[:, 1:]
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    pca = PCA(n_components=0.98)
    pca.fit(X_std)
    X_std_pca = pca.transform(X_std)
    # var_per = pca.explained_variance_ratio_
    # cum_var_per = pca.explained_variance_ratio_.cumsum()
    # n_comp = len(cum_var_per[cum_var_per <= 0.98])
    # print(var_per, cum_var_per, n_comp)
    # print(X_std_pca.shape)

    find_SVC_hyper_param(X_std_pca, y)
