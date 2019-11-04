from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
import os

from data_process import process_main


def forest_polynomialregression(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('rf_reg', RandomForestRegressor(n_estimators=100, oob_score=True, random_state=500))
    ])


def gb_polynomialregression(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('gb_reg', GradientBoostingRegressor(loss='ls', max_depth=3, max_leaf_nodes=10, min_samples_leaf=1, n_estimators=200, random_state=100))
    ])


def ls_polynomialregression(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('ls_reg', LassoCV(eps=1e-3, cv=4, max_iter=5000, random_state=100))
    ])


def get_training_goals(X, y, X_test):
    # 集成学习
    voting_reg = VotingRegressor(estimators=[
        ('rf_ploy', forest_polynomialregression(degree=3)),
        ('gb_ploy', gb_polynomialregression(degree=3)),
        ('ls_ploy', ls_polynomialregression(degree=3)),
        # ('rf_reg', RandomForestRegressor(n_estimators=100, oob_score=True, random_state=500)),
        # ('gb_reg', GradientBoostingRegressor(loss='ls', max_depth=3, max_leaf_nodes=10, min_samples_leaf=1, n_estimators=200, random_state=100)),
        # ('ls_reg', LassoCV(eps=1e-3, cv=4, max_iter=5000, random_state=100))
    ], weights=[0.2, 0.6, 0.2])
    voting_reg.fit(X, y)
    predict_y = voting_reg.predict(X_test)
    return predict_y


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score
    data_train, data_test = process_main()
    train_np = data_train.values
    X_test = data_test.values[:, 1:]
    y = train_np[:, 0]
    X = train_np[:, 1:]
    predict = get_training_goals(X, y, X_test)
    # data_y = pd.read_csv(os.path.abspath(os.path.dirname(os.getcwd())) + '/gender_submission.csv')
    # y = data_y['Survived'].values
    # score = accuracy_score(y, predict.astype(np.int64))
    # print(score)
    result = pd.DataFrame({'Id': data_test['Id'].values, 'SalePrice': predict.astype(np.float)})
    result.to_csv(os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), "../.."))) + '/predict_part_scaled_ploy-3.csv', index=False)