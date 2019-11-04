from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from data_process import data_train_process

data_train = data_train_process()
train_np = data_train.values
y = train_np[:, 0]
X = train_np[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.fit(X_train, y_train)
res2 = lin_reg.score(X_test, y_test)
print('lin_reg: ', res2)

rid_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
rid_reg.fit(X_train, y_train)
rid_reg.fit(X_train, y_train)
res4 = rid_reg.score(X_test, y_test)
print('rid_reg: ', res4)


forest_reg = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=500)
forest_reg.fit(X_train, y_train)
forest_reg.fit(X_train, y_train)
res5 = forest_reg.score(X_test, y_test)
print('forest_reg: ', res5)

grad_reg = GradientBoostingRegressor(n_estimators=500)
grad_reg.fit(X_train, y_train)
grad_reg.fit(X_train, y_train)
res6 = grad_reg.score(X_test, y_test)
print('grad_reg: ', res6)

ada_reg = AdaBoostRegressor(n_estimators=200)
ada_reg.fit(X_train, y_train)
ada_reg.fit(X_train, y_train)
res7 = ada_reg.score(X_test, y_test)
print('ada_reg: ', res7)


decision_reg = DecisionTreeRegressor(random_state=333, min_samples_leaf=3, max_leaf_nodes=5)
decision_reg.fit(X_train, y_train)
decision_reg.fit(X_train, y_train)
res8 = decision_reg.score(X_test, y_test)
print('decision_reg: ', res8)

extraTree_reg = ExtraTreeRegressor(random_state=333, min_samples_leaf=3, max_leaf_nodes=5)
extraTree_reg.fit(X_train, y_train)
extraTree_reg.fit(X_train, y_train)
res9 = extraTree_reg.score(X_test, y_test)
print('extraTree_reg: ', res9)
