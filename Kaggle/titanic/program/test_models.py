from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from data_preprocess import data_process

# test_df = data_test.filter(regex='Name_lenth_.*|Age_.*|SibSp_.*|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X_test = test_df.values

data_train, data_test = data_process()
train_df = data_train.filter(regex='Survived|Name_lenth_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass|Title_.*|Agelevel_.*')
train_np = train_df.values
y = train_np[:, 0]
X = train_np[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 逻辑回归分类器
# {'C': 0.5, 'penalty': 'l1', 'tol': 0.0001}
log_reg = LogisticRegression(penalty='l2', solver='newton-cg', tol=1e-4, C=0.5, multi_class='multinomial')
log_reg.fit(X_train, y_train)
predict_y = log_reg.predict(X_test)
score = log_reg.score(X_test, y_test)
print('log_reg', score)

# log_reg2 = LogisticRegression(penalty='l2', tol=1e-6, C=0.1, multi_class='multinomial', solver='newton-cg')
# log_reg2.fit(X_train, y_train)
# predict_y = log_reg2.predict(X_test)
# score = log_reg2.score(X_test, y_test)
# print('log_reg2', score)

# SVM分类器
# {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'probability': True, 'random_state': 0}
svc = SVC(C=1.5, kernel='rbf', gamma='auto', probability=True, tol=1e-3, random_state=0)
svc.fit(X_train, y_train)
predict_y = svc.predict(X_test)
score = svc.score(X_test, y_test)
print('svc', score)

# 高斯贝叶斯分类器
gsd_nb = GaussianNB()
gsd_nb.fit(X_train, y_train)
predict_y = gsd_nb.predict(X_test)
score = gsd_nb.score(X_test, y_test)
print('gsd_nb', score)

# 随机森林分类器
# {'max_leaf_nodes': 30, 'n_estimators': 500, 'random_state': 1000}
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=30, random_state=2000)
rf_clf.fit(X_train, y_train)
predict_y = rf_clf.predict(X_test)
score = rf_clf.score(X_test, y_test)
print('rf_clf', score)

# 梯度提升树分类器
# {'max_leaf_nodes': 15, 'n_estimators': 100, 'random_state': 666}
gb_clf = GradientBoostingClassifier(n_estimators=100, max_leaf_nodes=15, random_state=400)
gb_clf.fit(X_train, y_train)
predict_y = gb_clf.predict(X_test)
score = gb_clf.score(X_test, y_test)
print('gb_clf', score)

# 决策树分类器
# {'max_leaf_nodes': 20, 'random_state': 1000}
dt_clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, random_state=0)
dt_clf.fit(X_train, y_train)
predict_y = dt_clf.predict(X_test)
score = dt_clf.score(X_test, y_test)
print('dt_clf', score)

# 集成学习
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression(penalty='l2', solver='newton-cg', tol=1e-4, C=10, multi_class='multinomial')),
    ('svm_clf', SVC(C=0.5, kernel='rbf', gamma=0.1, tol=1e-3, probability=True, random_state=0)),
    ('dt_clf', DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, random_state=1)),
    ('rf_clf', RandomForestClassifier(n_estimators=500, max_leaf_nodes=25, random_state=162)),
    ('gb_clf', GradientBoostingClassifier(n_estimators=100, max_leaf_nodes=16, random_state=0))
], voting='soft')
voting_clf.fit(X_train, y_train)
score = voting_clf.score(X_test, y_test)
print('voting_clf', score)