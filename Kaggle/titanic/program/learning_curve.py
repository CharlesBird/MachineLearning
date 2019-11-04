from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data_preprocess import data_process
data_train, data_test = data_process()
train_df = data_train.filter(regex='Survived|Name_lenth_.*|Age_.*|SibSp_.*|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
y = train_np[:, 0]
X = train_np[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# 学习曲线
def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(5, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test[:i])
        test_score.append(mean_squared_error(y_test[:i], y_test_predict))

    plt.plot([i for i in range(5, len(X_train) + 1)], np.sqrt(train_score), label="train")
    plt.plot([i for i in range(5, len(X_train) + 1)], np.sqrt(test_score), label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 1])
    plt.show()

plot_learning_curve(SVC(C=10, kernel='rbf', gamma=0.1, probability=True), X_train, X_test, y_train, y_test)