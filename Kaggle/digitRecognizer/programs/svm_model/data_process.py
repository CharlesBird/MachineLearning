import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn import svm
data_train = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/test.csv')

label = data_train.label
data_train = data_train.drop('label', axis=1)
print('Data Shape: ', data_train.shape)
print("Label Shape: ", label.shape)

train_X, test_X, train_y, test_y = train_test_split(data_train, label, test_size=0.2, random_state=42)
print("Train Data Shape: ", train_X.shape)
print("Train Label Shape: ", train_y.shape)
print("Test Data Shape: ", test_X.shape)
print("Test Label Shape: ", test_y.shape)

i = 5000
score = []
fittime = []
scoretime = []
clf = svm.SVC(gamma='auto', random_state=42)
print("Default Parameters are: \n", clf.get_params)

start_time = time.time()
clf.fit(train_X[:i], train_y[:i].values.ravel())
fittime = time.time() - start_time
print("Time consumed to fit model: ",time.strftime("%H:%M:%S", time.gmtime(fittime)))
start_time = time.time()
score = clf.score(test_X, test_y)
print("Accuracy for grayscale: ", score)
scoretime = time.time() - start_time
print("Time consumed to score: ", time.strftime("%H:%M:%S", time.gmtime(scoretime)))
case1 = [score, fittime, scoretime]

