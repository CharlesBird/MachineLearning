import pandas as pd
import os
import matplotlib.pyplot as plt
data_train = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/train.csv')
data_test = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/test.csv')

label = data_train.label
data_train = data_train.drop('label', axis=1)

for x in range(0, 4):
    train_0 = data_train[label == x]
    data_new = []
    for idx in train_0.index:
        val = train_0.loc[idx].values.reshape(28, 28)
        data_new.append(val)
    plt.figure(figsize=(25, 25))
    for x in range(1, 5):
        ax1 = plt.subplot(1, 20, x)
        ax1.imshow(data_new[x], cmap='gray')




y = pd.value_counts(data_train.values.ravel()).sort_index()
N = len(y)
x = range(N)
width =0.9
plt.figure(figsize=[20, 8])
ax1=plt.subplot(1, 2, 1)
ax1.bar(x, y, width, color="blue")
plt.title('Pixel Value Frequency For Gray Scale Images')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')
#ax1.imshow(data_new[x],cmap='gray')
ax2 = plt.subplot(1, 2, 2)
ax2.bar(x, y, width, color="blue")
plt.title('Pixel Value Frequency (Log Scale)')
plt.yscale('log')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')

plt.show()