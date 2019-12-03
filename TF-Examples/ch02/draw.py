import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt("data.csv", delimiter=",")

x = []
y = []
for i in range(0, len(points)):
    x.append(points[i, 0])
    y.append(points[i, 1])


plt.scatter(x, y)

plt.show()