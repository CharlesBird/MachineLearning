# Himmelblau 函数优化实战
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Himmelblau 函数是用来测试优化算法的常用样例函数之一，它包含了两个自变量
𝑥, 𝑦，数学表达式: f(x, y) = (𝑥**2 + 𝑦 − 11)**2 + (𝑥 + 𝑦**2 − 7)**2
"""


def himmelblau(x):
    # himmelblau 函数实现
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# 通过np.meshgrid 函数(TensorFlow 中也有meshgrid 函数)生成二维平面网格点坐标
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y range: ', x.shape, y.shape)

# 生成x-y 平面采样网格点，方便可视化
X, Y = np.meshgrid(x, y)
print('X, Y maps: ', X.shape, Y.shape)

Z = himmelblau([X, Y])  # 计算网格点上的函数值

# 绘制himmelblau 函数曲面
fig = plt.figure('himmelblau')
# ax = Axes3D(fig)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# 参数的初始化值对优化的影响不容忽视，可以通过尝试不同的初始化值，
# 检验函数优化的极小值情况
# [1., 0.], [-4, 0.], [4, 0.], [-2., 2.]
x = tf.constant([-2., 2.])  # 初始化参数

for step in range(200):  # 循环优化
    with tf.GradientTape() as tape:  # 梯度跟踪
        tape.watch([x])  # 记录梯度
        y = himmelblau(x)  # 前向传播
    # 反向传播
    grads = tape.gradient(y, [x])[0]
    # 更新参数， 0.01为学习率
    x -= 0.01 * grads
    # 打印优化的极小值
    if step % 20 == 19:
        print('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))