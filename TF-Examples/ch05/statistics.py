# 数据统计
import tensorflow as tf
import numpy as np

# 向量范数
x = tf.ones([2, 2])
print(tf.norm(x, ord=1))  # L1 范数，定义为向量𝒙的所有元素绝对值之和

print(tf.norm(x, ord=2))  # L2 范数，定义为向量𝒙的所有元素的平方和，再开根号

print(tf.norm(x, ord=np.inf))  # ∞ −范数，定义为向量𝒙的所有元素绝对值的最大值


# 最大最小值、均值、和
x = tf.random.uniform([4, 10])
print(x)
print(tf.reduce_max(x, axis=1))  # 统计概率维度上的最大值
print(tf.reduce_max(x, axis=0))

print(tf.reduce_min(x, axis=1))  # 统计概率维度上的最小值

print(tf.reduce_mean(x, axis=1))  # 统计概率维度上的均值

# 当不指定 axis 参数时，tf.reduce_*函数会求解出全局元素的最大、最小、均值、和：
print(tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x))

# 通过 TensorFlow 的 MSE 误差函数可以求得每个样本的误差，需要计算样本的平均误差，
# 此时可以通过 tf.reduce_mean 在样本数维度上计算均值
out = tf.random.normal([4, 10])  # 网络预测输出
y = tf.constant([1, 2, 2, 0])  # 真实标签
y = tf.one_hot(y, depth=10)  # one-hot 编码
loss = tf.keras.losses.mse(y, out)  # 计算每个样本的误差
loss = tf.reduce_mean(loss)  # 平均误差
print(loss)

print(tf.reduce_sum(out, axis=-1))  # 求和
print(tf.reduce_sum(out, axis=1))  # 求和

out = tf.random.normal([2, 10])
print(out)
out = tf.nn.softmax(out, axis=1)  # 通过 softmax 转换为概率值
print(out)
# 通过 tf.argmax(x, axis)，tf.argmin(x, axis)可以求解在 axis 轴上，x 的最大值、最小值所在的索引号
pred = tf.argmax(out, axis=1)  # 选取概率最大的位置
print(pred)
