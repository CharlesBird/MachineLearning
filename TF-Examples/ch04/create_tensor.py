# 创建张量
"""
通过 tf.convert_to_tensor 可以创建新 Tensor，并将保存在 Python List 对象或者 Numpy
Array 对象中的数据导入到新 Tensor 中
"""
import tensorflow as tf
import numpy as np
a = tf.convert_to_tensor([1,2.])
print(a)

b = tf.convert_to_tensor(np.array([[1,2.],[3,4]]), dtype=tf.float32)
print(b)
"""
tf.constant()和 tf.convert_to_tensor()都能够自动的把 Numpy 数组或者 Python
List 数据类型转化为 Tensor 类型，这两个 API 命名来自 TensorFlow 1.x 的命名习惯，在
TensorFlow 2 中函数的名字并不是很贴切，使用其一即可。
"""

# 创建全0或者全1张量
a = tf.zeros([2,2])
print(a)

b = tf.ones([3,2])
print(b)

# 通过 tf.zeros_like, tf.ones_like 可以方便地新建与某个张量 shape 一致，内容全 0 或全 1 的张量
b = tf.ones_like(a)
print(b)

"""
tf.*_like 是一个便捷函数，可以通过 tf.zeros(a.shape)等方式实现
"""


# 创建自定义张量
a = tf.fill([1], -1)
print(a)
b = tf.fill([3, 2], 99)
print(b)


# 创建已知分布的张量
a = tf.random.normal([2, 2])
print(a)

b = tf.random.normal([10, 3], mean=1, stddev=2)  # 创建均值为 1，标准差为 2 的正太分布
print(b)

a = tf.random.uniform([2, 2])  # 创建采样自[𝑚𝑖𝑛𝑣𝑎𝑙, 𝑚𝑎𝑥𝑣𝑎𝑙]区间的均匀分布的张量。
print(a)

b = tf.random.uniform([2, 2], maxval=100, dtype=tf.int32)  # 如果需要均匀采样整形类型的数据，必须指定采样区间的最大值 maxval 参数，同时制定数据类型为 tf.int*型
print(b)


# 创建序列
a = tf.range(10)
print(a)

b = tf.range(10, delta=2)  # 创建 0~9，步长为 2 的整形序列
print(b)

c = tf.range(1, 10, delta=2)  # 通过 tf.range(start, limit, delta=1)可以创建[𝑠𝑡𝑎𝑟𝑡, 𝑙𝑖𝑚𝑖𝑡)，步长为 delta 的序列，不包含 limit本身
print(c)