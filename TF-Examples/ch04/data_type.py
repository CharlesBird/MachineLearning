import tensorflow as tf

# 数值类型
# 创建张量

a = 1.2
aa = tf.constant(1.2)  # 创建标量
print(type(a), type(aa), tf.is_tensor(aa))

x = tf.constant([1,2.,3.3])
print(x)
print(x.numpy())

a = tf.constant([1.2])  # 创建向量
print(a, a.shape)

a = tf.constant([[1,2],[3,4]])  # 创建矩阵
print(a, a.shape)

a = tf.constant([[[1,2],[3,4]], [[5,6],[7,8]]])  # 3维张量
print(a, a.shape)


# 字符串类型
a = tf.constant("Hello, Deep Learning")
print(a)
b = tf.strings.lower(a)
print(b)


# 布尔型
a = tf.constant(True)  # 创建布尔型张量
print(a)


"""
对于大部分深度学习算法，一般使用 tf.int32, tf.float32 可满足运算精度要求，部分对
精度要求较高的算法，如强化学习，可以选择使用 tf.int64, tf.float64 精度保存张量。
"""

a = tf.constant(10)
print(a.dtype)
a = tf.cast(a, tf.float32)  # 转换精度
print(a.dtype)

a = tf.constant([-1, 0, 1, 2])
b = tf.cast(a, tf.bool)
print(b)