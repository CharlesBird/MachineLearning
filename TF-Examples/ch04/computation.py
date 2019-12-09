import tensorflow as tf

# 数学运算
# 加减乘除
a = tf.range(5)
b = tf.constant(2)
print(a // b)
print(a / b)
print(a % b)
print(a * b)
print(a + b)
print(a - b)

# 乘方
x = tf.range(4)
print(tf.pow(x, 3))
print(x ** 2)

# 对于常见的平方和平方根运算，可以使用 tf.square(x)和 tf.sqrt(x)实现
x = tf.range(5)
x = tf.cast(x, dtype=tf.float32)
x = tf.square(x)
print(x)
print(tf.sqrt(x))

# 指数 对数
# 通过 tf.pow(a, x)或者**运算符可以方便实现指数运算
print(tf.exp(1.))  # 特别地，对于自然指数𝑒𝑥,可以通过 tf.exp(x)实现

x = tf.exp(3.)
print(tf.math.log(x))  # 自然对数log𝑒 𝑥可以通过 tf.math.log(x)实现

x = tf.constant([1.,2.])
x = 10**x
print(tf.math.log(x)/tf.math.log(10.))  # 计算其他底数的对数


# 矩阵相乘
"""
TensorFlow 会选择 a,b 的最后两个维度进行矩阵相乘，前面所有的维度都视作 Batch 维度。
a 和 b 能够矩阵相乘的条件是，a 的倒数第一个维度长度(列)和b 的倒数第二个维度长度(行)必须相等。
"""
a = tf.random.normal([4,3,23,32])
b = tf.random.normal([4,3,32,2])
print(a @ b)
print(tf.matmul(a, b))

# 矩阵相乘函数支持自动 Broadcasting 机制
a = tf.random.normal([4,28,32])
b = tf.random.normal([32,16])
print(tf.matmul(a, b))