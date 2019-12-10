# 数据限幅
import tensorflow as tf

x = tf.range(9)
print(x)
print(tf.maximum(x, 3))  # 下限幅

x = tf.range(9)
print(tf.minimum(x, 7))  # 上限幅


# ReLU 函数可以实现为
def relu(x):
    return tf.minimum(x, 0.)  # 下限幅为 0 即可


a = tf.random.normal((10, ))
print(a)
print(relu(a))


# 通过组合 tf.maximum(x, a)和 tf.minimum(x, b)可以实现同时对数据的上下边界限幅：𝑥 ∈ [𝑎, 𝑏]
x = tf.range(9)
print(tf.minimum(tf.maximum(x, 2), 7))  # 限幅为 2~7

# 可以使用 tf.clip_by_value 实现上下限幅
x = tf.range(9)
print(tf.clip_by_value(x, 2, 8))  # 限幅为 2~8