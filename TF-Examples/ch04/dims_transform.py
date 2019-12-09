import tensorflow as tf

# Reshape
x = tf.range(96)
x = tf.reshape(x, [2, 4, 4, 3])
print(x)

print(x.ndim, x.shape)  # 获得张量的维度数和形状

print(tf.reshape(x, [2, -1]))  # 通过 tf.reshape(x, new_shape)，可以将张量的视图任意的合法改变

print(tf.reshape(x, [2, 4, 12]))

print(tf.reshape(x, [2, -1, 3]))


# 增删维度
x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)
print(x)

a = tf.expand_dims(x, axis=2)  # 通过 tf.expand_dims(x, axis)可在指定的 axis 轴前可以插入一个新的维度
print(a, a.shape)

b = tf.expand_dims(a, axis=0)  # 在最前面插入一个新的维度，并命名为图片数量维度，长度为1，此时张量的 shape 变为[1,28,28,1]
print(b, b.shape)

c = tf.expand_dims(x, axis=1)
print(c, c.shape)

# 删除维度
"""
删除维度只能删除长度为 1 的维度，也不会改变张量的存储
"""
d = tf.squeeze(b, axis=0)  # 通过 tf.squeeze(x, axis)函数，axis 参数为待删除的维度的索引号
print(d, d.shape)

e = tf.squeeze(d, axis=2)  # 通过 tf.squeeze(x, axis)函数，axis 参数为待删除的维度的索引号
print(e, e.shape)

x = tf.random.uniform([1, 28, 28, 1], maxval=10, dtype=tf.int32)
print(tf.squeeze(x))  # 如果不指定维度参数 axis，即 tf.squeeze(x)，那么他会默认删除所有长度为 1 的维度


# 交换维度
"""
改变视图、增删维度都不会影响张量的存储。
在 TensorFlow 中，图片张量的默认存储格式是通道后行格式：[𝑏, ℎ, w, 𝑐]，但是部分库的图片格式是通道先行：[𝑏, 𝑐, ℎ, w]，
因此需要完成[𝑏, ℎ, w, 𝑐]到[𝑏, 𝑐, ℎ, w]维度交换运算，
使用 tf.transpose(x, perm)函数完成维度交换操作，其中 perm 表示新维度的顺序 List。考虑图
片张量 shape 为[2,32,32,3]，图片数量、行、列、通道数的维度索引分别为 0,1,2,3，如果需
要交换为[𝑏, 𝑐, ℎ, ]格式，则新维度的排序为图片数量、通道数、行、列，对应的索引号为
[0,3,1,2]
"""
x = tf.random.normal([2, 32, 32, 3])
print(x)
a = tf.transpose(x, perm=[0, 3, 1, 2])  # [𝑏, ℎ, w, 𝑐] => [𝑏, 𝑐, ℎ, w]
print(a, a.shape)
b = tf.transpose(x, perm=[0, 2, 1, 3])  # [𝑏, ℎ, w, 𝑐] => [𝑏, w, ℎ, c]
print(b, b.shape)


# 数据复制
b = tf.range(3)
b = tf.expand_dims(b, axis=0)
print(b)
# multiples 分别指定了每个维度上面的复制倍数，对应位置为 1 表明不复制，为 2 表明新长度为原来的长度的 2 倍，即数据复制一份，以此类推
print(tf.tile(b, multiples=[2, 1]))

x = tf.range(4)
x = tf.reshape(x, [2, 2])
print(x)
x = tf.tile(x, multiples=[1, 2])  # 首先在列维度复制 1 份数据
print(x)
x = tf.tile(x, multiples=[2, 1])  # 然后在行维度复制 1 份数据
print(x)
"""
注意: tf.tile 会创建一个新的张量来保存复制后的张量，由于复制操作涉及到
大量数据的读写 IO 运算，计算代价相对较高。神经网络中不同 shape 之间的运算操作十分
频繁，那么有没有轻量级的复制操作呢？这就是接下来要介绍的 Broadcasting 操作
"""


# Broadcasting
"""
Broadcasting 也叫广播机制(自动扩展也许更合适)，它是一种轻量级张量复制的手段，
在逻辑上扩展张量数据的形状，但是只要在需要时才会执行实际存储复制操作。对于大部
分场景，Broadcasting 机制都能通过优化手段避免实际复制数据而完成逻辑运算，从而相对
于 tf.tile 函数，减少了大量计算代价。
"""

"""
x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.random.normal([3])
y = x@w+b
上述加法并没有发生逻辑错误，那么它是怎么实现的呢？这是因为它自动调用 Broadcasting
函数 tf.broadcast_to(x, new_shape)，将 2 者 shape 扩张为相同的[2,3]，即上式可以等效为：
y = x@w + tf.broadcast_to(b,[2,3])
也就是说，操作符+在遇到 shape 不一致的 2 个张量时，会自动考虑将 2 个张量
Broadcasting 到一致的 shape，然后再调用 tf.add 完成张量相加运算，
"""
A = tf.random.normal([32, 1])
print(A)
b = tf.broadcast_to(A, [2, 32, 32, 3])  # 通过 tf.broadcast_to(x, new_shape)可以显式将现有 shape 扩张为 new_shape
print(b)

A = tf.random.normal([32, 2])
tf.broadcast_to(A, [2,32,32,4])  # 不满足普适性原则，报错