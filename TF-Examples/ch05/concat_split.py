# 合并与分割
import tensorflow as tf

# 合并
# 拼接方式，不改变维度
a = tf.random.normal([4,35,8])

b = tf.random.normal([6,35,8])

print(tf.concat([a, b], axis=0))  # 班级维度索引号为 0，即 axis=0，合并张量 A,B

a = tf.random.normal([10,35,3])
b = tf.random.normal([10,35,5])
print(tf.concat([a, b], axis=2))  # 在科目维度拼接

# 堆叠方式，改变维度
a = tf.random.normal([35, 8])

b = tf.random.normal([35, 8])

print(tf.stack([a, b], axis=0))  # 堆叠合并为 2 个班级

print(tf.stack([a, b], axis=-1))  # 在末尾插入班级维度


# 分割
"""
通过 tf.split(x, axis, num_or_size_splits)可以完成张量的分割操作
x：待分割张量
axis：分割的维度索引号
num_or_size_splits：切割方案。当 num_or_size_splits 为单个数值时，如 10，表示切割为 10 份；
当 num_or_size_splits 为 List 时，每个元素表示每份的长度，如[2,4,2,2]表示切割为 4 份，每份的长度分别为 2,4,2,2
"""
x = tf.random.normal([10,35,8])
result = tf.split(x, axis=0, num_or_size_splits=10)  # 切割为 10 份
print(result, len(result))
print(result[0].shape)

result = tf.split(x, axis=0, num_or_size_splits=[4, 2, 2, 2])  # 自定义长度的切割
print(result, len(result))
print(result[0].shape)

# 特别地，如果希望在某个维度上全部按长度为 1 的方式分割，还可以直接使用 tf.unstack(x, axis)。
# 这种方式是 tf.split 的一种特殊情况，切割长度固定为 1，只需要指定切割维度即可。
result = tf.unstack(x, axis=0) # Unstack 为长度为 1
print(result, len(result))
print(result[0].shape)