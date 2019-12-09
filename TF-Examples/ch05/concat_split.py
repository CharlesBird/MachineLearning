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