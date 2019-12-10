# 张量比较
import tensorflow as tf

out = tf.random.normal([100, 10])

out = tf.nn.softmax(out, axis=1)  # 输出转换为概率

pred = tf.argmax(out, axis=1)  # 选取预测值
print(pred)

y = tf.random.uniform([100], dtype=tf.int64, maxval=10)  #模拟真实标签
print(y)

# 通过 tf.equal(a, b)(或 tf.math.equal(a, b))函数可以比较这 2 个张量是否相等
out = tf.equal(pred, y)
print(out)

out = tf.cast(out, dtype=tf.float32)  # 布尔型转 int 型

correct = tf.reduce_sum(out)  # 统计 True 的个数
print(correct)

"""
函数 功能
tf.math.greater 𝑎 > 𝑏
tf.math.less 𝑎 < 𝑏
tf.math.greater_equal 𝑎 ≥ 𝑏
tf.math.less_equal 𝑎 ≤ 𝑏
tf.math.not_equal 𝑎 ≠ 𝑏
tf.math.is_nan 𝑎 = 𝑛𝑎n
"""