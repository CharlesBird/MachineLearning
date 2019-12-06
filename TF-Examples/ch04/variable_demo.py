# 待优化张量
"""
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，TensorFlow 增加了
一种专门的数据类型来支持梯度信息的记录：tf.Variable。
"""
import tensorflow as tf

a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
print(aa.name, aa.trainable)


aa = tf.Variable(a, name='aa')
print(aa.name, aa.trainable)

b = tf.constant([-1, 0, 1, 2])
bb = tf.Variable(b)
print(bb.name, bb.trainable)


a = tf.Variable([-1, 0, 1, 2])
print(a.name, a.trainable)