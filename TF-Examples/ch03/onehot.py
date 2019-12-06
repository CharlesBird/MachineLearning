import tensorflow as tf

y = tf.constant([1,2,3,7,9])
y_one_hot = tf.one_hot(y, depth=10)
print(y_one_hot)