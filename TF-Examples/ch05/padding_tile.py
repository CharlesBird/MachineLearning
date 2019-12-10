# 张量填充与复制
import tensorflow as tf

# 填充
"""
通过 tf.pad(x, paddings)函数实现，paddings 是包含了多个[𝐿𝑒𝑓𝑡 𝑃𝑎𝑑𝑑𝑖𝑛𝑔, 𝑅𝑖𝑔ℎ𝑡 𝑃𝑎𝑑𝑑𝑖𝑛𝑔]的嵌套方案 List，
如[[0,0],[2,1],[1,2]]表示第一个维度不填充，第二个维度左边(起始处)填充两个单元，右边(结束处)填充一个单元，
第三个维度左边填充一个单元，右边填充两个单元
"""
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([7, 8, 1, 6])
b = tf.pad(b, [[0, 2]])  # 填充
print(b)

# 填充后句子张量形状一致，再将这 2 句子 Stack 在一起
print(tf.stack([a, b], axis=0))

total_words = 10000  # 设定词汇量大小
max_review_len = 80  # 最大句子长度
embedding_len = 100  # 词向量长度
# 加载 IMDB 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating='post', padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating='post', padding='post')
print(x_train.shape, x_test.shape)


"""
考虑对图片的高宽维度进行填充。以 28x28 大小的图片数据为例，如果网络层所接受的数据高宽为 32x32，
则必须将 28x28 大小填充到32x32，可以在上、下、左、右方向各填充 2 个单元
"""
x = tf.random.normal([4, 28, 28, 1])
# 图片上下、左右各填充 2 个单元
print(tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]]))


# 复制
"""
通过 tf.tile 函数可以在任意维度将数据重复复制多份，如 shape 为[4,32,32,3]的数据，
复制方案 multiples=[2,3,3,1]，即通道数据不复制，高宽方向分别复制 2 份，图片数再复制
1 份
"""
x = tf.random.normal([4, 32, 32, 3])
print(tf.tile(x, [2, 3, 3, 1]))  # 数据复制