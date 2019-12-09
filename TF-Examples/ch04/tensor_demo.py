import tensorflow as tf

"""
以均方差误差函数为例，经过 tf.keras.losses.mse(或 tf.keras.losses.MSE)返回每个样本
上的误差值，最后取误差的均值作为当前 batch 的误差，它是一个标量
"""
out = tf.random.uniform([4, 10])  # 随机模拟网络输出
y = tf.constant([2, 3, 2, 0])  # 随机构造样本真实标签
y = tf.one_hot(y, depth=10)  # one-hot 编码
loss = tf.keras.losses.mse(y, out)  # 计算每个样本的 MSE
loss = tf.reduce_mean(loss)  # 平均 MSE
print(loss)


"""
向量是一种非常常见的数据载体，如在全连接层和卷积神经网络层中，偏置张量𝒃就
使用向量来表示。
"""
# z=wx,模拟获得激活函数的输入 z
z = tf.random.normal([4, 2])
b = tf.zeros([2])  # 模拟偏置向量
z = z + b  # 累加偏置

fc = tf.keras.layers.Dense(3)  # 创建一层 Wx+b，输出节点为 3
fc.build(input_shape=(2, 4))  # 通过 build 函数创建 W,b 张量，输入节点为 4
print(fc.bias)  # 查看偏置


"""
矩阵也是非常常见的张量类型，比如全连接层的批量输入𝑋 = [𝑏, 𝑑𝑖𝑛]，其中𝑏表示输入
样本的个数，即 batch size，𝑑𝑖𝑛表示输入特征的长度。
"""
x = tf.random.normal([2, 4])
w = tf.ones([4, 3])  # 定义 W 张量
b = tf.zeros([3])  # 定义 b 张量
o = x@w+b  # X@W+b 运算
fc = tf.keras.layers.Dense(3)  # 定义全连接层的输出节点为 3
fc.build(input_shape=(2, 4))  # 定义全连接层的输入节点为 4
print(fc.kernel)  # 通过全连接层的 kernel 成员名查看其权值矩阵 W


"""
三维的张量一个典型应用是表示序列信号，它的格式是
𝑋 = [𝑏, 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑙𝑒𝑛, 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 𝑙𝑒𝑛]
其中𝑏表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数，feature
len 表示每个点的特征长度。
"""
# 自动加载 IMDB 电影评价数据集
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)  # 将句子填充、截断为等长 80 个单词的句子
print(x_train.shape)
# 到 x_train 张量的 shape 为[25000,80]，其中 25000 表示句子个数，80 表示每个句子共 80 个单词，
# 每个单词使用数字编码方式。我们通过 layers.Embedding 层将数字编码的单词转换为长度为 100 个词向量
embedding = tf.keras.layers.Embedding(10000, 100)  # 创建词向量 Embedding 层类
out = embedding(x_train)  # 将数字编码的单词转换为词向量
# 经过 Embedding 层编码后，句子张量的 shape 变为[25000,80,100]，其中 100 表示每个单词编码为长度 100 的向量。
print(out.shape)


"""
4 维张量在卷积神经网络中应用的非常广泛，它用于保存特征图(Feature maps)数据，
格式一般定义为 [𝑏, ℎ, , 𝑐]
其中𝑏表示输入的数量，h/w分布表示特征图的高宽，𝑐表示特征图的通道数，部分深度学
习框架也会使用[𝑏, 𝑐, ℎ, ]格式的特征图张量，例如 PyTorch。图片数据是特征图的一种，
对于含有 RGB 3 个通道的彩色图片，每张图片包含了 h 行 w 列像素点，每个点需要 3 个数
值表示 RGB 通道的颜色强度，因此一张图片可以表示为[h,w, 3]
"""
x = tf.random.normal([4, 32, 32, 3])  # 创建 32x32 的彩色图片输入，个数为 4
layer = tf.keras.layers.Conv2D(16, kernel_size=3)  # 创建卷积神经网络
out = layer(x)  # 前向计算
print(out.shape)  # 输出大小
# 卷积核张量也是 4 维张量，可以通过 kernel 成员变量访问
print(layer.kernel.shape)