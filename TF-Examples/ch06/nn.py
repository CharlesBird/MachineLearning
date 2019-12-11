# 神经网络
import tensorflow as tf

# 感知机

# 全连接层
"""由于每个输出节点与全部的输入节点相连接，这种网络层称为全连接层(Fully-connected Layer)，或者稠密连接层
(Dense Layer)，W 矩阵叫做全连接层的权值矩阵，𝒃向量叫做全连接层的偏置。
"""
# 张量方式实现
"""
在 TensorFlow 中，要实现全连接层，只需要定义好权值张量 W 和偏置张量 b，并利用
TensorFlow 提供的批量矩阵相乘函数 tf.matmul()即可完成网络层的计算。如下代码创建输
入 X 矩阵为𝑏 = 2个样本，每个样本的输入特征长度为𝑑𝑖𝑛 = 784，输出节点数为𝑑𝑜𝑢𝑡 =
256，故定义权值矩阵 W 的 shape 为[784,256]，并采用正态分布初始化 W；偏置向量 b 的
shape 定义为[256]，在计算完X@W后相加即可，最终全连接层的输出 O 的 shape 为
[2,256]，即 2 个样本的特征，每个特征长度为 256。
"""
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1  # 线性变换
print(o1)
o1 = tf.nn.relu(o1)  # 激活函数
print(o1)

# 层方式实现
"""
全连接层本质上是矩阵的相乘相加运算，实现并不复杂。但是作为最常用的网络层之一，
TensorFlow 中有更加高层、使用更方便的层实现方式：layers.Dense(units, activation)，
只需要指定输出节点数 Units 和激活函数类型即可。输入节点数将根据第一次运算时的输入 shape 确定，
同时根据输入、输出节点数自动创建并初始化权值矩阵 W 和偏置向量 b，
使用非常方便。其中 activation 参数指定当前层的激活函数，可以为常见的激活函数或自定义激活函数，也可以指定为 None 无激活函数。
"""
x = tf.random.normal([4, 784])
fc = tf.keras.layers.Dense(512, activation=tf.nn.relu)  # 创建全连接层，指定输出节点数和激活函数
h1 = fc(x)  # 通过 fc 类完成一次全连接层的计算
print(h1)
# 通过类内部的成员名kernel 和 bias 来获取权值矩阵 W 和偏置 b
print(fc.kernel, fc.bias)  # 获取 Dense 类的权值矩阵, 获取 Dense 类的偏置向量
# 在优化参数时，需要获得网络的所有待优化的参数张量列表，可以通过类的trainable_variables 来返回待优化参数列表
print(fc.trainable_variables)
# 如果希望获得所有参数列表，可以通过类的 variables 返回所有内部张量列表
print(fc.variables)  # 返回所有参数列表
"""
利用网络层类对象进行前向计算时，只需要调用类的__call__方法即可，即写成 fc(x)
方式，它会自动调用类的__call__方法，在__call__方法中自动调用 call 方法，全连接层类
在 call 方法中实现了𝜎(𝑋@𝑊 + 𝒃)的运算逻辑，最后返回全连接层的输出张量。
"""


# 神经网络
"""
通过层层堆叠全连接层，保证前一层的输出节点数与当前层的输入节点数
匹配，即可堆叠出任意层数的网络。我们把这种由神经元构成的网络叫做神经网络。
"""
# 张量方式实现
# 隐藏层 1 张量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层 2 张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层 3 张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
with tf.GradientTape() as tape:  # 梯度记录器
    # x: [b, 28*28]
    # 隐藏层 1 前向计算，[b, 28*28] => [b, 256]
    h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # 隐藏层 2 前向计算，[b, 256] => [b, 128]
    h2 = h1 @ w2 + b2
    h2 = tf.nn.relu(h2)
    # 隐藏层 3 前向计算，[b, 128] => [b, 64]
    h3 = h2 @ w3 + b3
    h3 = tf.nn.relu(h3)
    # 输出层前向计算，[b, 64] => [b, 10]
    h4 = h3 @ w4 + b4
"""
在使用 TensorFlow 自动求导功能计算梯度时，需要将前向计算过程放置在
tf.GradientTape()环境中，从而利用 GradientTape 对象的 gradient()方法自动求解参数的梯
度，并利用 optimizers 对象更新参数。
"""

# 层方式实现
fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)  # 隐藏层 1
fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)  # 隐藏层 2
fc3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)  # 隐藏层 3
fc4 = tf.keras.layers.Dense(10, activation=None)  # 输出层
x = tf.random.normal([4, 28*28])
h1 = fc1(x)  # 通过隐藏层 1 得到输出
h2 = fc2(h1)  # 通过隐藏层 2 得到输出
h3 = fc3(h2)  # 通过隐藏层 3 得到输出
h4 = fc4(h3)  # 通过输出层得到网络输出
print(h4)

"""
对于这种数据依次向前传播的网络，也可以通过 Sequential 容器封装成一个网络大类对
象，调用大类的前向计算函数即可完成所有层的前向计算
"""
# 通过 Sequential 容器封装为一个网络类
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])
out = model(x)  # 前向计算得到输出
print(out)


# 优化目标
"""
我们把神经网络从输入到输出的计算过程叫做前向传播(Forward propagation)。神经网
络的前向传播过程，也是数据张量(Tensor)从第一层流动(Flow)至输出层的过程：从输入数
据开始，途径每个隐藏层，直至得到输出并计算误差，这也是 TensorFlow 框架名字意义所在。
"""

"""
利用误差反向传播算法进行反向计算的过程也叫反向传播(Backward propagation)。
"""
"""
原始的特征通常具有较高的维度，包含了很多底层特征及无用信息，通过神
经网络的层层特征变换，将较高的维度降维到较低的维度，此时的特征一般包含了与任务
强相关的高层特征信息，通过对这些特征进行简单的逻辑判定即可完成特定的任务，如图片的分类。
"""
"""
网络的参数量是衡量网络规模的重要指标。那么怎么计算全连接层的参数量呢？
考虑权值矩阵W，偏置 b，输入特征长度为𝑑𝑖𝑛，输出特征长度为𝑑𝑜𝑢𝑡的网络层，其参数量为
𝑑𝑖𝑛 ∗ 𝑑𝑜𝑢𝑡，再加上偏置 b 的参数，总参数量为𝑑𝑖𝑛 ∗ 𝑑𝑜𝑢𝑡 + 𝑑𝑜𝑢𝑡。对于多层的全连接神经网
络，比如784 → 256 → 128 → 64 → 10，总参数量计算表达式：
256 ∗ 784 + 256 + 128 ∗ 256 + 128 + 64 ∗ 128 + 64 + 10 ∗ 64 + 10 = 242762约 242K 个参数量。
"""


# 激活函数
# 1.Sigmoid
"""
1.概率分布 [0,1]区间的输出和概率的分布范围契合，可以通过 Sigmoid 函数将输出转译为概率输出
2.信号强度 一般可以将 0~1 理解为某种信号的强度，如像素的颜色强度，1 代表当前通道颜色最强，0 代表当前通道无颜色；
抑或代表门控值(Gate)的强度，1 代表当前门控全部开放，0 代表门控关闭
"""
# 通过 tf.nn.sigmoid 实现 Sigmoid 函数：
x = tf.linspace(-6., 6., 10)
print(x)
print(tf.nn.sigmoid(x))  # 通过 Sigmoid 函数

# 2.ReLU
"""
在 ReLU(REctified Linear Unit，修正线性单元)激活函数提出之前，Sigmoid 函数通常
是神经网络的激活函数首选。但是 Sigmoid 函数在输入值较大或较小时容易出现梯度值接
近于 0 的现象，称为梯度弥散现象，网络参数长时间得不到更新，很难训练较深层次的网络模型。
"""
print(tf.nn.relu(x))  # 通过 ReLU 激活函数

# 3.LeakyReLU
"""
ReLU 函数在𝑥 < 0时梯度值恒为 0，也可能会造成梯度弥散现象，为了克服这个问
题，LeakyReLU 函数被提出，如图 6.11 所示，LeakyReLU 表达式为:
𝐿𝑒𝑎𝑘𝑦𝑅𝑒𝐿𝑈 = {
𝑥 𝑥 ≥ 0
𝑝 ∗ 𝑥 𝑥 < 0
"""
print(tf.nn.leaky_relu(x, alpha=0.1))  # 通过 LeakyReLU 激活函数

# 4.Tanh
"""
Tanh 函数能够将𝑥 ∈ 𝑅的输入“压缩”到[−1,1]区间
"""
print(tf.nn.tanh(x))  # 通过 tanh 激活函数


# 输出层设计
# 普通实数空间
"""
这一类问题比较普遍，像正弦函数曲线预测、年龄的预测、股票走势的预测等都属于
整个或者部分连续的实数空间，输出层可以不加激活函数。误差的计算直接基于最后一层
的输出𝒐和真实值 y 进行计算，如采用均方差误差函数度量输出值𝒐与真实值𝒚之间的距
离：ℒ = 𝑔(𝒐,𝒚)
其中𝑔代表了某个具体的误差计算函数。
"""

# [0, 1]区间
"""
输出值属于[0,1]区间也比较常见，比如图片的生成，二分类问题等。在机器学习中，
一般会将图片的像素值归一化到[0,1]区间，如果直接使用输出层的值，像素的值范围会分
布在整个实数空间。为了让像素的值范围映射到[0,1]的有效实数空间，需要在输出层后添
加某个合适的激活函数𝜎，其中 Sigmoid 函数刚好具有此功能。
"""

# [0,1]区间，和为 1
"""
输出值𝑜𝑖 ∈ [0,1]，所有输出值之和为 1，这种设定以多分类问题最为常见。
可以通过在输出层添加 Softmax 函数实现。Softmax 函数不仅可以将输出值映射到[0,1]区间，还满足所有的输出值之和为 1 的特性。
"""
z = tf.constant([2., 1., 0.1])
print(tf.nn.softmax(z))  # 通过 tf.nn.softmax
"""
通过类 layers.Softmax(axis=-1)可以方便添加 Softmax 层，其中 axis 参数指定需要进行计算的维度
"""
"""
在 Softmax 函数的数值计算过程中，容易因输入值偏大发生数值溢出现象；在计算交叉熵时，也会出现数值溢出的问题。
为了数值计算的稳定性，TensorFlow 中提供了一个统一的接口，将 Softmax 与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常，
一般推荐使用，避免单独使用 Softmax 函数与交叉熵损失函数。
函数式接口为tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)，其中 y_true 代表了
one-hot 编码后的真实标签，y_pred 表示网络的预测值，当 from_logits 设置为 True 时，y_pred 表示须为未经过 Softmax 函数的变量 z；
当 from_logits 设置为 False 时，y_pred 表示为经过 Softmax 函数的输出。
"""
z = tf.random.normal([2, 10])  # 构造输出层的输出
y_onehot = tf.constant([1, 3])  # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)  # one-hot 编码
# 输出层未使用 Softmax 函数，故 from_logits 设置为 True
loss = tf.keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)  # 计算平均交叉熵损失
print(loss)
"""
利用 losses.CategoricalCrossentropy(from_logits)类方式同时实现 Softmax 与交叉熵损失函数的计算
"""
criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot, z)  # 计算损失
print(loss)

#  [-1, 1]
"""如果希望输出值的范围分布在[−1, 1]，可以简单地使用 tanh 激活函数"""
x = tf.linspace(-6., 6., 10)
print(tf.tanh(x))  # tanh 激活函数


# 误差计算
"""
常见的误差计算函数有均方差、交叉熵、KL 散度、Hinge Loss 函数等，其中均方差函
数和交叉熵函数在深度学习中比较常见，均方差主要用于回归问题，交叉熵主要用于分类问题。
"""
# 均方差
# 均方差误差(Mean Squared Error, MSE)函数把输出向量和真实向量映射到笛卡尔坐标系的两个点上，
# 通过计算这两个点之间的欧式距离(准确地说是欧式距离的平方)来衡量两个向量之间的差距
# MSE 误差函数的值总是大于等于 0，当 MSE 函数达到最小值 0 时，输出等于真实标签，此时神经网络的参数达到最优状态。
o = tf.random.normal([2, 10])  # 构造网络输出
y_onehot = tf.constant([1, 3])  # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = tf.keras.losses.MSE(y_onehot, o)  # 计算均方差
print(loss)
# TensorFlow MSE 函数返回的是每个样本的均方差，需要在样本数量上再次平均来获得batch 的均方差
loss = tf.reduce_mean(loss)  # 计算 batch 均方差
print(loss)
# 也可以通过层方式实现，对应的类为 keras.losses.MeanSquaredError()
criteon = tf.keras.losses.MeanSquaredError()
loss = criteon(y_onehot, o)  # 计算 batch 均方差
print(loss)

# 交叉熵
"""
熵在信息学科中也叫信息熵，或者香农熵。熵越大，代表不确定性越大，信息量也就越大。
交叉熵可以分解为 p 的熵𝐻(𝑝)与 p,q 的 KL 散度(Kullback-Leibler Divergence)的和
"""
