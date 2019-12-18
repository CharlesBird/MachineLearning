# Keras 高层接口
import tensorflow as tf

# 常见功能模块
"""
Keras 提供了一系列高层的神经网络类和函数，如常见数据集加载函数，网络层类，模型容器，损失函数类，优化器类，经典模型类等等。
对于常见数据集，通过一行代码即可下载、管理、加载功能函数，这些数据集包括Boston 房价预测数据集，CIFAR 图片数据集，
MNIST/FashionMNIST 手写数字图片数据集，IMDB 文本数据集等。
"""
# 1.常见网络层类
"""
对于常见的神经网络层，可以使用张量方式的底层接口函数来实现，这些接口函数一般在tf.nn 模块中。
更常用地，对于常见的网络层，我们一般直接使用层方式来完成模型的搭建，
在tf.keras.layers 命名空间(下文使用layers 指代tf.keras.layers)中提供了大量常见网络层的类接口，
如全连接层，激活含水层，池化层，卷积层，循环神经网络层等等。对于这些网络层类，只需要在创建时指定网络层的相关参数，
并调用__call__方法即可完成前向计算。在调用__call__方法时，Keras 会自动调用每个层的前向传播逻辑，这些逻辑一般实现在类的call 函数中。
"""
x = tf.constant([2., 1., 0.1])
layer = tf.keras.layers.Softmax(axis=-1)  # 创建Softmax 层
print(layer(x))  # 调用softmax 前向计算


def pro_process(x, y):
    x = tf.reshape(x, [-1])
    return x, y


# x: [60k, 28, 28],
# y: [60k]
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# x: [0~255] => [0~1.]
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(pro_process).batch(128)

val_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_db = val_db.shuffle(1000).map(pro_process).batch(128)

x, y = next(iter(train_db))
print(x.shape, y.shape)

# 2.网络容器
"""
对于常见的网络，需要手动调用每一层的类实例完成前向传播运算，当网络层数变得较深时，这一部分代码显得非常臃肿。
可以通过Keras 提供的网络容器Sequential 将多个网络层封装成一个大网络模型，只需要调用网络模型的实例一次即可完成数据从第一层到最末层的顺序运算。
"""
network = tf.keras.Sequential([  # 封装为一个网络
    tf.keras.layers.Dense(3, activation=None),  # 全连接层
    tf.keras.layers.ReLU(),  # 激活函数层
    tf.keras.layers.Dense(2, activation=None),  # 全连接层
    tf.keras.layers.ReLU()  # 激活函数层
])
x = tf.random.normal([4, 3])
print(network(x))  # 输入从第一层开始，逐层传播至最末层

# Sequential 容器也可以通过add()方法继续追加新的网络层，实现动态创建网络的功能
layers_num = 2  # 堆叠2 次
network = tf.keras.Sequential([])  # 先创建空的网络
for _ in range(layers_num):
    network.add(tf.keras.layers.Dense(3))  # 添加全连接层
    network.add(tf.keras.layers.ReLU())  # 添加激活函数层
network.build(input_shape=(None, 4)) # 创建网络参数
network.summary()
"""
上述代码通过指定任意的layers_num 参数即可创建对应层数的网络结构，在完成网络创建
时，很多类并没有创建内部权值张量等成员变量，此时通过调用类的build 方法并指定输
入大小，即可自动创建所有层的内部张量。通过summary()函数可以方便打印出网络结构和参数量:
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              multiple                  15        
_________________________________________________________________
re_lu_2 (ReLU)               multiple                  0         
_________________________________________________________________
dense_3 (Dense)              multiple                  12        
_________________________________________________________________
re_lu_3 (ReLU)               multiple                  0         
=================================================================
Total params: 27
Trainable params: 27
Non-trainable params: 0
_________________________________________________________________
"""
# 当我们通过Sequential 容量封装多层网络层时，所有层的参数列表将会自动并入Sequential 容器的参数列表中，不需要人为合并网络参数列表。
# Sequential 对象的trainable_variables 和variables 包含了所有层的待优化张量列表和全部张量列表
for p in network.trainable_variables:
    print(p.name, p.shape)


# 模型装配、训练与测试
"""
在训练网络时，一般的流程是通过前向计算获得网络的输出值，再通过损失函数计算网络误差，然后通过自动求导工具计算梯度并更新，同时间隔性地测试网络的性能。
对于这种常用的训练逻辑，可以直接通过Keras 提供的模型装配与训练高层接口实现，简洁清晰。
"""
# 1.模型装配
"""
在 Keras 中，有2 个比较特殊的类：keras.Model 和keras.layers.Layer 类。
其中Layer类是网络层的母类，定义了网络层的一些常见功能，如添加权值，管理权值列表等。
Model 类是网络的母类，除了具有Layer 类的功能，还添加了保存、加载模型，训练与测试模型等便捷功能。
"""
# 创建5 层的全连接层网络
network = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)])
network.build(input_shape=(4, 28*28))
network.summary()

# 采用Adam 优化器，学习率为0.01;采用交叉熵损失函数，包含Softmax
network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])  # 设置测量指标为准确率

# 2.模型训练
"""
模型装配完成后，即可通过fit()函数送入待训练的数据和验证用的数据集
"""
# 指定训练集为train_db，验证集为val_db,训练5 个epochs，每2 个epoch 验证一次返回训练信息保存在history 中
history = network.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2)
"""
其中train_db 为tf.data.Dataset 对象，也可以传入Numpy Array 类型的数据；epochs 指定训
练迭代的epochs 数；validation_data 指定用于验证(测试)的数据集和验证的频率validation_freq。
"""

print(history.history)  # 打印训练记录
"""
通过compile&fit 方式实现的代码非常简洁和高效，大大缩减了开发时间。但是因为接口非常高层，灵活性也降低了，是否使用需要用户自行判断。
"""

# 3.模型测试
"""
Model 基类除了可以便捷地完成网络的装配与训练、验证，还可以非常方便的预测和测试。
关于验证和测试的区别，我们会在过拟合章节详细阐述，此处可以将验证和测试理解为模型评估的一种方式。
"""
# 加载一个batch 的测试数据
x, y = next(iter(val_db))
print('predict x:', x.shape)
out = network.predict(x)  # 模型预测
print(out)

"""
如果只是简单的测试模型的性能，可以通过Model.evaluate(db)即可循环测试完db 数据集上所有样本，并打印出性能指标
"""
# network.evaluate(val_db)  # 模型测试


# 模型保存与加载
"""
模型训练完成后，需要将模型保存到文件系统上，从而方便后续的模型测试与部署工作。
实际上，在训练时间隔性地保存模型状态也是非常好的习惯，这一点对于训练大规模的网络尤其重要，一般大规模的网络需要训练数天乃至数周的时长，
一旦训练过程被中断或者发生宕机等意外，之前训练的进度将全部丢失。如果能够间断的保存模型状态到文件系统，即使发生宕机等意外，
也可以从最近一次的网络状态文件中恢复，从而避免浪费大量的训练时间。因此模型的保存与加载非常重要。
在 Keras 中，有三种常用的模型保存与加载方法。
"""
# 1.张量方式
"""
网络的状态主要体现在网络的结构以及网络层内部张量参数上，因此在拥有网络结构源文件的条件下，直接保存网络张量参数到文件上是最轻量级的一种方式。
"""
# 保存模型参数到文件上
# network.save_weights('weights.ckpt')
# print('saved weights.')
# del network  # 删除网络对象
# # 重新创建相同的网络结构
# network = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(10)])
# network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
#                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# # 从参数文件中读取数据并写入当前网络
# network.load_weights('weights.ckpt')
# print('loaded weights!')
"""
这种保存与加载网络的方式最为轻量级，文件中保存的仅仅是参数张量的数值，并没有其他额外的结构参数。
但是它需要使用相同的网络结构才能够恢复网络状态，因此一般在拥有网络源文件的情况下使用。
"""

# 2.网络方式
"""
我们来介绍一种不需要网络源文件，仅仅需要模型参数文件即可恢复出网络模型的方式。
通过Model.save(path)函数可以将模型的结构以及模型的参数保存到一个path 文件上，在不需要网络源文件的条件下，通过keras.models.load_model(path)即可恢复网络结构和网络参数。
"""
# 保存模型结构与模型参数到文件
# network.save('model.h5')
# print('saved total model.')
# del network  # 删除网络对象
# network = tf.keras.models.load_model('model.h5')  # 从文件恢复网络结构与网络参数

# 3.SavedModel 方式
"""
TensorFlow 之所以能够被业界青睐，除了优秀的神经网络层API 支持之外，还得益于它强大的生态系统，包括移动端和网页端的支持。
当需要将模型部署到其他平台时，采用TensorFlow 提出的SavedModel 方式更具有平台无关性。
"""
# 保存模型结构与模型参数到文件
tf.keras.models.save_model(network, 'model-savedmodel', save_format="tf")
print('export saved model.')
del network  # 删除网络对象
network = tf.keras.models.load_model('model-savedmodel')  # 从文件恢复网络结构与网络参数