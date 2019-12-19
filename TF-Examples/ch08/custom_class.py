# 自定义类
"""
尽管 Keras 提供了很多的常用网络层，但深度学习可以使用的网络层远远不止这些经典的网络层，对于需要创建自定义逻辑的网络层，可以通过自定义类来实现。
在创建自定义网络层类时，需要继承自layers.Layer 基类；创建自定义的网络类，需要继承自keras.Model 基类，这样产生的自定义类才能够方便的利用Layer/Model 基类提供的参数管
理功能，同时也能够与其他的标准网络层类交互使用。
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

# 1.自定义网络层
"""
对于自定义的网络层，需要实现初始化__init__方法和前向传播逻辑call 方法。我们以某个具体的自定义网络层为例，假设我们需要一个没有偏置的全连接层，即bias 为0，
同时固定激活函数为ReLU 函数。尽管这可以通过标准的Dense 层创建，但我们还是实现这个自定义类。
"""
class MyDense(layers.Layer):
    # 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
        """
        self.add_variable 会返回此张量的python 引用，而变量名name 由TensorFlow 内部维护，使用的比较少。
        """

    def call(self, inputs, training=None):
        # 实现自定义类的前向计算逻辑
        # X@W
        out = inputs @ self.kernel
        # 执行激活函数运算
        out = tf.nn.relu(out)
        return out

net = MyDense(4, 3)
print(net.variables)  # 类的参数列表
"""
[<tf.Variable 'w:0' shape=(4, 3) dtype=float32, numpy=
array([[-0.86399794, -0.3858019 , -0.48228753],
       [ 0.12365818,  0.65513134,  0.46821296],
       [-0.12535477,  0.21050882, -0.18889159],
       [ 0.41835344, -0.26609415,  0.64884424]], dtype=float32)>]
"""
print(net.trainable_variables)  # 类的待优化参数列表
"""
[<tf.Variable 'w:0' shape=(4, 3) dtype=float32, numpy=
array([[-0.86399794, -0.3858019 , -0.48228753],
       [ 0.12365818,  0.65513134,  0.46821296],
       [-0.12535477,  0.21050882, -0.18889159],
       [ 0.41835344, -0.26609415,  0.64884424]], dtype=float32)>]
"""

# 通过修改为self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=False)，我们可
# 以设置张量不需要被优化，此时再来观测张量的管理状态
class MyDense2(layers.Layer):
    # 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super(MyDense2, self).__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=False)
        """
        self.add_variable 会返回此张量的python 引用，而变量名name 由TensorFlow 内部维护，使用的比较少。
        """
net2 = MyDense2(4, 3)
print(net2.variables)  # 类的参数列表
"""
[<tf.Variable 'w:0' shape=(4, 3) dtype=float32, numpy=
array([[-0.86399794, -0.3858019 , -0.48228753],
       [ 0.12365818,  0.65513134,  0.46821296],
       [-0.12535477,  0.21050882, -0.18889159],
       [ 0.41835344, -0.26609415,  0.64884424]], dtype=float32)>]
"""
print(net2.trainable_variables)  # 类的待优化参数列表
"""
[]
"""
"""
可以看到，此时张量并不会被trainable_variables 管理。此外，类初始化中通过tf.Variable
添加的类成员变量也会自动纳入张量管理中
"""
class MyDense3(layers.Layer):
    # 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super(MyDense3, self).__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = tf.Variable(tf.random.normal([inp_dim, outp_dim]), trainable=False)
        """
        self.add_variable 会返回此张量的python 引用，而变量名name 由TensorFlow 内部维护，使用的比较少。
        """

    # 自定义类的前项运算逻辑，对于我们这个例子，只需要完成𝑂 = 𝑋@𝑊矩阵运算，并通过激活函数即可:
    def call(self, inputs, training=None):
        # 实现自定义类的前向计算逻辑
        # X@W
        out = inputs @ self.kernel
        # 执行激活函数运算
        out = tf.nn.relu(out)
        return out
net3 = MyDense3(4, 3)
print(net3.variables)  # 类的参数列表
"""
[<tf.Variable 'Variable:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.6241027 , -0.22049993, -1.4134765 ],
       [ 1.4997742 ,  0.9707121 , -1.1954745 ],
       [-0.05857228, -0.5735732 , -1.1255926 ],
       [ 1.485413  ,  0.8672904 ,  1.1035906 ]], dtype=float32)>]
"""
print(net3.trainable_variables)  # 类的待优化参数列表
"""
[]
"""


# 2.自定义网络
"""
通过堆叠我们的自定义类，一样可以实现5 层的全连接层网络，每层全连接层无偏置张量，同时激活函数固定使用ReLU。
"""
network = Sequential([
    MyDense(784, 256),  # 使用自定义的层
    MyDense(256, 128),
    MyDense(128, 64),
    MyDense(64, 32),
    MyDense(32, 10)])
network.build(input_shape=(None, 28*28))
network.summary()


class MyModel(Model):
    # 自定义网络类，继承自Model 基类
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        # 自定义前向运算逻辑
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x