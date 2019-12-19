# 模型乐园
"""
对于常用的网络模型，如ResNet，VGG 等，不需要手动创建网络，可以直接从keras.applications 子模块下一行代码即可创建并使用这些经典模型，
同时还可以通过设置weights 参数加载预训练的网络参数，非常方便。
"""
import tensorflow as tf
from tensorflow import keras

# 1.加载模型
"""
以 ResNet50 迁移学习为例，一般将ResNet50 去掉最后一层后的网络作为新任务的特
征提取子网络，即利用ImageNet 上面预训练的特征提取方法迁移到我们自定义的数据集
上，并根据自定义任务的类别追加一个对应数据类别数的全连接分类层，从而可以在预训
练网络的基础上可以快速、高效地学习新任务。
"""
# 加载预训练网络模型，并去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4, 224, 224, 3])
out = resnet(x)
print(out.shape)

"""
上述代码自动从服务器下载模型结构和在ImageNet 数据集上预训练好的网络参数，由于去掉最后一层，网络的输出大小为[b, 7,7,2048]。
对于某个具体的任务，需要设置自定义的输出节点数，以100 类的分类任务为例，我们在ResNet50 基础上重新构建新网络。
新建一个池化层(这里的池化层可以理解为维度缩减功能)，将特征从[b, 7,7,2048]降维到[b, 2048]:
"""
# 新建池化层
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 7, 7, 2048])
out = global_average_layer(x)
print(out.shape)

"""最后新建一个全连接层，并设置输出节点数为100"""
# 新建全连接层
fc = tf.keras.layers.Dense(100)
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 2048])
out = fc(x)
print(out.shape)

"""
在得到预训练的ResNet50 特征层和我们新建的池化层、全连接层后，我们重新利用Sequential 容器封装成一个新的网络：
"""
# 重新包裹成我们的网络模型
mynet = keras.Sequential([resnet, global_average_layer, fc])
mynet.summary()

"""
通过设置resnet.trainable = False 可以选择冻结ResNet 部分的网络参数，只训练新建的网络层，从而快速、高效完成网络模型的训练。
"""
resnet.trainable = False
mynet.summary()