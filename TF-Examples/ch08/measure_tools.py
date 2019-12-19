# 测量工具
"""
在网络的训练过程中，经常需要统计准确率，召回率等信息，除了可以通过手动计算并平均方式获取统计数据外，
Keras 提供了一些常用的测量工具keras.metrics，专门用于统计训练过程中需要的指标数据。
Keras 的测量工具的使用一般有4 个基本操作流程：新建测量器，写入数据，读取统计数据和清零测量器。
"""
import tensorflow as tf
from tensorflow import keras

# 1.新建测量器
# 新建平均测量器，适合Loss 数据
loss_meter = keras.metrics.Mean()

# 2.写入数据
# 记录采样的数据
loss_meter.update_state()

# 3.读取统计信息
# 打印统计的平均loss
print('loss:', loss_meter.result())

# 4.清除
"""
由于测量器会统计所有历史记录的数据，因此在合适的时候有必要清除历史状态。通
过reset_states()即可实现。例如，在每次读取完平均误差后，清零统计信息，以便下一轮统计的开始：
"""
loss_meter.reset_states()  # 清零测量器

# 5.准确率统计实战
"""
按照测量工具的使用方法，我们利用准确率测量器Accuracy 类来统计训练过程中的准
确率。首先新建准确率测量器：
"""
acc_meter = keras.metrics.Accuracy()  # 创建准确率测量器
"""
在每次前向计算完成后，记录训练准确率。需要注意的是，Accuracy 类的update_state 函
数的参数为预测值和真实值，而不是已经计算过的batch 的准确率：
"""
# 根据预测值与真实值写入测量器
acc_meter.update_state()
"""
在统计完测试集所有batch 的预测值后，打印统计的平均准确率并清零测量器。
"""
# 读取统计结果
print('Evaluate Acc:', acc_meter.result().numpy())
acc_meter.reset_states()  # 清零测量器