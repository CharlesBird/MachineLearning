# 可视化
"""
在网络训练的过程中，通过Web 端监控网络的训练进度，可视化网络的训练结果对于
提高开发效率是非常重要的。TensorFlow 提供了一个专门的可视化工具，叫做
TensorBoard，他通过TensorFlow 将监控数据写入到文件系统，并利用Web 后端监控对应
的文件目录，从而可以允许用户从远程查看网络的监控数据。
"""
import tensorflow as tf

# 1.模型端

# 创建监控类，监控数据将写入log_dir 目录
log_dir = ''
summary_writer = tf.summary.create_file_writer(log_dir)


# 2.浏览器端
"""
在运行程序时，通过运行tensorboard --logdir path 指定Web 后端监控的文件目录path
"""


"""
实际上，除了TensorBoard 工具可以无缝监控TensorFlow 的模型数据外，Facebook 开
发的Visdom 工具同样可以方便可视化数据，并且支持的可视化方式更丰富，实时性更
高，使用起来更加方便，图 8.9 展示了Visdom 数据的可视化方式。Visdom 可以直接接受
PyTorch 的张量数据，但不能直接接受TensorFlow 的张量类型，需要转换为Numpy 数据。
对于追求多样性的可视化手段和实时性的读者，Visdom 可能是更好的选择。
"""