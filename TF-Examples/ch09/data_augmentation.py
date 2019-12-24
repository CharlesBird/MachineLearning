# 数据增强
"""
除了上述介绍的方式可以有效检测和抑制过拟合现象之外，增加数据集大小是解决过
拟合最重要的途径。但是收集样本数据和标注往往是代价昂贵的，在有限的数据集上，通
过数据增强技术可以增加训练的样本数量，获得一定程度上的性能提升。数据增强(Data
Augmentation)是指在维持样本标签不变的条件下，根据先验知识改变样本的特征，使得新
产生的样本也符合或者近似符合数据的真实分布。
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# 图片缩放
x = tf.io.read_file('lenna.png')
x = tf.io.decode_jpeg(x, channels=3)  # RGBA
# 图片缩放到244x244 大小，这个大小根据网络设定自行调整
resize_x = tf.image.resize(x, [244, 244])
resize_x = tf.cast(resize_x, tf.float32) / 255
plt.imsave('lenna_resize.png', resize_x.numpy())

# 旋转
# 通过 tf.image.rot90(x, k=1)可以实现图片按逆时针方式旋转k 个90 度
rot_x = tf.image.rot90(x, 2)  # 图片逆时针旋转180 度
rot_x = tf.cast(rot_x, tf.float32) / 255
plt.imsave('lenna_rotate.png', rot_x.numpy())

# 翻转
# 随机水平翻转
flip_x = tf.image.random_flip_left_right(x)
flip_x = tf.cast(flip_x, tf.float32) / 255
plt.imsave('lenna_flip.png', flip_x.numpy())
# 随机竖直翻转
flip_x = tf.image.random_flip_up_down(x)
flip_x = tf.cast(flip_x, tf.float32) / 255
plt.imsave('lenna_flip2.png', flip_x.numpy())

# 裁剪
# 图片先缩放到稍大尺寸
resize_x = tf.image.resize(x, [244, 244])
# 再随机裁剪到合适尺寸
crop_x = tf.image.random_crop(resize_x, [224, 224, 3])
crop_x = tf.cast(crop_x, tf.float32) / 255
plt.imsave('lenna_crop.png', crop_x.numpy())
crop_x = tf.image.random_crop(resize_x, [224, 224, 3])
crop_x = tf.cast(crop_x, tf.float32) / 255
plt.imsave('lenna_crop2.png', crop_x.numpy())

quality_x = tf.image.random_jpeg_quality(x, 50, 60)
quality_x = tf.cast(quality_x, tf.float32) / 255
plt.imsave('lenna_quality.png', quality_x.numpy())

# 色调
hue_x = tf.image.random_hue(x, 0.4)
hue_x = tf.cast(hue_x, tf.float32) / 255
plt.imsave('lenna_hue.png', hue_x.numpy())

# 对比度
contrast_x = tf.image.random_contrast(x, 10, 60)
contrast_x = tf.cast(contrast_x, tf.float32) / 255
plt.imsave('lenna_contrast.png', contrast_x.numpy())

