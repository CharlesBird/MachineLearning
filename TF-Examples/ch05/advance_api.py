# 高级操作
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tf.gather
# 可以实现根据索引号收集数据的目的
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)
print(x)

print(tf.gather(x, [0, 1], axis=0))  # 在班级维度收集第 1-2 号班级成绩册

# 上述需求，通过切片[:2]可以更加方便实现。但是对于不规则的索引方式，比如，需要抽查所有班级的第 1,4,9,12,13,27 号同学的成绩，
# 则切片方式实现起来非常麻烦，而 tf.gather 则是针对于此需求设计的，使用起来非常方便

print(tf.gather(x, [0,3,8,11,12,26], axis=1))  # 收集第 1,4,9,12,13,27 号同学成绩

print(tf.gather(x, [2, 4], axis=2))  # 第 3，5 科目的成绩

a = tf.range(8)
a = tf.reshape(a, [4, 2])
print(a)
print(tf.gather(a, [3, 1, 0, 2], axis=0))  # 收集第 4,2,1,3 号元素


"""
如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目成绩，则可以通过组合多个 tf.gather 实现
"""
students = tf.gather(x, [1, 2], axis=0)  # 收集第 2,3 号班级
print(students)
print(tf.gather(students, [2,3,5,26], axis=1))  # 收集第 3,4,6,27 号同学

"""
这次我们希望抽查第 2 个班级的第 2 个同学的所有科目，
第 3 个班级的第 3 个同学的所有科目，第 4 个班级的第 4 个同学的所有科目。那么怎么实现呢？
"""
# 通过笨方式一个一个的手动提取
print(tf.stack([x[1, 1], x[2, 2], x[3, 3]], axis=0))  # 最后通过 stack方式合并采样结果
"""
它最大的问题在于手动串行方式执行采样，计算效率极低。有没有更好的方式实现呢？这就是下一节要介绍的 tf.gather_nd 的功能。
"""

# tf.gather_nd
# 可以通过指定每次采样的坐标来实现采样多个点的目的
"""
那么这 3 个采样点的索引坐标可以记为:[1,1],[2,2],[3,3]，我们将这个采样方案合并为一个 List 参数：[[1,1],[2,2],[3,3]]，
通过tf.gather_nd 实现
"""
print(tf.gather_nd(x, [[1,1], [2,2], [3,3]]))  # 根据多维度坐标收集数据

print(tf.gather_nd(x, [[1,1,2], [2,2,3], [3,3,4]]))


# tf.boolean_mask
# 以通过给定掩码(mask)的方式采样
print(tf.boolean_mask(x, mask=[True, False,False,True], axis=0))  # 采样第 1 和第 4 个班级
"""
注意掩码的长度必须与对应维度的长度一致，如在班级维度上采样，则必须对这 4 个班级是否采样的掩码全部指定，掩码长度为 4
"""
print(tf.boolean_mask(x, mask=[True,False,False,True,True,False,False,True], axis=2))  # 根据掩码方式采样科目

# 采样第 1 个班级的第 1-2 号学生，第 2 个班级的第 2-3 号学生，通过tf.gather_nd 可以实现为
x = tf.random.uniform([2, 3, 8], maxval=100, dtype=tf.int32)
print(tf.gather_nd(x, [[0,0], [0,1], [1,1], [1,2]]))  # 多维坐标采集

# 成绩册掩码采样方案
"""
    学生0 学生1 学生2
班级0 True True False
班级1 False True True
"""
print(tf.boolean_mask(x, [[True,True,False],[False,True,True]]))  # 多维掩码采样


# tf.where
# 通过 tf.where(cond, a, b)操作可以根据 cond 条件的真假从 a 或 b 中读取数据
a = tf.ones([3, 3])  # 构造 a 为全 1
b = tf.zeros([3, 3])  # 构造 b 为全 0
cond = tf.constant([[True,False,False],[False,True,False],[True,True,False]])
print(tf.where(cond, a, b))  # 根据条件从 a,b 中采样

# 当 a=b=None 即 a,b 参数不指定时，tf.where 会返回 cond 张量中所有 True 的元素的索引坐标
print(tf.where(cond, None, None))

# 实际应用例子
x = tf.random.normal([3, 3])
print(x)
# 通过比较运算，得到正数的掩码
mask = x > 0  # 比较操作，等同于 tf.equal()
print(mask)
# 通过 tf.where 提取此掩码处 True 元素的索引
indices = tf.where(mask)  # 提取所有大于 0 的元素索引
print(indices)
# 拿到索引后，通过 tf.gather_nd 即可恢复出所有正数的元素
print(tf.gather_nd(x, indices))  # 提取正数的元素值
# 实际上，当我们得到掩码 mask 之后，也可以直接通过 tf.boolean_mask 获取对于元素
print(tf.boolean_mask(x, mask))  # 通过掩码提取正数的元素值


# scatter_nd
# 通过 tf.scatter_nd(indices, updates, shape)可以高效地刷新张量的部分数据，但是只能在全 0 张量的白板上面刷新，
# 因此可能需要结合其他操作来实现现有张量的数据刷新功能。
indices = tf.constant([[4], [3], [1], [7]])  # 构造需要刷新数据的位置
updates = tf.constant([4.4, 3.3, 1.1, 7.7])  # 构造需要写入的数据
print(tf.scatter_nd(indices, updates, [8]))  # 在长度为 8 的全 0 向量上根据 indices 写入 updates

indices = tf.constant([[1], [3]])  # 构造写入位置
updates = tf.constant([
 [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],
 [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
])  # 构造写入数据
# 在 shape 为[4,4,4]白板上根据 indices 写入 updates
print(tf.scatter_nd(indices, updates, [4, 4, 4]))


# meshgrid
# 通过 tf.meshgrid 可以方便地生成二维网格采样点坐标，方便可视化等应用场合。
x = tf.linspace(-8., 8, 100)  # 设置 x 坐标的间隔
y = tf.linspace(-8., 8, 100)  # 设置 y 坐标的间隔
x, y = tf.meshgrid(x, y)  # 生成网格点，并拆分后返回
print(x.shape, y.shape)  # 打印拆分后的所有点的 x,y 坐标张量 shape
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z  # sinc 函数实现
fig = plt.figure()
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)  # 根据网格点绘制 sinc 函数 3D 曲面
plt.show()