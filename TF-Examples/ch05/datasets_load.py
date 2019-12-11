# 经典数据集加载
import tensorflow as tf

(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # 加载 MNIST 数据集
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

# 通过 Dataset.from_tensor_slices 可以将训练部分的数据图片 x 和标签 y 都转换成Dataset 对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
print(train_db)


# 随机打散
# 通过 Dataset.shuffle(buffer_size)工具可以设置 Dataset 对象随机打散数据之间的顺序，
# 防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息
train_db = train_db.shuffle(10000)
print(train_db)


# 批训练
# 为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们
# 把这种训练方式叫做批训练，其中样本的数量叫做 batch size。
train_db = train_db.batch(128)
print(train_db)


# 预处理
# 从 keras.datasets 中加载的数据集的格式大部分情况都不能满足模型的输入要求，因此
# 需要根据用户的逻辑自己实现预处理函数。Dataset 对象通过提供 map(func)工具函数可以
# 非常方便地调用用户自定义的预处理逻辑，它实现在 func 函数里
def preprocess(x, y):  # 自定义的预处理函数
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.  #  标准化到 0~1
    x = tf.reshape(x, [-1, 28*28])  # 打平
    y = tf.cast(y, dtype=tf.int32)  # 转成整形张量
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x, y

train_db = train_db.map(preprocess)  # 预处理函数实现在 preprocess 函数中，传入函数引用即可
print(train_db)


# 循环训练
# 通过多个 step 来完成整个训练集的一次迭代，叫做一个 epoch。在实际训练时，
# 通常需要对数据集迭代多个 epoch 才能取得较好地训练效果
for epoch in range(20):  # 训练 Epoch 数
    for step, (x, y) in enumerate(train_db):  # 迭代 Step 数
        print(step, x, y)

# 也可以通过设置repeat，
train_db = train_db.repeat(20)  # 数据集跌打 20 遍才终止
for step, (x, y) in enumerate(train_db):
    print(step, x, y)
"""
使得 for x,y in train_db 循环迭代 20 个 epoch 才会退出。不管使用上述哪种方式，都能取得一样的效果。
"""