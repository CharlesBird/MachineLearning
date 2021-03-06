# 导数与梯度
"""
导数本身是标量，没有方向，但是导数表征了函数值在某个方向Δ𝒙的变化率。在这些任意Δ𝒙方向中，沿着坐标轴的几个方向比较特殊，
此时的导数也叫做偏导数(Partial Derivative)。对于一元函数，导数记为𝑑𝑦/𝑑𝑥；对于多元函数的偏导数，记为
𝜕𝑦/𝜕𝑥1,𝜕𝑦/𝜕𝑥2, …等。偏导数是导数的特例，也没有方向。
"""

# 基本函数的导数
"""
常数函数c 导数为0，如𝑦 = 2函数导数𝑦′ = 0
线性函数𝑦 = 𝑎 ∗ 𝑥 + 𝑐 导数为𝑎，如函数𝑦 = 2 ∗ 𝑥 + 1导数𝑦′ = 2
幂函数𝑥𝑎 导数为𝑎 ∗ 𝑥𝑎−1，如𝑦 = 𝑥2函数𝑦′ = 2 ∗ 𝑥
指数函数𝑎𝑥 导数为𝑎𝑥 ∗ 𝑙𝑛 𝑎，如𝑦 = 𝑒𝑥函数𝑦′ = 𝑒𝑥 ∗ 𝑙𝑛 𝑒 = 𝑒𝑥
对数函数𝑙𝑜𝑔𝑎 𝑥 导数为1/𝑥𝑙𝑛 𝑎，如𝑦 = 𝑙𝑛 𝑥函数𝑦′ = 1/𝑥𝑙𝑛 𝑒 = 1/𝑥
"""

# 常用导数性质
"""
函数加减 (𝑓 + 𝑔)′ = 𝑓′ + 𝑔′
函数相乘 (𝑓𝑔)′ = 𝑓′ ∗ 𝑔 + 𝑓 ∗ 𝑔′
函数相除 (𝑓/𝑔)′ = (𝑓′𝑔−𝑓𝑔′)/𝑔2，g ≠ 0
复合函数的导数 考虑复合函数𝑓(𝑔(𝑥))，令𝑢 = 𝑔(𝑥)，其导数 𝑑𝑓(𝑔(𝑥))/𝑑𝑥 = 𝑑𝑓(𝑢)/𝑑𝑢 * 𝑑𝑔(𝑥)/𝑑𝑥 = 𝑓′(𝑢) ∗ 𝑔′(𝑥)
"""