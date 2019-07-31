# Manipulate data with ndarray

import numpy as np
from mxnet import nd    # nd属于mxnet

# 指定内容数组 位于cpu上
array_1 = nd.array(((1, 2, 3), (4, 5, 6)))
print(array_1)

# 指定shape 生成全1矩阵 shape可以是[] 也可以是()
array_2 = nd.ones([2, 3])
array_3 = nd.ones((2, 3))
print(array_2)
print(array_3)

# 指定shape，值全为2 tf2中是tf.fill(shape, value)
array_4 = nd.full((2, 3), 5)
print(array_4)

# shape为（2, 3）值随即均匀分布在-1,1之间
array_5 = nd.random.uniform(-1, 1, (2, 3))
print(array_5)

# 获取shape, size, dtype 直接.
# shape: (2, 3) size: 6 dtype:numpy.float32
# 虽然dtype带一个numpy， 但是不能和numpy直接操作 
print(array_5.shape, array_5.size, array_5.dtype)

# 与np互相转化
np_array = np.zeros((2, 3))
print(np_array)
# 不能和numpy直接操作
try:
    print(np_array + array_5)
except:
    print("不能和numpy直接操作 需要通过 .asnumpy()/nd.array() 进行转化")
# 需要转化后才能操作
print(np_array + array_5.asnumpy())     # mxnet转np
print(nd.array(np_array) + array_5)     # np转mxnet

# 转置矩阵
print(array_5.T)

# 乘法运算：逐元素相乘
one = nd.ones((3, 3))
random_array = nd.random.uniform(-1, 1, (3, 3))
print("one: {}".format(one))
print("随机矩阵: {}".format(random_array))
print("相乘： {}".format(one * random_array))

# 乘法运算：矩阵相乘
array_6 = nd.full((2, 3), 5)
array_7 = nd.full((3, 2), 4)
print(nd.dot(array_6, array_7))

# 矩阵取幂 我没看懂是不是取幂运算
array_8 = nd.array(((1,0), (0,1)))
print(array_8)
print("取幂: {}".format(array_8.exp()))

# 索引
array_9 = nd.random.uniform(-1, 1, (4, 5))
print(array_9)
print(array_9[2,2])
print(array_9[..., 0]) # 居然不支持这么骚的操作