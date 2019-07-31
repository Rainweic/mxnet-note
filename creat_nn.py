from mxnet import nd
from mxnet.gluon import nn  # 神经网络模块都在gluon.nn中

# 创建一个全链接层
layer = nn.Dense(2)
# 输出信息
print(layer)
# 初始化权重 随即初始化值在[-0.7, 0.7]之间
layer.initialize()

# 生成一个数据看看
x = nd.random.uniform(-1, 1, (3, 4))    # shape=(3, 4)
# 将x输入layer
y = layer(x)    # 输出 y 将是 shape=(3, 2) 的矩阵 mxnet会自动设置输入的纬度
print(y)
# 输出权重看看(要先输入数据，获得shape 否则会报错 原因：没有输入，layer就没有shape，无法生成数据)
print(layer.weight.data())  # shape = (2, 4)
# layer的weight在第一次输入时候就被设置了 输入的shape必须一致
try:
    layer(nd.random.uniform(-1, 1, (5, 6)))
except:
    print("报错：{}".format("mxnet.base.MXNetError: Shape inconsistent, Provided = [2,4], inferred shape=(2,6)"))
    print("是说提供的 weight 的shape为[2, 4]， 但是输入所需的weight的shape为[2, 6]")

# 使用 Sequential() 创建model
net = nn.Sequential()
# 通过.add()增加层
net.add(
    nn.Conv2D(channels=6, kernel_size=6, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation='relu'),
    nn.Dense(86, activation='relu'),
    nn.Dense(10)
)
net.initialize()
x = nd.random.uniform(shape=(4, 1, 28, 28))
y = net(x)
print(y.shape)
print(net)  # 输出网络 方便查看
# 可以通过索引获取net中指定layer
print(net[1])
print(net[0].weight.data())
print(net[0].bias.data())


# 通过继承nn.Block创建类进行网络创建 更为灵活
# 要实现 __init__ forward 俩个函数
class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(
            nn.Dense(3, activation='relu'),
            nn.Dense(4, activation='relu')
        )
        self.dense = nn.Dense(5)    # 通过self.xxx对该层命名为xxx
    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)

net = MixMLP()
print(net)  # 输出网络 通过各层名称获取信息

net.initialize()
x = nd.random.uniform(shape=(2,2))
net(x)
print(net.blk[1].weight.data())