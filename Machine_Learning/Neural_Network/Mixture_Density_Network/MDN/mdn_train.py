"""A script that shows how to use the MDN. It's a simple MDN with a single
nonlinearity that's trained to output 1D samples given a 2D input.
"""
import matplotlib.pyplot as plt
import sys
# ys.path.append('../mdn')  # 用于添加自己编写的包的路径
import mdn_model
import torch
import torch.nn as nn
import torch.optim as optim

# 输入为2维的向量，输出为3维的向量，高斯分布的成分有5个
input_dims = 2
output_dims = 3
num_gaussians = 5


def translate_cluster(cluster, dim, amount):
    """Translates a cluster in a particular dimension by some amount
    torch.add_:
            一般来说函数加了下划线的属于内建函数，将要改变原来的值，没有加下划线的并不会改变原来的数据，
            引用时需要另外赋值给其他变量
    """
    translation = torch.ones(cluster.size(0)) * amount
    cluster.transpose(0, 1)[dim].add_(translation)
    return cluster

# 准备样本数据
print("Generating training data... ", end='')
cluster1 = torch.randn((50, input_dims + output_dims)) / 4    # shape ：[50, input_dims + output_dims]
cluster1 = translate_cluster(cluster1, 1, 1.2)   # shape ：[50, input_dims + output_dims]
cluster2 = torch.randn((50, input_dims + output_dims)) / 4
cluster2 = translate_cluster(cluster2, 0, -1.2)
cluster3 = torch.randn((50, input_dims + output_dims)) / 4
cluster3 = translate_cluster(cluster3, 2, -1.2)
training_set = torch.cat([cluster1, cluster2, cluster3])   # shape ：[150, input_dims + output_dims]
print('Done')

# 创建模型并初始化
print("Initializing model... ", end='')
model = nn.Sequential(
    nn.Linear(input_dims, 5),
    nn.Tanh(),
    mdn_model.MDN(5, output_dims, num_gaussians)
)
optimizer = optim.Adam(model.parameters())
print('Done')

# 训练模型
print('Training model... ', end='')
sys.stdout.flush()  # 显示地让缓冲区的内容输出
# training_set的前两列作为训练数据，后一列作为预测值，对应in_features和out_features
for epoch in range(1000):
    model.zero_grad()  # 模型参数的梯度置为0
    pi, sigma, mu = model(training_set[:, 0:input_dims])
    loss = mdn_model.mdn_loss(pi, sigma, mu, training_set[:, input_dims:])
    loss.backward()  # 计算梯度
    optimizer.step()   # 通过梯度，更新参数
    if epoch % 100 == 99:
        print(f' {round(epoch/10)}%', end='')
        sys.stdout.flush()
print('Done')

# 查看训练后的模型的参数
# parameters = list(model.parameters())
# print("model.parameters: ", parameters)
# print("模型的参数数量:", len(parameters)) # 模型参数有8个，分别为单层全连接神经网络的weigh和bias参数，MDN网络的混合系数alfa、sigma、mu所对应训练网络的weight和bias参数
# # 遍历和操作每个参数
# for parameter in parameters:
#     print(parameter.size())

# 从训练后的MDN模型采样
# 这一步骤用来计算预测值，实际上均值已经可以表示预测值，作者在这里加上了方差*随机噪声用来表示模型的不确定性
print('Generating samples... ', end='')
pi, sigma, mu = model(training_set[:, 0:input_dims])
samples = mdn_model.sample(pi, sigma, mu)
print('Done')

print('Saving samples.png... ', end='')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = training_set[:, 0]
ys = training_set[:, 1]
zs = training_set[:, 2]

ax.scatter(xs, ys, zs, label='target')
ax.scatter(xs, ys, samples[:, 2], label='samples')
ax.legend()
fig.savefig('samples.png')
print('Done')

