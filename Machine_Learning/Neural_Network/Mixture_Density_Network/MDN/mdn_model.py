"""A module for a mixture density network layer
For more info on MDNs, see _C. M. Bishop. Mixture density networks. Technical Report, 1994._
   https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)  # 1/根号2π

# 标注一下输出维度out_features = 1，对应论文从神经网络出来的维度是（c + 2） * m = (1 + 2) * 5,但是mdn里面用了三个linear层来分别表示pi，sigma和mu
class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance. # O is also dimension of output/target vector

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        # 使用3个神经网络得到MDN的参数所对应的的网络输出z_alfa, z_sima, z_mu
        self.pi = nn.Sequential(  # 序贯模型，即顺序连接模型. 这里的pi对应MDN中的混合系数alfa
            nn.Linear(in_features, num_gaussians),  # 线性层，输入神经元数量为in_features， 输出神经元数量为num_gaussians
            nn.Softmax(dim=1)  # 沿着维度1方向softmax，这里是沿着列方向
        )
        self.z_sigma = nn.Linear(in_features, out_features * num_gaussians)  # 对应论文中的网络输出z_sigma
        self.z_mu = nn.Linear(in_features, out_features * num_gaussians)     # 对应论文中的网络输出z_mu

    def forward(self, minibatch):
        pi = self.pi(minibatch)  # shape为[btz, num_gaussians]
        sigma = torch.exp(self.z_sigma(minibatch))  # shape为[btz, out_features * num_gaussians]
        # 因为sigma和mu都与输出的维度有关，所以在这里还要展开，给输出一个维度
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.z_mu(minibatch)                   # shape为[btz, out_features * num_gaussians]
        mu = mu.view(-1, self.num_gaussians, self.out_features)  # shape为[btz, num_gaussians, out_features]
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):  # 对应论文中的第i个核函数kernel function： phi_i
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    example: sigma: torch.Size([150, 5, 1]) mu: torch.Size([150, 5, 1]) target: [150, 1]
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions, here, I = O.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
        返回高斯混合分布的component：phi,如果输出是多维的,则有 exp(a) * exp(b) = exp(a+b)
    """
    target = target.unsqueeze(1).expand_as(mu)  # 将target先在维度1上增加一个维度，再展开成形状与 mu 一致.
                                                # expand_as(mu):这里实际上对target的每一个数据点进行num_gaussians-1次复制
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma  # phi_i为正态分布概率密度
    return torch.prod(ret, 2)  # shape为[btz, num_gaussians], 输出phi


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
        pi: torch.Size([150, 5]) sigma: torch.Size([150, 5, 1]) mu: torch.Size([150, 5, 1])
    The loss is the negative log likelihood of the data given the MoG parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)  # 对应论文中公式（22），p(t|x),这里是batch_size个
    nll = -torch.log(torch.sum(prob, dim=1))  # shape为[btz], 计算出batch_size个 E_q（误差方程）
    return torch.mean(nll)  # 论文中loss function 是求和（公式（28）），这里是求均值，最终优化结果都是一样的


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from,返回采样点的索引
    # 返回的是 均值 + 方差*随机噪声 的形式
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)  # 从分类分布中采样
                                                           # Samples are integers from {0,…,K−1} where K is pi.size(-1).
                                                           # torch.view: 维度变换
                                                           # pi.size(0)返回pi的第1维度的大小
                                                           # pi.size(-1)返回pi的最后一个维度的大小
                                                           # pis.shape: [B, 1, 1]

    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)  # 生成正态随机分布张量，张量大小为[O, B]
    # torch.gather(dim=1) 表示按照列号进行索引，寻找采样的pi对应的sigma
    variance_samples = sigma.gather(1, pis).detach().squeeze()   # sigma.gather(1, pis)：在sigma的第2维度上按照索引张量pis取值
                                                                 # tensor.detach(): 从计算图中分离一个张量，这意味着它创建了
                                                                 # 一个新的张量，与原始张量共享数据，但不再参与任何计算图。
                                                                 # 这意味着这个新的张量不依赖于过去的计算值
                                                                 # variance_samples大小为[B]

    mean_samples = mu.detach().gather(1, pis).squeeze()          # tensor.squeeze():从张量中删除尺寸为1的维度，相当于从内而外剥除符号[]
                                                                 # mean_samples大小为[B]
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)  # Tensor.transpose(0, 1):交换Tensor的第1和第2维度
                                                                                    # 未转置前的大小为[O, B], 转置后的大小为[B, O]