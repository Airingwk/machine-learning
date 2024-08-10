#! -*- coding: utf-8 -*-

'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
# from tensorflow.keras.layers import Input, Dense, Lambda
import keras
# print(keras.__version__)
from keras import layers
from keras import Model
from keras import ops
from keras import backend as K

batch_size = 100
original_dim = 784  # 784
latent_dim = 2  # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256
epochs = 50


# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
# print("mnist_digits.shape = ", mnist_digits.shape)  # mnist_digits.shape =  (70000, 28, 28, 1)

# x_train = x_train.reshape(-1, original_dim) / 255.0
# x_test = x_test.reshape(-1, original_dim) / 255.0
# print("x_train.shape = ", x_train.shape)  # x_train.shape =  (60000, 28, 28)
# print("x_test.shape = ", x_test.shape)

# 图像数据归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# mnist_digits = mnist_digits.astype('float32') / 255.
# print("x_train.shape = ", x_train.shape)  # x_train.shape =  (60000, 28, 28)
# print("x_test.shape = ", x_test.shape)  # x_test.shape =  (10000, 28, 28)

# 将图像数据转换为784维的向量
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# mnist_digits = mnist_digits.reshape((len(mnist_digits), np.prod(mnist_digits.shape[1:])))
# print("x_train.shape = ", x_train.shape)  # x_train.shape =  (60000, 784)
# print("x_train.dtype = ", x_train.dtype)  # float32
# print("x_test.shape = ", x_test.shape)  # x_test.shape =  (10000, 784)
# print("x_test.dtype = ", x_train.dtype)  # float32

x = keras.layers.Input(shape=(original_dim,))
# print("x.shape = ", x.shape)  # x.shape =  (None, 784)
# print("x.dtype = ", x.dtype)  # x.dtype =  float32

h = keras.layers.Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)


# 重参数技巧
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# 重参数层，相当于给输入加入噪声
# z = keras.layers.Lambda(Sampling(), output_shape=(latent_dim,))([z_mean, z_log_var])
z = Sampling()([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = keras.layers.Dense(intermediate_dim, activation='relu')
decoder_mean = keras.layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
# print("x_decoded_mean.shape", x_decoded_mean.shape)

# 建立模型
vae = Model(x, x_decoded_mean)


class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        x, x_decoded_mean = inputs
        loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
        return loss


# xent_loss是重构loss，kl_loss是KL loss
bin_CrossEntropy = KLDivergenceLayer()([x, x_decoded_mean])
xent_loss = ops.sum(bin_CrossEntropy, axis=-1)
kl_loss = - 0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
vae_loss = ops.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
# vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.RMSprop(), loss=KLDivergenceLayer)
# vae.summary()
# print("mnist_digits.shape = ", mnist_digits.shape)
x_train = x_train[:10000, :]
vae.fit(x=x_train,
        y=x_test,
        # shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        # validation_data=(x_test, x_test),
        )

# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

# 将所有测试集中的图片通过encoder转换为隐含变量（二维变量），并将其在二维空间中进行绘图
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()

# 构建生成器
decoder_input = keras.layers.Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 观察隐变量的两个维度变化是如何影响输出结果的
n = 15  # figure with 15x15 digits
digit_size = 28 # 每个图像的大小为28*28
figure = np.zeros((digit_size * n, digit_size * n)) # 初始化为0

# 用正态分布的分位数来构建隐变量对
# 生成因变量空间（二维）中的数据，数据满足高斯分布。这些数据构成隐变量，用于图像的生成。
# ppf为累积分布函数（cdf）的反函数，累积分布函数是概率密度函数（pdf）的积分。
# np.linspace(0.05, 0.95, n)为累计分布函数的输出值（y值），现在我们需要其对应的x值，所以使用cdf的反函数，这些x值构成隐变量。
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
