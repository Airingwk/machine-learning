"""
Title: Variational AutoEncoder （based on Keras cvae）
Author: Cam
Date created: 2024/08/07
Last modified: 2024/08/10
Description: Variational AutoEncoder (VAE) trained on MNIST digits. Based on Keras3.4.1,tensorflow2.17.0.
Accelerator: GPU
Reference: https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
Reference paper: https://arxiv.org/pdf/1312.6114
Notes: This code is successfully running on Pycharm2023.3.2 , Windows 11.
"""

"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers



batch_size = 100
hidden_dim = 500
epochs = 50

"""
## Create a sampling layer
"""


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
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon  # reparameterization: z = mu + sigma * epsilon


"""
## Build the encoder  # 通过 函数式API 方式， 构建encoder模型
"""

latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))  # encoder_inputs.shape =  (None, 28, 28, 1)
# x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)  # 卷积层
# print("x = ", x.shape)  # x.shape =  (None, 14, 14, 32)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# print("x = ", x.shape)  # x.shape =  (None, 7, 7, 64)
x = layers.Flatten()(encoder_inputs)  # 将输入数据压缩成一维
print("x = ", x.shape)  # x.shape = (None, 3136)
x = layers.Dense(hidden_dim, activation="relu")(x)  # 神经网络中的全连接层, 输出数量16
print("x = ", x.shape)  # x =  (None, 16)
print("x = ", x.dtype)  # x =  float32
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
print("z_mean.shape = ", z_mean.shape)  # z_mean.shape =  (None, 2)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
print("z_log_var.shape = ", z_log_var.shape)  # z_log_var.shape =  (None, 2)
z = Sampling()([z_mean, z_log_var])  # 基于后验分布p(z|x)的均值和方差，采样z，采样一次就够了
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder  # 通过 函数式API 方式， 构建decoder模型
"""

latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# print("x.shape = ", x.shape)  # x.shape =  (None, 3136)
# x = layers.Reshape((7, 7, 64))(x)
# print("x.shape = ", x.shape)  # x.shape =  (None, 7, 7, 64)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)  # 反卷积
# print("x.shape64 = ", x.shape)  # x.shape64 =  (None, 14, 14, 64)
# x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# print("x.shape32 = ", x.shape)  # x.shape32 =  (None, 28, 28, 32)
# decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# print("decoder_outputs.shape = ", decoder_outputs.shape)  # decoder_outputs.shape =  (None, 28, 28, 1)

x = layers.Dense(hidden_dim, activation="relu")(latent_inputs)
x = layers.Dense(28 * 28 * 1, activation="sigmoid")(x)
print("x.shape = ", x.shape)  # x.shape =  (None, 784)
decoder_outputs = layers.Reshape((28, 28, 1))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
# 通过 子类化模型 方式， 构建VAE模型
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")  # 对传入的 total_loss 求均值
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):  # 重载Keras的train_step
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)  # 这里的z是基于后验分布采样得到的z
            reconstruction = self.decoder(z)  # 编码重构：基于采样的z,重构x
            reconstruction_loss = ops.mean(  # 重构损失
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))  # 为什么是负的？因为采用Adam优化器，梯度下降求最小值
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)  # 计算梯度：total_loss 对 self.trainable_weights 的梯度
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))  # 更新训练参数的梯度，因为采用的是Adam优化器，所以这里是梯度下降求最小值
        self.total_loss_tracker.update_state(total_loss)  # 更新 total_loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE
"""

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)  # 拼接输入的向量
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
print("mnist_digits.shape", mnist_digits.shape)  # mnist_digits.shape = (70000, 28, 28, 1)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())  # 随机梯度下降算法
vae.fit(mnist_digits, epochs=epochs, batch_size=batch_size)

"""
## Display a grid of sampled digits
"""

import matplotlib.pyplot as plt


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))  # 返回一个 digit_size * n 行，digit_size * n 列的全零矩阵
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)  # 从-scale到scale，间隔为n-1的数值序列
    # print("grid_x = ", grid_x)  # 长度为 n 的数组
    grid_y = np.linspace(-scale, scale, n)[::-1]  # [::-1] 倒序全部取值
    # print("grid_y = ", grid_y)
    for i, yi in enumerate(grid_y):  # 返回索引和对应的元素
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])  # 输出是1行2列的矩阵 例如：z_sample =  [[-0.79310345 -0.31034483]]
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            # print("x_decoded.shape = ", x_decoded.shape)  # x_decoded.shape =  (1, 28, 28, 1)
            # print("x_decoded = ", x_decoded)
            digit = x_decoded[0].reshape(digit_size, digit_size)  # digit.shape =  (28, 28)
            # print("digit = ", digit)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
            # print("figure = ", figure)

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2  # 整除
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(vae)

"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)