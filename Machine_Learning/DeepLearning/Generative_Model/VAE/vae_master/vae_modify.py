import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, InputLayer, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
import tensorflow.compat.v1 as tf
import pandas as pd

tf.disable_v2_behavior()

# Load Data and map gray scale 256 to number between zero and 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

print(x_train.shape)

# Find dimensions of input images
img_rows, img_cols, img_chns = x_train.shape[1:]

# Specify hyperparameters
original_dim = img_rows * img_cols
intermediate_dim = 256
latent_dim = 2
batch_size = 100
epochs = 10
epsilon_std = 1.0


def nll(y_true, y_pred):
    """Negative log likelihood (Bernoulli)."""

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch))

        return inputs


class InimLayer(Layer):
    def call(self, inputs):
        return K.shape(inputs)[0]


# Encoder

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation="relu")(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

# Reparametrization trick
z_sigma = Lambda(lambda t: K.exp(0.5 * t))(z_log_var)

eps = Input(tensor=K.random_normal(shape=(InimLayer()(x), latent_dim)))

z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

# This defines the Encoder which takes noise and input and outputs
# the latent variable z
encoder = Model(inputs=[x, eps], outputs=z)

# Decoder is MLP specified as single Keras Sequential Layer
decoder = Sequential(
    [
        Dense(intermediate_dim, input_dim=latent_dim, activation="relu"),
        Dense(original_dim, activation="sigmoid"),
    ]
)

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred, name="vae")
vae.compile(optimizer="rmsprop", loss=nll)
vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim) / 255.0
x_test = x_test.reshape(-1, original_dim) / 255.0

hist = vae.fit(
    x_train,
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, x_test),
)


# %matplotlib inline

# for pretty plots
golden_size = lambda width: (width, 2.0 * width / (1 + np.sqrt(5)))

fig, ax = plt.subplots(figsize=golden_size(6))

hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)

ax.set_ylabel("NELBO")
ax.set_xlabel("# epochs")

ax.set_ylim(0.99 * hist_df[1:].values.min(), 1.1 * hist_df[1:].values.max())
plt.show()


x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=golden_size(6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap="nipy_spectral")
plt.colorbar()
plt.show()