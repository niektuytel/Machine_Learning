import matplotlib.pyplot as plt
import numpy as np
import sys, os

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

    def sampling(self, mu, log_var):
        epsilon = keras.backend.random_normal(keras.backend.shape(mu))
        return mu + keras.backend.exp(log_var / 2) * epsilon

    def build_encoder(self):
        image_input = Input(shape=self.img_shape)

        # Encoder
        h = Flatten()(image_input)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)

        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = self.sampling(mu, log_var)
        
        return Model(image_input, latent_repr)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"./sample_keras_output/image_{epoch}.png")
        plt.close()

if __name__ == '__main__':
    os.makedirs("sample_keras_output", exist_ok=True)
    network = AdversarialAutoencoder()
    network.train(epochs=20000, batch_size=32, sample_interval=200)



# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np

# import keras
# from keras import layers
# from keras.datasets import mnist

# # data
# (x_train, y_train), (x_test, _) = mnist.load_data()
# x_data = np.concatenate([x_train, x_test], axis=0)
# x_data = np.expand_dims(x_data, -1).astype("float32") / 255
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255

# # settings
# n_epochs = 30
# batch_size = 128

# # Create a sampling layer
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# class VAE(keras.Model):
#     def __init__(self, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         latent_dim = 2

#         # encoder
#         encoder_inputs = keras.Input(shape=(28, 28, 1))
#         x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
#         x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(16, activation="relu")(x)

#         z_layers = [layers.Dense(latent_dim)(x), layers.Dense(latent_dim)(x)]
#         z_layers.append(Sampling()(z_layers))
#         self.encoder = keras.Model(encoder_inputs, z_layers)
#         self.encoder.summary()

#         # decoder
#         latent_inputs = keras.Input(shape=(latent_dim,))
#         x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
#         x = layers.Reshape((7, 7, 64))(x)
#         x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#         x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#         x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
#         self.decoder = keras.Model(latent_inputs, x)
#         self.decoder.summary()


#         self.total_loss_tracker = keras.metrics.Mean()

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum( keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2) )
#             )
#             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#             total_loss = reconstruction_loss + kl_loss

#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)

#         return { "loss": self.total_loss_tracker.result() }

# # define network
# network = VAE()
# network.compile(optimizer=keras.optimizers.Adam())
# network.fit(x_data, epochs=n_epochs, batch_size=batch_size)

# def plot_latent_space(network, n=30, figsize=15):
#     # Display a n*n 2D manifold of digits
#     digit_size = 28
#     scale = 1.0
#     figure = np.zeros((digit_size * n, digit_size * n))

#     # linearly spaced coordinates corresponding to the 2D plot 
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = network.decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 i * digit_size : (i + 1) * digit_size,
#                 j * digit_size : (j + 1) * digit_size
#             ] = digit

#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)

#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()

# def plot_label_clusters(network, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = network.encoder.predict(data)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()

# plot_label_clusters(network, x_train, y_train)
# plot_latent_space(network)


