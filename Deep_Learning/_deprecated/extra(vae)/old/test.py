
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon





# Build the encoder
latent_dim = 2
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Build the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# define VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss":self.total_loss_tracker.result(),
            "reconstruction_loss":self.reconstruction_loss_tracker.result(),
            "kl_loss":self.kl_loss_tracker.result(),
        }


# Trian the VAE
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

# Display a gri of sampled digits
import matplotlib.pyplot as plt

def plot_latent_space(vae, n=30, figsize=15):
    # Display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot 
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
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

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)

plot_latent_space(vae)



# import argparse
# import torch
# import os
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image

# torch.manual_seed(1)

# device = "cpu"
# batch_size = 128
# epochs = 10
# log_interval = 10

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor()), 
#     batch_size=batch_size, 
#     shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data', train=False, transform=transforms.ToTensor()), 
#     batch_size=batch_size, 
#     shuffle=True
# )

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD


# def train(epoch, epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):

#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()

#         if batch_idx % log_interval == 0:
#             data_now = (batch_idx * len(data))
#             data_total = len(train_loader.dataset)
#             procent_passed = (100. * batch_idx / len(train_loader))
#             loss_value = (loss.item() / len(data))
#             print(f"\r[{epoch}/{epochs}] [{data_now}/{data_total}] \t loss: {loss_value}", end="")
#     print(f"  ====> Epoch: {epoch} Average loss: {(train_loss / len(train_loader.dataset))}")

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(), './sample_pytorch_output/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss}')

# def make_dir():
#     image_dir = "./sample_pytorch_output"
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

# make_dir()
# for epoch in range(1, epochs + 1):
#     train(epoch, epochs)
#     test(epoch)
#     with torch.no_grad():
#         sample = torch.randn(64, 20).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28), './sample_pytorch_output/sample_' + str(epoch) + '.png')











# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras.losses import sparse_categorical_crossentropy
# from tensorflow.keras.optimizers import Adam

# # Model configuration
# batch_size = 50
# img_width, img_height, img_num_channels = 32, 32, 3
# loss_function = sparse_categorical_crossentropy
# no_classes = 10
# no_epochs = 100
# optimizer = Adam()
# validation_split = 0.2
# verbosity = 1

# # Load CIFAR-10 data
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Dentermine shape of the data
# input_shape = (img_width, img_height, img_num_channels)

# # Parse numbers as floats
# X_train = X_train.astype("float32")
# X_test = X_test.astype("float32")

# # Scale data
# X_train = X_train / 255
# X_test = X_test / 255

# # Create the model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
# model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
# model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dense(no_classes, activation="softmax"))

# # Compile the model
# model.compile(
#     loss=loss_function,
#     optimizer=optimizer,
#     metrics=["accuracy"]
# )

# # Fit data to model
# history = model.fit(
#     X_train, y_train,
#     batch_size = batch_size,
#     epochs=no_epochs,
#     verbose=verbosity,
#     validation_split=validation_split
# )

# # generate generalization metrics
# score = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")





