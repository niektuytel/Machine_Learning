# import matplotlib.pyplot as plt
# import numpy as np
# import sys, os

# import keras
# from keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Sequential, Model
# from keras.optimizers import Adam


# sys.path.insert(1, os.getcwd() + "/../../network") 
# import layers
# # from layers import Dense_V2, Dropout, Reshape, Flatten, BatchNormalization, Network_V2

# sys.path.insert(1, "D:\Programming\learn\AI\sample\ML-From-Scratch") 
# from mlfromscratch.deep_learning.optimizers import Adam
# from mlfromscratch.deep_learning.loss_functions import CrossEntropy, SquareLoss
# import mlfromscratch.deep_learning.layers as _layers
# from mlfromscratch.deep_learning import NeuralNetwork


# class AdversarialAutoencoder():
#     def __init__(self):
#         self.channels = 1
#         self.img_rows = self.img_cols = self.image_size = 28
#         self.img_shape = (self.img_rows, self.img_cols, self.channels)
#         self.latent_dim = 10

#         optimizer = Adam(learning_rate=0.0002, b1=0.5)
#         loss_function = CrossEntropy

#         # Build and compile the discriminator
#         self.discriminator = self.build_discriminator(optimizer, loss_function)

#         optimizer = Adam(0.0002, 0.5)
#         self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#         # Build the encoder / decoder
#         self.encoder = self.build_encoder()
#         self.encoder.summary()

#         # self.mu = _layers.Dense(self.latent_dim, input_shape=(512, ))
#         # self.log_var = _layers.Dense(self.latent_dim, input_shape=(512, ))
#         # self.latent_repr = self.sampling(self.mu, self.log_var)
        
#         self.decoder = self.build_decoder(optimizer, loss_function)
#         self.decoder.summary()

#         # The generator takes the image, encodes it and reconstructs it
#         # from the encoding
#         img = Input(shape=self.img_shape)
#         encoded_repr = self.encoder(img)
#         reconstructed_img = self.decoder(encoded_repr)

#         # For the adversarial_autoencoder model we will only train the generator
#         self.discriminator.trainable = False

#         # The discriminator determines validity of the encoding
#         validity = self.discriminator(encoded_repr)

#         # The adversarial_autoencoder model  (stacked generator and discriminator)
#         self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
#         self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

#     def sampling(self, mu, log_var):
#         epsilon = keras.backend.random_normal(keras.backend.shape(mu))
#         return mu + keras.backend.exp(log_var / 2) * epsilon

#     def build_encoder(self):
#         image_input = Input(shape=self.img_shape)

#         # Encoder
#         h = layers.Flatten(input_shape=self.img_shape)#(image_input)
#         h = Dense(512)(h)
#         h = LeakyReLU(alpha=0.2)(h)
#         h = Dense(512)(h)
#         h = LeakyReLU(alpha=0.2)(h)

#         mu = Dense(self.latent_dim)(h)
#         log_var = Dense(self.latent_dim)(h)
#         latent_repr = self.sampling(mu, log_var)
        
#         return Model(image_input, latent_repr)

#         # model = NeuralNetwork(optimizer=optimizer, loss=loss_function)
#         # model.add(_layers.Dense(512, input_shape=self.img_shape))
#         # model.add(_layers.Activation('leaky_relu'))
#         # model.add(_layers.Dense(512, input_shape=(512, )))
#         # model.add(_layers.Activation('leaky_relu'))


#         # # model.add(_layers.Dense(np.prod(self.img_shape), input_shape=(512, )))
#         # # model.add(_layers.Activation('tanh'))
#         # # model.add(_layers.Reshape(self.img_shape))

#         # # return model

#         # # image_input = Input(shape=self.img_shape)

#         # # # Encoder
#         # # h = Flatten()(image_input)
#         # # h = _layers.Dense(512, input_shape=(self.img_shape, ))(h)
#         # # h = _layers.Activation('leaky_relu')(h)
#         # # h = _layers.Dense(512, input_shape=(512, ))(h)
#         # # h = _layers.Activation('leaky_relu')(h)

#         # mu = Dense(self.latent_dim)
#         # log_var = Dense(self.latent_dim)
#         # latent_repr = self.sampling(mu, log_var)
        
#         # return Model(image_input, latent_repr)

#     def build_decoder(self, optimizer, loss_function):
#         # model = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        
#         # model.add(_layers.Dense(512, input_shape=(self.latent_dim, )))
#         # model.add(_layers.Activation('leaky_relu'))
#         # model.add(_layers.Dense(512, input_shape=(512, )))
#         # model.add(_layers.Activation('leaky_relu'))
#         # model.add(_layers.Dense(np.prod(self.img_shape), input_shape=(512, )))
#         # model.add(_layers.Activation('tanh'))
#         # model.add(_layers.Reshape(self.img_shape))

#         # return model

#         model = Sequential()

#         model.add(Dense(512, input_dim=self.latent_dim))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(512))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(np.prod(self.img_shape), activation='tanh'))
#         model.add(Reshape(self.img_shape))

#         model.summary()

#         z = Input(shape=(self.latent_dim,))
#         img = model(z)

#         return Model(z, img)

#     def build_discriminator(self, optimizer, loss_function):
#         # model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

#         # model.add(_layers.Dense(512, input_shape=(self.latent_dim, )))
#         # model.add(_layers.Activation('leaky_relu'))
#         # model.add(_layers.Dense(256, input_shape=(512, )))
#         # model.add(_layers.Activation('leaky_relu'))
#         # model.add(_layers.Dense(1, input_shape=(256, )))
#         # model.add(_layers.Activation('sigmoid'))

#         model = Sequential()
#         model.add(Dense(256))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dense(1, activation="sigmoid"))
#         model.summary()

#         encoded_repr = Input(shape=(self.latent_dim, ))
#         validity = model(encoded_repr)

#         return Model(encoded_repr, validity)

#     def train(self, epochs, batch_size=128, sample_interval=50):

#         # Load the dataset
#         (X_train, _), (_, _) = mnist.load_data()

#         # Rescale -1 to 1
#         X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#         X_train = np.expand_dims(X_train, axis=3)

#         # Adversarial ground truths
#         valid = np.ones((batch_size, 1))
#         fake = np.zeros((batch_size, 1))

#         for epoch in range(epochs):

#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#             idx = np.random.randint(0, X_train.shape[0], batch_size)
#             imgs = X_train[idx]

#             latent_fake = self.encoder.predict(imgs)
#             latent_real = np.random.normal(size=(batch_size, self.latent_dim))

#             # Train the discriminator
#             d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
#             d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#             # ---------------------
#             #  Train Generator
#             # ---------------------
#             g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

#             # Plot the progress
#             print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

#             # If at save interval => save generated image samples
#             if epoch % sample_interval == 0:
#                 self.sample_images(epoch)

#     def sample_images(self, epoch):
#         r, c = 5, 5

#         z = np.random.normal(size=(r*c, self.latent_dim))
#         gen_imgs = self.decoder.predict(z)

#         gen_imgs = 0.5 * gen_imgs + 0.5

#         fig, axs = plt.subplots(r, c)
#         cnt = 0
#         for i in range(r):
#             for j in range(c):
#                 axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#                 axs[i,j].axis('off')
#                 cnt += 1
#         fig.savefig(f"./sample_scratch_output/image_{epoch}.png")
#         plt.close()

# if __name__ == '__main__':
#     os.makedirs("sample_scratch_output", exist_ok=True)
#     network = AdversarialAutoencoder()
#     network.train(epochs=20000, batch_size=32, sample_interval=200)





# # import matplotlib.pyplot as plt
# # import tensorflow as tf
# # import numpy as np
# # import sys, os

# # import keras
# # from keras import layers
# # from keras.datasets import mnist
# # from extra.network.helpers import sigmoid, lrelu, tanh, img_tile, mnist_reader, relu, BCE_loss

# # # sys.path.insert(1, "D:\Programming\learn\AI\sample\ML-From-Scratch") 
# # # from mlfromscratch.deep_learning.optimizers import Adam
# # # from mlfromscratch.deep_learning.loss_functions import CrossEntropy, SquareLoss
# # # from mlfromscratch.deep_learning.layers import Dense, Conv2D, Activation, BatchNormalization, Flatten, Sampling2, Reshape, UpSampling2D
# # # from mlfromscratch.deep_learning import NeuralNetwork

# # # # data
# # # (x_train, _), (x_test, _) = mnist.load_data()
# # # x_train = x_train.astype('float32') / 255.
# # # x_test = x_test.astype('float32') / 255.
# # # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# # # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# # sys.path.insert(1, os.getcwd() + "./../../network") 
# # from layers import Dense_V2, Network_V2


# # class Autoencoder():
# #     """An Autoencoder with deep fully-connected neural nets.
# #     Training Data: MNIST Handwritten Digits (28x28 images)
# #     """
# #     def __init__(self):
# #         self.img_rows = 28
# #         self.img_cols = 28
# #         self.img_dim = self.img_rows * self.img_cols
# #         self.latent_dim = 2 # The dimension of the data embedding

# #         # # encoder
# #         # encoder_inputs = keras.Input(shape=(28, 28, 1))
# #         # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# #         # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# #         # x = layers.Flatten()(x)
# #         # x = layers.Dense(16, activation="relu")(x)

# #         # z_layers = [layers.Dense(latent_dim)(x), layers.Dense(latent_dim)(x)]
# #         # z_layers.append(Sampling()(z_layers))
# #         # self.encoder = keras.Model(encoder_inputs, z_layers)
# #         # self.encoder.summary()

# #         # # decoder
# #         # latent_inputs = keras.Input(shape=(latent_dim,))
# #         # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# #         # x = layers.Reshape((7, 7, 64))(x)
# #         # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# #         # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# #         # x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# #         # self.decoder = keras.Model(latent_inputs, x)
# #         # self.decoder.summary()
        


# #         # encoder
# #         self.encoder = Network_V2(loss_name="MSE")
# #         self.encoder.add(Dense_V2(n_units=512, input_shape=(self.img_dim,), activation="relu"))
# #         self.encoder.add(Dense_V2(n_units=256, input_shape=(512,), activation="relu"))
# #         self.encoder.add(Dense_V2(n_units=128, input_shape=(256,), activation="relu"))
# #         self.encoder.add(Dense_V2(n_units=64, input_shape=(128,), activation="relu"))
# #         self.encoder.add(Dense_V2(n_units=32, input_shape=(64,), activation="relu"))
# #         self.encoder.add(Dense_V2(n_units=16, input_shape=(32,), activation="relu"))

# #         self.sampling_1 = Dense_V2(n_units=self.latent_dim, input_shape=(16,))
# #         self.sampling_2 = Dense_V2(n_units=self.latent_dim, input_shape=(16,))

# #         # decoder
# #         self.decoder = Network_V2(loss_name="MSE")
# #         self.decoder.add(Dense_V2(n_units=16, input_shape=(self.latent_dim,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=32, input_shape=(16,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=64, input_shape=(32,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=128, input_shape=(64,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=256, input_shape=(128,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=512, input_shape=(256,), activation="relu"))
# #         self.decoder.add(Dense_V2(n_units=self.img_dim, input_shape=(512,)))

# #         # # define network
# #         # self.network = Network_V2(loss_name="MSE")
# #         # self.network.layers = np.concatenate((self.encoder.layers, self.decoder.layers))
# #         # self.network.summary(name="Autoencoder")

# #     def sample_encode(self, x):
# #         x = self.encoder._forward(x)
# #         return self.sampling_1.forward(x), self.sampling_2.forward(x)

# #     def sample_reparameterize(self, z_mean, z_log_var):
# #         batch = np.shape(z_mean)[0]
# #         dim = np.shape(z_mean)[1]
# #         rand_sample = np.random.standard_normal(size=(batch, dim))

# #         std = np.exp(0.5 * z_log_var)
# #         return z_mean + std * rand_sample

# #     def sample_encode_forward(self, x):
# #         mu, logvar = self.sample_encode(x)

# #         z = self.sample_reparameterize(mu, logvar)
# #         return z, mu, logvar
    
# #     def forward_pass(self, x):        
# #         z, mu, log_var = self.sample_encode_forward(x)
# #         return self.decoder._forward(z)

    
# #     # def learn(self, X):
# #     #     X_batch = X

# #     #     for i in range(self.iter):
# #     #         if self.batch_size > 0 and self.batch_size < X.shape[0]:
# #     #             k = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
# #     #             X_batch = X[k,:]

# #     #         X_hat = self._forwardstep(X_batch)
# #     #         grad_encoder, grad_decoder = self._backwardstep(X_batch, X_hat)

# #     #         for j in range(len(self.encoder.weights)):
# #     #             self.encoder.weights[j] -= self.encoder.alpha * grad_encoder[j]

# #     #         for j in range(len(self.decoder.weights)):
# #     #             self.decoder.weights[j] -= self.decoder.alpha * grad_decoder[j]

# #     def train_on_batch(self, X, y):
# #         z, mu, log_var = self.sample_encode_forward(X)
# #         y_pred = self.decoder._forward(z)

# #         # # Reconstruction Loss
# #         rec_loss = BCE_loss(y_pred, y)
        
# #         # print(rec_loss)

# #         # #K-L Divergence
# #         # kl = -0.5 * np.sum(1 + log_var - np.power(mu, 2) - np.exp(log_var))
        
# #         # loss = rec_loss + kl

# #         # loss = loss / len(mu)
        
# #         # # #Loss Recordkeeping
# #         # # total_loss += rec_loss / self.batch_size
# #         # # total_kl += kl / self.batch_size
# #         # # total += 1


# #         # print("loss: " + str(loss))


# #         ################################
# #         #		Backward Pass
# #         ################################
# #         # for every result in the batch
# #         # calculate gradient and update the weights using Adam
# #         # self.backward(train_batch, out)	



# #         # Calculate the gradient of the loss function wrt y_pred
# #         loss = np.mean(self.decoder.loss_function(y, y_pred)) 
# #         loss_grad = self.decoder.loss_function.gradient(y, y_pred)

# #         # encoder
# #         for layer in reversed(self.decoder.layers):
# #             loss_grad = layer.backward(loss_grad)
            
# #         # sampling
# #         # for layer in reversed(self.decoder.layers):
# #         #     loss_grad = layer.backward(loss_grad)

# #         loss_extra = self.sampling_1.backward(loss_grad)
# #         loss_grad  = self.sampling_2.backward(loss_grad)
            
# #         # decoder
# #         for layer in reversed(self.encoder.layers):
# #             loss_grad = layer.backward(loss_grad)

# #         # print("loss gradient: " + str(np.mean(loss_grad)))


# #         # Backpropagate. Update weights
# #         # self._backward(loss_grad, y_pred)

# #         return np.mean(loss_grad)

# #     def train(self, X, y, n_epochs, batch_size=128, save_interval=50):
# #         for epoch in range(n_epochs):
# #             # Select a random image
# #             image = X[np.random.randint(0, X.shape[0], batch_size)]

# #             # Train the Autoencoder
# #             loss = self.train_on_batch(image, image)

# #             # Display the progress
# #             print (f"\r[{epoch}/{n_epochs}] loss: {loss}", end="")

# #             # If at save interval => save generated image samples
# #             if epoch % save_interval == 0:
# #                 self.save_image(epoch, X)

# #     def save_image(self, epoch, X):
# #         r, c = 5, 5 # Grid size

# #         # Select a random half batch of images
# #         idx = np.random.randint(0, X.shape[0], r*c)
# #         imgs = X[idx]

# #         # Generate images and reshape to image shape
# #         gen_imgs = self.forward_pass(imgs).reshape((-1, self.img_rows, self.img_cols))

# #         # Rescale images 0 - 1
# #         gen_imgs = 0.5 * gen_imgs + 0.5

# #         fig, axs = plt.subplots(r, c)
# #         plt.suptitle("Autoencoder")
# #         cnt = 0
# #         for i in range(r):
# #             for j in range(c):
# #                 axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
# #                 axs[i,j].axis('off')
# #                 cnt += 1
# #         fig.savefig(f"./sample_scratch_output/image_{epoch}.png")
# #         plt.close()

# # # Create a sampling layer
# # class Sampling(keras.layers.Layer):
# #     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
# #     def call(self, inputs):
# #         z_mean, z_log_var = inputs


# #         print(z_mean.shape)
# #         print(z_log_var.shape)

# #         batch = tf.shape(z_mean)[0]
# #         dim = tf.shape(z_mean)[1]
# #         epsilon = keras.backend.random_normal(shape=(batch, dim))
# #         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# # class VAE(keras.Model):
# #     def __init__(self, **kwargs):
# #         super(VAE, self).__init__(**kwargs)
# #         latent_dim = 2

# #         # encoder
# #         encoder_inputs = keras.Input(shape=(28, 28, 1))
# #         x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# #         x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# #         x = layers.Flatten()(x)
# #         x = layers.Dense(16, activation="relu")(x)

# #         z_layers = [layers.Dense(latent_dim)(x), layers.Dense(latent_dim)(x)]
# #         z_layers.append(Sampling()(z_layers))
# #         self.encoder = keras.Model(encoder_inputs, z_layers)
# #         self.encoder.summary()

# #         # decoder
# #         latent_inputs = keras.Input(shape=(latent_dim,))
# #         x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# #         x = layers.Reshape((7, 7, 64))(x)
# #         x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# #         x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# #         x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# #         self.decoder = keras.Model(latent_inputs, x)
# #         self.decoder.summary()


# #         self.total_loss_tracker = keras.metrics.Mean()

# #     def train_step(self, data):
# #         with tf.GradientTape() as tape:
# #             z_mean, z_log_var, z = self.encoder(data)
# #             reconstruction = self.decoder(z)
# #             reconstruction_loss = tf.reduce_mean(
# #                 tf.reduce_sum( keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2) )
# #             )
# #             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
# #             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
# #             total_loss = reconstruction_loss + kl_loss

# #         grads = tape.gradient(total_loss, self.trainable_weights)
# #         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
# #         self.total_loss_tracker.update_state(total_loss)

# #         return { "loss": self.total_loss_tracker.result() }



# # def make_dir():
# #     image_dir = "./sample_scratch_output"
# #     if not os.path.exists(image_dir):
# #         os.makedirs(image_dir)

# # make_dir()
# # ae = Autoencoder()
# # ae.train(X=x_train, y=x_train, n_epochs=200000, batch_size=64, save_interval=400)



# # # data
# # (X_train, y_train), (X_test, _) = mnist.load_data()

# # # Rescale -1 to 1
# # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# # X_train = np.expand_dims(X_train, axis=3)

# # # x_data = np.concatenate([x_train, x_test], axis=0)
# # # x_data = np.expand_dims(x_data, -1).astype("float32") / 255
# # # x_train = np.expand_dims(x_train, -1).astype("float32") / 255





# # # settings
# # n_epochs = 10
# # batch_size = 128


# # # def make_dir():
# # #     image_dir = "./sample_scratch_output"
# # #     if not os.path.exists(image_dir):
# # #         os.makedirs(image_dir)

# # # define network
# # network = VAE()
# # network.compile(optimizer=keras.optimizers.Adam())

# # # # define network OKIDO
# # # make_dir()
# # # ae = VAEncoder()


# # # trainings
# # network.fit(x_data, epochs=n_epochs, batch_size=batch_size)
# # # ae.train(X=x_train2, y=x_train2, n_epochs=n_epochs, batch_size=batch_size, save_interval=400)


# # # # above OK





# # def plot_latent_space(network, n=30, figsize=15):
# #     # Display a n*n 2D manifold of digits
# #     digit_size = 28
# #     scale = 1.0
# #     figure = np.zeros((digit_size * n, digit_size * n))

# #     # linearly spaced coordinates corresponding to the 2D plot 
# #     # of digit classes in the latent space
# #     grid_x = np.linspace(-scale, scale, n)
# #     grid_y = np.linspace(-scale, scale, n)[::-1]

# #     for i, yi in enumerate(grid_y):
# #         for j, xi in enumerate(grid_x):
# #             z_sample = np.array([[xi, yi]])
# #             x_decoded = network.decoder.predict(z_sample)
# #             digit = x_decoded[0].reshape(digit_size, digit_size)
# #             figure[
# #                 i * digit_size : (i + 1) * digit_size,
# #                 j * digit_size : (j + 1) * digit_size
# #             ] = digit

# #     plt.figure(figsize=(figsize, figsize))
# #     start_range = digit_size // 2
# #     end_range = n * digit_size + start_range
# #     pixel_range = np.arange(start_range, end_range, digit_size)
# #     sample_range_x = np.round(grid_x, 1)
# #     sample_range_y = np.round(grid_y, 1)

# #     plt.xticks(pixel_range, sample_range_x)
# #     plt.yticks(pixel_range, sample_range_y)
# #     plt.xlabel("z[0]")
# #     plt.ylabel("z[1]")
# #     plt.imshow(figure, cmap="Greys_r")
# #     plt.show()

# # def plot_label_clusters(network, data, labels):
# #     # display a 2D plot of the digit classes in the latent space
# #     z_mean, _, _ = network.encoder.predict(data)
# #     plt.figure(figsize=(12, 10))
# #     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
# #     plt.colorbar()
# #     plt.xlabel("z[0]")
# #     plt.ylabel("z[1]")
# #     plt.show()

# # plot_label_clusters(network, x_train, y_train)
# # plot_latent_space(network)






# # # ##############################################################################################################
# #     # class VAEncoder():
# #     #     """An Autoencoder with deep fully-connected neural nets.
# #     #     Training Data: MNIST Handwritten Digits (28x28 images)
# #     #     """
# #     #     def __init__(self):
# #     #         self.img_rows = 28
# #     #         self.img_cols = 28
# #     #         self.img_dim = self.img_rows * self.img_cols
# #     #         self.latent_dim = 2 # The dimension of the data embedding

# #     #         optimizer = Adam(learning_rate=0.0002, b1=0.5)
# #     #         loss_function = SquareLoss
            
# #     #         # ENCODER
# #     #         self.encoder = self.build_encoder(optimizer, loss_function)
# #     #         print(self.encoder.summary())

# #     #         # SAMPLER
# #     #         self.sampling_0 = Dense(self.latent_dim, input_shape=(16))
# #     #         self.sampling_1 = Dense(self.latent_dim, input_shape=(16))

# #     #         # DECODER
# #     #         self.decoder = self.build_decoder(optimizer, loss_function)
# #     #         print(self.decoder.summary())

# #     #         # self.autoencoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
# #     #             # self.autoencoder.layers.extend(self.encoder.layers)
# #     #             # self.autoencoder.layers.extend(self.decoder.layers)

# #     #             # self.autoencoder.summary(name="Variational Autoencoder")

# #     #             # latent_dim = 2

# #     #             # # encoder
# #     #             # encoder_inputs = keras.Input(shape=(28, 28, 1))
# #     #             # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# #     #             # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# #     #             # x = layers.Flatten()(x)
# #     #             # x = layers.Dense(16, activation="relu")(x)

# #     #             # z_layers = [layers.Dense(latent_dim)(x), layers.Dense(latent_dim)(x)]
# #     #             # z_layers.append(Sampling()(z_layers))
# #     #             # self.encoder = keras.Model(encoder_inputs, z_layers)
# #     #             # self.encoder.summary()

# #     #             # # decoder
# #     #             # latent_inputs = keras.Input(shape=(latent_dim,))
# #     #             # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# #     #             # x = layers.Reshape((7, 7, 64))(x)
# #     #             # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# #     #             # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# #     #             # x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# #     #             # self.decoder = keras.Model(latent_inputs, x)
# #     #             # self.decoder.summary()
            
# #     #     def sample_encode(self, x):
# #     #         x = self.encoder.forward(x)
# #     #         return self.sampling_0.forward(x), self.sampling_1.forward(x)

# #     #     def sample_reparameterize(self, z_mean, z_log_var):
# #     #         std = np.exp(0.5 * z_log_var)
# #     #         epsilon = np.randn_like(std)
# #     #         return z_mean + std * epsilon

# #     #     def sample_forward(self, x):
# #     #         mu, logvar = self.sample_encode(x.view(-1, 784))
# #     #         z = self.sample_reparameterize(mu, logvar)
# #     #         return z, mu, logvar

# #     #     def build_encoder(self, optimizer, loss_function, latent_dim=2):
# #     #         encoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
# #     #         encoder.add(Conv2D(n_filters=32, filter_shape=(3,3), input_shape=(1, 28, 28), stride=2, padding='same'))
# #     #         encoder.add(Activation('relu'))
# #     #         encoder.add(Conv2D(n_filters=64, filter_shape=(3,3), input_shape=(32, 14, 14), stride=2, padding='same'))
# #     #         encoder.add(Activation('relu'))
# #     #         encoder.add(Flatten(input_shape=(64, 7, 7)))
# #     #         encoder.add(Dense(16))
# #     #         encoder.add(Activation('relu'))

# #     #         return encoder

# #     #     def build_decoder(self, optimizer, loss_function, latent_dim=2):
# #     #         decoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
# #     #         decoder.add(Dense((7 * 7 * 64), input_shape=(latent_dim,)))
# #     #         decoder.add(Activation('relu'))
# #     #         decoder.add(Reshape((64, 7, 7), input_shape=((7 * 7 * 64),)))

# #     #         # decoder.add(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
# #     #         # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# #     #         # x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# #     #         decoder.add(UpSampling2D())
# #     #         decoder.add(Conv2D(n_filters=64, filter_shape=(3,3), input_shape=(64, 14, 14), stride=2, padding='same'))
# #     #         decoder.add(Activation('relu'))
# #     #         decoder.add(UpSampling2D())

# #     #         decoder.add(UpSampling2D())
# #     #         decoder.add(Conv2D(n_filters=32, filter_shape=(3,3), input_shape=(64, 28, 28), stride=2, padding='same'))
# #     #         decoder.add(Activation('relu'))

# #     #         decoder.add(UpSampling2D())
# #     #         decoder.add(Conv2D(n_filters=1, filter_shape=(3,3), input_shape=(32, 28, 28), padding='same'))
# #     #         decoder.add(Activation('sigmoid'))

# #     #         return decoder




# #     #     def train(self, X, y, n_epochs, batch_size=128, save_interval=50):
# #     #         for epoch in range(n_epochs):
# #     #             # Select a random image
# #     #             image = X[np.random.randint(0, X.shape[0], batch_size)]

# #     #             print(image.shape)

# #     #             # Train the Autoencoder
# #     #             loss, _ = self.network.train_on_batch(image, image)

# #     #             # Display the progress
# #     #             print (f"\r[{epoch}/{n_epochs}] loss: {loss}", end="")

# #     #             # If at save interval => save generated image samples
# #     #             if epoch % save_interval == 0:
# #     #                 self.save_image(epoch, X)

# #     #     def save_image(self, epoch, X):
# #     #         r, c = 5, 5 # Grid size

# #     #         # Select a random half batch of images
# #     #         idx = np.random.randint(0, X.shape[0], r*c)
# #     #         imgs = X[idx]

# #     #         # Generate images and reshape to image shape
# #     #         gen_imgs = self.network.predict(imgs).reshape((-1, self.img_rows, self.img_cols))

# #     #         # Rescale images 0 - 1
# #     #         gen_imgs = 0.5 * gen_imgs + 0.5

# #     #         fig, axs = plt.subplots(r, c)
# #     #         plt.suptitle("Autoencoder")
# #     #         cnt = 0
# #     #         for i in range(r):
# #     #             for j in range(c):
# #     #                 axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
# #     #                 axs[i,j].axis('off')
# #     #                 cnt += 1
# #     #         fig.savefig("./sample_scratch_output/ae_%d.png" % epoch)
# #     #         plt.close()
# # # ##############################################################################################################




# # # # import pickle, gzip
# # # # import matplotlib.pyplot as plt 
# # # # import numpy as np
# # # # import sys

# # # # np.random.seed(0)


# # # # from vae import VAE


# # # # from keras.datasets import mnist

# # # # # data
# # # # (x_train, y_train), (x_test, _) = mnist.load_data()
# # # # x_data = np.concatenate([x_train, x_test], axis=0)
# # # # x_data = np.expand_dims(x_data, -1).astype("float32") / 255
# # # # x_train = np.expand_dims(x_train, -1).astype("float32") / 255


# # # # with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
# # # #     train, test, val = pickle.load(f, encoding='latin1')
# # # #     mnist_train = train[0]
# # # #     mnist_test = test[0]



# # # #     params = {
# # # #         'alpha' : 0.02,
# # # #         'iter' : 20000,
# # # #         'activation': 'sigmoid',
# # # #         'loss': 'squared_error',
# # # #         'batch_size': 150
# # # #     }

# # # #     example = VAE([[784, 200], [200, 784]], 2, params)
# # # #     example.learn(mnist_train, x_train)

# # # #     # print(np.array(train[0]).shape)
# # # #     # print(x_train.shape)
# # # #     print("FINISHED")
# # # #     pass


# # # #     fig, ax = plt.subplots(2,3, figsize = (10, 8))

# # # #     for i in range(3):
# # # #         in_digit = mnist_test[i][None,:]
# # # #         out_digit = example.encode_decode(in_digit)
# # # #         ax[0,i].matshow(in_digit.reshape((28,28)),  cmap='gray', clim=(0,1))
# # # #         ax[1,i].matshow(out_digit.reshape((28,28)), cmap='gray', clim=(0,1))
# # # #     # pass


# # # #     plt.show()



# # # #     fig, ax = plt.subplots(2,2, figsize = (6, 6))

# # # #     a = np.array([1, 3])
# # # #     b = np.array([1, 3])

# # # #     for i, z1 in enumerate(a):
# # # #         for j, z2 in enumerate(b):
# # # #             ax[i,j].matshow(example.generate(np.array([z1,z2])).reshape((28,28)),  cmap='gray', clim=(0,1))
# # # #     # pass


# # # #     plt.show()



# # # # # import argparse
# # # # # import numpy as np
# # # # # from extra.network.helpers import sigmoid, lrelu, tanh, img_tile, mnist_reader, relu, BCE_loss

# # # # # import os, sys
# # # # # np.random.seed(0)
# # # # # cpu_enabled = 0

# # # # # sys.path.insert(1, os.getcwd() + "./../../network") 
# # # # # from layers import *
# # # # # from algorithms.activation_functions import act_functions 
# # # # # from algorithms.loss_functions import loss_functions 
# # # # # from algorithms.optimizer_functions import opt_functions


# # # # # class Hidden:
# # # # #     def __init__(self, input_shape, weights, bias, activation="relu"):
# # # # #         self.input_shape = input_shape
# # # # #         self.activation = act_functions[activation]()

# # # # #         # Xavier initialization
# # # # #         limit = 1 / np.sqrt(self.input_shape[0])
# # # # #         self.W = weights
# # # # #         self.b = bias

# # # # #     def forward(self, X):
# # # # #         self.wsum = X.dot(self.W) + self.b
# # # # #         return self.activation(self.wsum)
        
# # # # # class VAE():
# # # # #     def __init__(self):
# # # # #         self.numbers = [1, 2, 3]
# # # # #         self.epochs = 40
# # # # #         self.batch_size = 64
# # # # #         self.learning_rate = 0.0001
# # # # #         self.decay = 0.001
# # # # #         self.nz = 20
# # # # #         self.layersize = 400

# # # # #         # structure = [
# # # # #         #     # layers.Dense(n_units=self.layersize, input_shape=(len(self.e_input), len(self.e_input[0])), activation="relu"),
# # # # #         #     # layers.Dense(n_units=self.layersize, input_shape=(784, self.layersize), activation="relu"),
# # # # #         # ]
# # # # #         self.network = Network2()
# # # # #         self.dense_0 = Dense(n_units=784, input_shape=(784, self.layersize), activation="relu")


# # # # #         self.img_path = "./extra/images"
# # # # #         if not os.path.exists(self.img_path):
# # # # #             os.makedirs(self.img_path)
        
# # # # #         # Xavier initialization is used to initialize the weights
# # # # #         # init encoder weights
# # # # #         self.e_W0 = np.random.randn(784, self.layersize).astype(np.float32) * np.sqrt(2.0/(784))
# # # # #         self.e_b0 = np.zeros(self.layersize).astype(np.float32)

# # # # #         self.e_W_mu = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
# # # # #         self.e_b_mu = np.zeros(self.nz).astype(np.float32)
        
# # # # #         self.e_W_logvar = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
# # # # #         self.e_b_logvar = np.zeros(self.nz).astype(np.float32)

# # # # #         # init decoder weights 
# # # # #         self.d_W0 = np.random.randn(self.nz, self.layersize).astype(np.float32) * np.sqrt(2.0/(self.nz))
# # # # #         self.d_b0 = np.zeros(self.layersize).astype(np.float32)
        
# # # # #         self.d_W1 = np.random.randn(self.layersize, 784).astype(np.float32) * np.sqrt(2.0/(self.layersize))
# # # # #         self.d_b1 = np.zeros(784).astype(np.float32)
             
# # # # #         # init sample
# # # # #         self.sample_z = 0
# # # # #         self.rand_sample = 0
        
# # # # #         # init Adam optimizer
# # # # #         self.b1 = 0.9
# # # # #         self.b2 = 0.999
# # # # #         self.e = 1e-8
# # # # #         self.m = [0] * 10
# # # # #         self.v = [0] * 10
# # # # #         self.t = 0
        

        
# # # # #     def encoder(self, img):
# # # # #         #self.e_logvar : log variance 
# # # # #         #self.e_mean : mean

# # # # #         self.e_input = np.reshape(img, (self.batch_size,-1))

# # # # #         # print(self.batch_size.shape)
# # # # #         # print(self.img.shape)

# # # # #         # self.e_h0 = layers.Dense(
# # # # #         #     n_units=self.layersize, 
# # # # #         #     input_shape=(len(self.e_input), len(self.e_input[0])), 
# # # # #         #     activation="relu"
# # # # #         # ),


    
# # # # #         self.e_h0 = Hidden(
# # # # #             input_shape=(len(self.e_input), len(self.e_input[0])),
# # # # #             weights=self.e_W0,
# # # # #             bias=self.e_b0,
# # # # #             activation="relu"
# # # # #         )

# # # # #         self.e_h0_a = self.e_h0.forward(self.e_input)
# # # # #         self.e_h0_a = result = self.dense_0.forward(self.e_input)

# # # # #         print(self.e_input.shape)
# # # # #         print(self.e_h0_a.shape)
# # # # #         print(result.shape)


# # # # #         self.e_h0_l = self.e_h0.wsum

# # # # #         # self.e_h0_l = self.e_input.dot(self.e_W0) + self.e_b0
# # # # #         # self.e_h0_a = lrelu(self.e_h0_l)

# # # # #         self.e_logvar = self.e_h0_a.dot(self.e_W_logvar) + self.e_b_logvar
# # # # #         self.e_mu = self.e_h0_a.dot(self.e_W_mu) + self.e_b_mu
    
# # # # #         return self.e_mu, self.e_logvar
    
# # # # #     def decoder(self, z):
# # # # #         #self.d_out : reconstruction image 28x28
		
# # # # #         self.z = np.reshape(z, (self.batch_size, self.nz))
        
        
# # # # #         self.d_h0 = Hidden(
# # # # #             input_shape=(len(self.z), len(self.z[0])),
# # # # #             weights=self.d_W0,
# # # # #             bias=self.d_b0,
# # # # #             activation="relu"
# # # # #         )
# # # # #         self.d_h0_a = self.d_h0.forward(self.z)
# # # # #         self.d_h0_l = self.d_h0.wsum

# # # # #         # self.d_h0_l = self.z.dot(self.d_W0) + self.d_b0		
# # # # #         # self.d_h0_a = relu(self.d_h0_l)

        
# # # # #         self.d_h1 = Hidden(
# # # # #             input_shape=(len(self.d_h0_a), len(self.d_h0_a[0])),
# # # # #             weights=self.d_W1,
# # # # #             bias=self.d_b1,
# # # # #             activation="sigmoid"
# # # # #         )
# # # # #         self.d_h1_a = self.d_h1.forward(self.d_h0_a)
# # # # #         self.d_h1_l = self.d_h1.wsum

# # # # #         # self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
# # # # #         # self.d_h1_a = sigmoid(self.d_h1_l)

# # # # #         self.d_out = np.reshape(self.d_h1_a, (self.batch_size, 28, 28, 1))

# # # # #         return self.d_out

# # # # #     def forward(self, x):
# # # # #         #Encode
# # # # #         mu, logvar = self.encoder(x)
        
# # # # #         #use reparameterization trick to sample from gaussian
# # # # #         self.rand_sample = np.random.standard_normal(size=(self.batch_size, self.nz))
# # # # #         self.sample_z = mu + np.exp(logvar * .5) * np.random.standard_normal(size=(self.batch_size, self.nz))
        
# # # # #         decode = self.decoder(self.sample_z)
        
# # # # #         return decode, mu, logvar
    
# # # # #     def backward(self, x, out):
# # # # #         ########################################
# # # # #         #Calculate gradients from reconstruction
# # # # #         ########################################
# # # # #         y = np.reshape(x, (self.batch_size, -1))
# # # # #         out = np.reshape(out, (self.batch_size, -1))
        
# # # # #         #Calculate decoder gradients
# # # # #         #Left side term
# # # # #         dL_l = -y * (1 / out)
# # # # #         dsig = sigmoid(self.d_h1_l, derivative=True)
# # # # #         dL_dsig_l = dL_l * dsig
        
# # # # #         drelu = relu(self.d_h0_l, derivative=True)

# # # # #         dW1_d_l = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(dL_dsig_l, axis=1))
# # # # #         db1_d_l = dL_dsig_l 
        
# # # # #         db0_d_l = dL_dsig_l.dot(self.d_W1.T) * drelu
# # # # #         dW0_d_l = np.matmul(np.expand_dims(self.sample_z, axis=-1), np.expand_dims(db0_d_l, axis=1))
        
# # # # #         #Right side term
# # # # #         dL_r = (1 - y) * (1 / (1 - out))
# # # # #         dL_dsig_r = dL_r * dsig
        
# # # # #         dW1_d_r = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(dL_dsig_r, axis=1))
# # # # #         db1_d_r = dL_dsig_r
        
# # # # #         db0_d_r = dL_dsig_r.dot(self.d_W1.T) * drelu
# # # # #         dW0_d_r = np.matmul(np.expand_dims(self.sample_z, axis=-1), np.expand_dims(db0_d_r, axis=1))
        
# # # # #         # Combine gradients for decoder
# # # # #         grad_d_W0 = dW0_d_l + dW0_d_r
# # # # #         grad_d_b0 = db0_d_l + db0_d_r
# # # # #         grad_d_W1 = dW1_d_l + dW1_d_r
# # # # #         grad_d_b1 = db1_d_l + db1_d_r
         
# # # # #         #Calculate encoder gradients from reconstruction
# # # # #         #Left side term
# # # # #         d_b_mu_l  = db0_d_l.dot(self.d_W0.T)
# # # # #         d_W_mu_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_l, axis=1))
        
# # # # #         db0_e_l = d_b_mu_l.dot(self.e_W_mu.T) * lrelu(self.e_h0_l, derivative=True)
# # # # #         dW0_e_l = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_l, axis=1)) 
        
# # # # #         d_b_logvar_l = d_b_mu_l * np.exp(self.e_logvar * .5) * .5 * self.rand_sample
# # # # #         d_W_logvar_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_l, axis=1))
        
# # # # #         db0_e_l_2 = d_b_logvar_l.dot(self.e_W_logvar.T) * lrelu(self.e_h0_l, derivative=True)
# # # # #         dW0_e_l_2 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_l_2, axis=1)) 
        
# # # # #         #Right side term
# # # # #         d_b_mu_r  = db0_d_r.dot(self.d_W0.T)
# # # # #         d_W_mu_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_r, axis=1))
        
# # # # #         db0_e_r = d_b_mu_r.dot(self.e_W_mu.T) * lrelu(self.e_h0_l, derivative=True)
# # # # #         dW0_e_r = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_r, axis=1)) 
        
# # # # #         d_b_logvar_r = d_b_mu_r * np.exp(self.e_logvar * .5) * .5 * self.rand_sample
# # # # #         d_W_logvar_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_r, axis=1))
        
# # # # #         db0_e_r_2 = d_b_logvar_r.dot(self.e_W_logvar.T) * lrelu(self.e_h0_l, derivative=True)
# # # # #         dW0_e_r_2 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0_e_r_2, axis=1))
        
# # # # #         ########################################
# # # # #         #Calculate encoder gradients from K-L
# # # # #         ########################################
    
# # # # #         #logvar terms
# # # # #         dKL_b_log = -.5 * (1 - np.exp(self.e_logvar))
# # # # #         dKL_W_log = np.matmul(np.expand_dims(self.e_h0_a, axis= -1), np.expand_dims(dKL_b_log, axis= 1))
        
# # # # #         #Heaviside step function
# # # # #         dlrelu = lrelu(self.e_h0_l, derivative=True)  

# # # # #         dKL_e_b0_1 = .5 * dlrelu * (np.exp(self.e_logvar) - 1).dot(self.e_W_logvar.T)
# # # # #         dKL_e_W0_1 = np.matmul(np.expand_dims(y, axis= -1), np.expand_dims(dKL_e_b0_1, axis= 1))
        
# # # # #         #m^2 term
# # # # #         dKL_W_m = .5 * (2 * np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(self.e_mu, axis=1)))
# # # # #         dKL_b_m = .5 * (2 * self.e_mu)
        
# # # # #         dKL_e_b0_2 = .5 * dlrelu * (2 * self.e_mu).dot(self.e_W_mu.T)
# # # # #         dKL_e_W0_2 = np.matmul(np.expand_dims(y, axis= -1), np.expand_dims(dKL_e_b0_2, axis= 1))
        
# # # # #         # Combine gradients for encoder from recon and KL
# # # # #         grad_b_logvar = dKL_b_log + d_b_logvar_l + d_b_logvar_r
# # # # #         grad_W_logvar = dKL_W_log + d_W_logvar_l + d_W_logvar_r
# # # # #         grad_b_mu = dKL_b_m + d_b_mu_l + d_b_mu_r
# # # # #         grad_W_mu = dKL_W_m + d_W_mu_l + d_W_mu_r
# # # # #         grad_e_b0 = dKL_e_b0_1 + dKL_e_b0_2 + db0_e_l + db0_e_l_2 + db0_e_r + db0_e_r_2
# # # # #         grad_e_W0 = dKL_e_W0_1 + dKL_e_W0_2 + dW0_e_l + dW0_e_l_2 + dW0_e_r + dW0_e_r_2
        
        
# # # # #         grad_list = [grad_e_W0, grad_e_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar,
# # # # #                      grad_d_W0, grad_d_b0, grad_d_W1, grad_d_b1]
        
# # # # #         ########################################
# # # # #         #Calculate update using Adam
# # # # #         ########################################
# # # # #         self.t += 1
# # # # #         for i, grad in enumerate(grad_list):
# # # # #             self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
# # # # #             self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.power(grad, 2)
# # # # #             m_h = self.m[i] / (1 - (self.b1 ** self.t))
# # # # #             v_h = self.v[i] / (1 - (self.b2 ** self.t))
# # # # #             grad_list[i] = m_h / (np.sqrt(v_h) + self.e)
        
# # # # #         # Update all weights
# # # # #         for idx in range(self.batch_size):
# # # # #             # Encoder Weights
# # # # #             self.e_W0 = self.e_W0 - self.learning_rate*grad_list[0][idx]
# # # # #             self.e_b0 = self.e_b0 - self.learning_rate*grad_list[1][idx]
    
# # # # #             self.e_W_mu = self.e_W_mu - self.learning_rate*grad_list[2][idx]
# # # # #             self.e_b_mu = self.e_b_mu - self.learning_rate*grad_list[3][idx]
            
# # # # #             self.e_W_logvar = self.e_W_logvar - self.learning_rate*grad_list[4][idx]
# # # # #             self.e_b_logvar = self.e_b_logvar - self.learning_rate*grad_list[5][idx]
    
# # # # #             # Decoder Weights
# # # # #             self.d_W0 = self.d_W0 - self.learning_rate*grad_list[6][idx]
# # # # #             self.d_b0 = self.d_b0 - self.learning_rate*grad_list[7][idx]
            
# # # # #             self.d_W1 = self.d_W1 - self.learning_rate*grad_list[8][idx]
# # # # #             self.d_b1 = self.d_b1 - self.learning_rate*grad_list[9][idx]
    
# # # # #     def train(self):
        
# # # # #         #Read in training data
# # # # #         trainX, _, train_size = mnist_reader(self.numbers)
        
# # # # #         np.random.shuffle(trainX)
        
# # # # #         #set batch indices
# # # # #         batch_idx = train_size//self.batch_size
        
# # # # #         total_loss = 0
# # # # #         total_kl = 0
# # # # #         total = 0
        
# # # # #         for epoch in range(self.epochs):
# # # # #             for idx in range(batch_idx):
# # # # #                 # prepare batch and input vector z
# # # # #                 train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
# # # # #                 #ignore batch if there are insufficient elements 
# # # # #                 if train_batch.shape[0] != self.batch_size:
# # # # #                     break
                
# # # # #                 ################################
# # # # #                 #		Forward Pass
# # # # #                 ################################
                
# # # # #                 out, mu, logvar = self.forward(train_batch)
                
# # # # #                 # Reconstruction Loss
# # # # #                 rec_loss = BCE_loss(out, train_batch)
                
# # # # #                 #K-L Divergence
# # # # #                 kl = -0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar))
                
# # # # #                 loss = rec_loss + kl
# # # # #                 loss = loss / self.batch_size
                
# # # # #                 #Loss Recordkeeping
# # # # #                 total_loss += rec_loss / self.batch_size
# # # # #                 total_kl += kl / self.batch_size
# # # # #                 total += 1

# # # # #                 ################################
# # # # #                 #		Backward Pass
# # # # #                 ################################
# # # # #                 # for every result in the batch
# # # # #                 # calculate gradient and update the weights using Adam
# # # # #                 self.backward(train_batch, out)	

# # # # #                 self.img = np.squeeze(out, axis=3) * 2 - 1

# # # # #                 print("Epoch [%d] Step [%d]  RC Loss:%.4f  KL Loss:%.4f  lr: %.4f"%(
# # # # #                         epoch, idx, rec_loss / self.batch_size, kl / self.batch_size, self.learning_rate))
                
# # # # #             # if cpu_enabled == 1:
# # # # #             sample = np.array(self.img)
# # # # #             # else: 
# # # # #                 # sample = np.asnumpy(self.img)
            
# # # # #             #save image result every epoch
# # # # #             img_tile(sample, self.img_path, epoch, idx, "res", True)


# # # # # if __name__ == '__main__':

# # # # #     # Adjust the numbers that appear in the training data. Less numbers helps 
# # # # #     # run the program to see faster results
# # # # #     model = VAE()
# # # # #     model.train()
    