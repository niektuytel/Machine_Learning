from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def get_encoder(input_layer, encoded):
    # This model maps an input to its encoded representation
    return Model(input_layer, encoded)

def get_decoder(hidden_size, decoded):
    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(hidden_size,))
    
    # Create the decoder model
    decoder = Model(encoded_input, decoded(encoded_input))  

    return decoder

def get_network(input_size, hidden_size):
    # This is our input image
    input_layer = Input(shape=(input_size,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(hidden_size, activation='relu')(input_layer)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_size, activation='sigmoid')(encoded)

    # define neural network
    network = Model(input_layer, decoded)
    network.compile(optimizer='adam', loss='binary_crossentropy')

    # This model maps an input to its encoded representation
    encoder = get_encoder(input_layer, encoded)
    decoder = get_decoder(hidden_size, network.layers[-1])
    return network, encoder, decoder

if __name__ == "__main__":
    # get structure mnist image dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # nomalize data between 0 and 1 
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype('float32')  / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # parmeters
    input_size = 784
    hidden_size = 64

    # define auto encoder model with encoder and decoder
    autoencoder, encoder, decoder = get_network(input_size, hidden_size)

    # train model
    autoencoder.fit(x_train, x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # Display result of original and decoded data
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gray()

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gray()

    plt.show()

