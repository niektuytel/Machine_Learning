




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





