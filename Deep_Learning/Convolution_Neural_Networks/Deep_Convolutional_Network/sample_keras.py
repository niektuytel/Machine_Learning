import matplotlib.pyplot as plt
from keras.datasets import mnist
# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten



# download mnist data split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot the first image iin the dataset
plt.imshow(X_train[0])

# print(X_train[0])
# print(X_train[0].shape)

# reshape data to fit model (pre-processing)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train[0])
# print(X_[0])


# Create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

# compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
print(model.predict(X_test[:4]))
print(y_test[:4])



