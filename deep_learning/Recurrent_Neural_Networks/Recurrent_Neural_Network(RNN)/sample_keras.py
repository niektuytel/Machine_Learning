import numpy as np
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# generate data
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=30000, maxlen=50, test_split=0.3)
X_train = pad_sequences(X_train, padding = 'post')
X_test = pad_sequences(X_test, padding = 'post')
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

y_data = to_categorical(np.concatenate((y_train, y_test)))
y_train, y_test = y_data[:1395], y_data[1395:]

# define network
def network():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape = (49,1), return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model

# train network
model = KerasClassifier(build_fn = network, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)

# network result
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(f"Accuracy: {accuracy_score(y_pred, y_test_)}")