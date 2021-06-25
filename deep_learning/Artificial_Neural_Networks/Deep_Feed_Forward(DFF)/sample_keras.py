import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# data
df = pd.read_csv("./data/housepricedata.csv")
dataset = df.values

X = dataset[:, 0:10]
Y = dataset[:, 10]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, _, Y_train, _ = train_test_split(X_scale, Y, test_size=0.3)

# model
model = Sequential([
    Dense(32, activation="relu", input_shape=(10,)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=1000, epochs=1000, verbose=1)

# predict
unknown_input = [[8050,3,2,350,0,1,3,5,0,390]]
result = model.predict(unknown_input)
print("\n\n")
print("the prediction of the neural network: " + str(result))
print("\n\n")