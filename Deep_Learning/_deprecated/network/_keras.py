# https://keras.io/examples/generative/lstm_character_level_text_generation/
import numpy as np
import random, io, sys, os, keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

sys.path.insert(1, os.getcwd() + "/../") 
import data

# get data
x, y = data.vectorization()

# build the model: a single LSTM
model = keras.Sequential()
model.add(LSTM(128, input_shape=(data.seq_length, len(data.chars))))
model.add(Dense(len(data.chars), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# parameters
epochs = 40
batch_size = 128

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train + predict trained network
for epoch in range(epochs):
    print("-" * 120)
    model.fit(x, y, batch_size=batch_size, epochs=1)

    # define test data
    start_index = random.randint(0, len(data.text) - data.seq_length - 1)
    sentence = data.text[start_index : start_index + data.seq_length]

    print(f'[{epoch}]: seed = "{sentence}"')
    for temparature in [0.2, 0.5, 1.0, 1.2]:
        prediction = ""

        for i in range(100):
            x_pred = np.zeros((1, data.seq_length, len(data.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, data.char2idx[char]] = 1.0

            # prediction by network
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temparature)
            next_char = data.idx2char[next_index]

            # set result + prepare for net iteration
            sentence = sentence[1:] + next_char
            prediction += next_char

        print(f'on temperature: "{temparature}" the prediction = "{prediction}"')
    print()


# Result:

# ------------------------------------------------------------------------------------------------------------------------
# 112/112 [==============================] - 17s 142ms/step - loss: 3.3002
# [0]: seed = " messages. I haven't seen anything like that since then. All I hear right now is Baby Shark being bl"
# on temperature: "0.2" the prediction = "a the the the the the the in the thit in the ton the the the the the the thin the the the the the I "
# on temperature: "0.5" the prediction = "in it  il pous anhit ancthatmer it inlis thet are s wire, bort. I I was as I I stan afs bond thond i"
# on temperature: "1.0" the prediction = "nohemee hr kut tor. doolywo fas s , km nas no Tir Lnu. ouce I wivurdar the bur thuve tadid foo imy. "
# on temperature: "1.2" the prediction = "nuve iogg, tothory0 vadonchu n silm Lfofr,tpeld dukd'veonlaad. aes Hut bLlc, ryltot Tas Bave Rxiw0hi"

# .......

# ------------------------------------------------------------------------------------------------------------------------
# 112/112 [==============================] - 16s 147ms/step - loss: 0.5112
# [39]: seed = "sense that this LTE isn’t very long yet. Whenever I look at this webpage, it looks long at first gla"
# on temperature: "0.2" the prediction = "ils because I saintle in hape you take to the only make inters to gonna dispes. If how "nomberacters"
# on temperature: "0.5" the prediction = ") word it was ago and make on this LTE is back or songs to say distand in the eadn the olly still be"
# on temperature: "1.0" the prediction = "ing bike you we lifccal. But it is 5. Oh out forgen word and disple lake the whenever reason’t think"
# on temperature: "1.2" the prediction = " and I gle endire alo have pon't know what this det makeviog have song doesiag the pame is pretsa so"