import numpy as np
import random, io, sys, os

sys.path.insert(1, os.getcwd() + "./../../network") 
import layers

n_epochs = 40
batch_size = 128
sequences_step = 5
seq_length = 100

# generate data
def longest_text():
    return """Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really """
def generate_dataset():
    # cut the text in semi-redundant sequences of seq_length characters
    sentences  = []
    next_chars = []
    for i in range(0, len(longest_text()) - seq_length, sequences_step):
        sentences.append(longest_text()[i : i+seq_length])
        next_chars.append(longest_text()[    i+seq_length])

    n_sentences = len(sentences)
    n_chars = len(chars)
    # print("Sentences size: ", n_sentences)

    x = np.zeros((n_sentences, seq_length, n_chars), dtype=np.int)
    y = np.zeros((n_sentences, seq_length, n_chars), dtype=np.int)
    # y = np.zeros((n_sentences, n_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char2idx[char]] = 1

            if(t > 0):
                y[i, t-1, char2idx[char]] = 1
        y[i, -1, char2idx[next_chars[i]]] = 1
    return x, y
chars = sorted(list(set(longest_text())))
char2idx = {w: i for i,w in enumerate(chars)}
idx2char = {i: w for i,w in enumerate(chars)}
x, y = generate_dataset()

# helper function to sample an index from a probability array
def predict_index(preds, temperature=1.0):
    preds = np.exp(np.log(np.asarray(preds).astype("float64")) / temperature)
    return np.argmax(np.random.multinomial(1, preds / np.sum(preds), 1))
    
# define network
network2 = layers.Network2(loss="CrossEntropy", activation_output="softmax")
network2.add(
# layers.GRU(n_units=128, input_shape=(seq_length, len(chars)), activation="tanh", activation_output="softmax", optimizer="adam")
layers.LSTM(n_units=seq_length, input_shape=(seq_length, len(chars)), activation="tanh", activation_output="softmax", optimizer="adam")
)

print(x.shape)
print(y.shape)
network2.fit(x, y, batch_size=seq_length, n_epochs=n_epochs)


# # define network
# network = keras.Sequential()
# network.add(LSTM(128, input_shape=(seq_length, len(chars))))
# network.add(Dense(len(chars), activation="softmax"))
# network.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# # train network
# for epoch in range(n_epochs):
#     print("-" * 120)
#     network.fit(x, y, batch_size=batch_size, epochs=1)

#     # generate data
#     start_index = random.randint(0, len(longest_text()) - seq_length - 1)
#     sample = longest_text()[start_index : start_index + seq_length]

#     print(f'[{epoch}/{n_epochs}] sample: "{sample}"')
#     for temparature in [0.2, 0.5, 1.0, 1.2]:
#         prediction = ""

#         for i in range(seq_length):
#             x_pred = np.zeros((1, seq_length, len(chars)))
#             for t, char in enumerate(sample):
#                 x_pred[0, t, char2idx[char]] = 1.0

#             # prediction by network
#             preds = network.predict(x_pred, verbose=0)[0]
#             next_index = predict_index(preds, temparature)
#             next_char = idx2char[next_index]

#             # set result + prepare for net iteration
#             sample = sample[1:] + next_char
#             prediction += next_char

#         print(f'on temperature: "{temparature}" the prediction = "{prediction}"')
#     print()


# Result:
# ------------------------------------------------------------------------------------------------------------------------
# 112/112 [==============================] - 15s 120ms/step - loss: 3.3071
# [0]: seed = " shelf contains 32 volumes. Each volume contains 410 pages of 3200 characters each. Everything you c"
# on temperature: "0.2" the prediction = "ore tor the the the the the ing at ing ane ing ang the the the toe the the the to the the ange the t"
# on temperature: "0.5" the prediction = "inge se a wore Oint aite thered ane anle thit ing the theke sore tor ane ise ting d ang th ute there"
# on temperature: "1.0" the prediction = "lsYo'l oasony t e fsaveve ind warSe d, Iting ot't umeise y,d aane!blnde besd O't wrer1on, io Toldsko"
# on temperature: "1.2" the prediction = "rSWd lic- cangin'let klla w Rveekegtld tlelel'se v'nnocstnne'lnstuy ti 0ayovret. ITe: herykyucyog ey"

# ------------------------------------------------------------------------------------------------------------------------
# 112/112 [==============================] - 16s 147ms/step - loss: 0.5112
# [39]: seed = "sense that this LTE isn’t very long yet. Whenever I look at this webpage, it looks long at first gla"
# on temperature: "0.2" the prediction = "ils because I saintle in hape you take to the only make inters to gonna dispes. If how "nomberacters"
# on temperature: "0.5" the prediction = ") word it was ago and make on this LTE is back or songs to say distand in the eadn the olly still be"
# on temperature: "1.0" the prediction = "ing bike you we lifccal. But it is 5. Oh out forgen word and disple lake the whenever reason’t think"
# on temperature: "1.2" the prediction = " and I gle endire alo have pon't know what this det makeviog have song doesiag the pame is pretsa so"
