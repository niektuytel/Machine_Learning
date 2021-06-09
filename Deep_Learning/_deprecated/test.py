import numpy as np
import matplotlib.pyplot as plt

BIN_DIM = 8
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1

ALPHA = 0.1
ITER_NUM = 10000
LOG_ITER = ITER_NUM // 10
PLOT_ITER = ITER_NUM // 200

largest = pow(2, BIN_DIM)
decimal = np.array([range(largest)]).astype(np.uint8).T
binary = np.unpackbits(decimal, axis=1)

# weight values
w0 = np.random.normal(0, 1, [INPUT_DIM, HIDDEN_DIM])
wh = np.random.normal(0, 1, [HIDDEN_DIM, HIDDEN_DIM])
w1 = np.random.normal(0, 1, [HIDDEN_DIM, OUTPUT_DIM])

# delta values
d0 = np.zeros_like(w0)
dh = np.zeros_like(wh)
d1 = np.zeros_like(w1)

errs = list()
accs = list()

error = 0
accuracy = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(out):
    return out * (1 - out)

def bin2dec(b):
    out = 0
    for i, x in enumerate(b[::-1]):
        out += x * pow(2, i)

    return out

for i in range(ITER_NUM + 1):
    # a + b = c
    a_dec = np.random.randint(largest / 2)
    b_dec = np.random.randint(largest / 2)
    c_dec = a_dec + b_dec

    a_bin = binary[a_dec]
    b_bin = binary[b_dec]
    c_bin = binary[c_dec]

    pred = np.zeros_like(c_bin)
    overall_err = 0# total error in the whole calculation process.

    output_deltas = list()
    hidden_values = list()
    hidden_values.append(np.zeros(HIDDEN_DIM))

    future_delta = np.zeros(HIDDEN_DIM)

    # forward propagation
    for pos in range(BIN_DIM)[::-1]:
        X = np.array([[a_bin[pos], b_bin[pos]]])# shape=(1, 2)
        Y = np.array([[c_bin[pos]]])# shape=(1, 1)

        hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
        output = sigmoid(np.dot(hidden, w1))

        pred[pos] = np.round(output[0][0])

        # squared mean error
        output_err = Y - output
        output_deltas.append(output_err * deriv_sigmoid(output))
        hidden_values.append(hidden)

        overall_err += np.abs(output_err[0])
    
    # backwardpropagation through time
    for pos in range(BIN_DIM):
        X = np.array([[a_bin[pos], b_bin[pos]]])

        hidden = hidden_values[-(pos + 1)]
        prev_hidden = hidden_values[-(pos + 2)]

        output_delta = output_deltas[-(pos + 1)]
        hidden_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w1.T)) * deriv_sigmoid(hidden)

        d1 += np.dot(np.atleast_2d(hidden).T, output_delta)
        dh += np.dot(np.atleast_2d(prev_hidden).T, hidden_delta)
        d0 +=np.dot(X.T, hidden_delta)

        future_delta = hidden_delta
    
    w1 += ALPHA * d1
    w0 += ALPHA * d0
    wh += ALPHA * dh

    d1 *= 0
    dh *= 0
    d0 *= 0

    error += overall_err
    if(bin2dec(pred) == c_dec):
        accuracy += 1

    if(i % PLOT_ITER == 0):
        errs.append(error / PLOT_ITER)
        accs.append(accuracy / PLOT_ITER)

        error = 0
        accuracy = 0

    if(i % LOG_ITER == 0):
        print("Iter", i)
        print("Error :", overall_err)
        print("Pred :", pred)
        print("True :", c_bin)
        print(a_dec, "+", b_dec, "=", bin2dec(pred))
        print("---------------")












# # Print:
# # --iteration: 0, loss: 64.07458186056506 --
# #  sot’mene tarnoltthsy ily intey arukeuwladelent bin.! to trad oif yonnilcilt ggssnefuln irle ys ldy vitenrus sovwodhe tondrit ubow ceem. akzl.nnem i tyy tt belthad yerr ke coby n. moto sully, veris, a
# #  .......
# # --iteration: 150, loss: 41.537047003921835 --
# # ng! bet reviss vie activer lies are nough now be that. of it lige wamt? in os linting to the suter day that i vingiras i was did i lally a ent the der about? it’t that ungure. mesplof how guts. pracre

# # Note:
# # The criteria is the performance on the validation set.  
# # Typically LSTM outperforms RNN, as it does a better job at avoiding the vanishing gradient problem,  
# # and can model longer dependences.  


# import numpy as np
# import parameter


# class Tanh():
#     def __call__(self, x):
#         return 2 / (1 + np.exp(-2 * x)) - 1

#     def derivative(self, x):
#         return 1 - (self.__call__(x) ** 2)

# act_functions = {
#     "tanh":Tanh
# }

# def _one_hot_encode(inputs, vocab_size):
#     output = {}
#     for t, x in enumerate(inputs):
#         output[t] = np.zeros((vocab_size,1))
#         output[t][x] = 1
#     return output

# class RNN:
#     def __init__(self, vocab_size, hidden_size, activation="tanh"):
#         self.vocab_size = vocab_size
#         self.activation = act_functions[activation]()

#         # weights to input
#         self.Wx = np.random.randn(hidden_size, vocab_size)*0.01
#         self.mWx = np.zeros_like(self.Wx)# memory variables for Adagrad

#         # weights to hidden
#         self.Wh = np.random.randn(hidden_size, hidden_size)*0.01
#         self.mWh = np.zeros_like(self.Wh)# memory variables for Adagrad

#         # weights to output
#         self.Wy = np.random.randn(vocab_size, hidden_size)*0.01 
#         self.mWy = np.zeros_like(self.Wy)# memory variables for Adagrad

#         # bias to hidden
#         self.bh = np.zeros((hidden_size, 1)) 
#         self.mbh = np.zeros_like(self.bh)# memory variables for Adagrad
        
#         # bias to output
#         self.by = np.zeros((vocab_size, 1)) 
#         self.mby = np.zeros_like(self.by)# memory variables for Adagrad

#         # set gradients
#         self._empty_gradients()

#     def one_to_many(self, h, char_idx, output_size):
#         """ 
#         sample a sequence of integers from the model 
#         h is memory state, seed_ix is seed letter for first time step
#         """
#         # one hot encoding
#         x = np.zeros((vocab_size, 1))
#         x[char_idx] = 1

#         output = []
#         for t in range(output_size):
#             h = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, h) + self.bh)
#             y = np.dot(self.Wy, h) + self.by
#             p = np.exp(y) / np.sum(np.exp(y))

#             ix = np.random.choice(range(self.vocab_size), p=p.ravel())
#             x = np.zeros((self.vocab_size, 1))
#             x[ix] = 1
#             output.append(ix)

#         return output

#     def forward(self, inputs, targets, hprev):
#         loss = 0
#         self.h = {-1: np.copy(hprev)}
#         self.x = _one_hot_encode(inputs, self.vocab_size)
#         self.p = {}

#         # forward pass
#         for t in range(len(inputs)-1):

#             wsum = np.dot(self.Wx, self.x[t]) + np.dot(self.Wh, self.h[t-1]) + self.bh
#             self.h[t] = self.activation(wsum) # hidden state

#             y = np.dot(self.Wy, self.h[t]) + self.by # unnormalized log probabilities for next chars
#             self.p[t] = np.exp(y) / np.sum(np.exp(y)) # probabilities for next chars

#             loss += -np.log(self.p[t][targets[t],0]) # softmax (cross-entropy loss)

#         return loss

#     def backward(self, targets, learning_rate):
#         self._empty_gradients()
#         dhnext = np.zeros_like(self.h[0])

#         # back propagation (chain-rule)
#         for t in reversed(range(len(inputs)-1)):
#             self.dy = np.copy(self.p[t])
#             self.dy[targets[t]] -= 1 
            
#             self.dWy += np.dot(self.dy, self.h[t].T)
#             self.dby += self.dy

#             self.dh = np.dot(self.Wy.T, self.dy) + dhnext # backprop into h
#             dhraw = (1 - self.h[t] * self.h[t]) * self.dh # backprop through tanh nonlinearity
#             self.dbh += dhraw

#             self.dWx += np.dot(dhraw, self.x[t].T)
#             self.dWh += np.dot(dhraw, self.h[t-1].T)
#             dhnext = np.dot(self.Wh.T, dhraw)
        
#         # clip to mitigate exploding gradients
#         self._limit_gradients(5)

#         # update layer parameters
#         self._update_params(learning_rate)

#         return self.h[len(inputs)-2]

#     def _empty_gradients(self):
#         # weights to input
#         self.dWx = np.zeros_like(self.Wx)

#         # weights to hidden
#         self.dWh = np.zeros_like(self.Wh) 

#         # weights to output
#         self.dWy = np.zeros_like(self.Wy) 

#         # bias to hidden
#         self.dbh = np.zeros_like(self.bh)
        
#         # bias to output
#         self.dby = np.zeros_like(self.by)

#     def _limit_gradients(self, limit):
#         # clip to mitigate exploding gradients
#         np.clip(self.dWx, -limit, limit, out=self.dWx) 
#         np.clip(self.dWh, -limit, limit, out=self.dWh) 
#         np.clip(self.dWy, -limit, limit, out=self.dWy) 
#         np.clip(self.dbh, -limit, limit, out=self.dbh) 
#         np.clip(self.dby, -limit, limit, out=self.dby) 

#     def _update_params(self, learning_rate):
#         # weights to input
#         self.mWx += self.dWx * self.dWx
#         self.Wx = -learning_rate * self.dWx / np.sqrt(self.mWx + 1e-8)

#         # weights to hidden
#         self.mWh += self.dWh * self.dWh
#         self.Wh += -learning_rate * self.dWh / np.sqrt(self.mWh + 1e-8)

#         # weights to output
#         self.mWy += self.dWy * self.dWy
#         self.Wy += -learning_rate * self.dWy / np.sqrt(self.mWy + 1e-8)

#         # bias to hidden
#         self.mbh += self.dbh * self.dbh
#         self.bh += -learning_rate * self.dbh / np.sqrt(self.mbh + 1e-8)
        
#         # bias to output
#         self.mby += self.dby * self.dby
#         self.by += -learning_rate * self.dby / np.sqrt(self.mby + 1e-8)

# # PARAMETERS
# debug = True
# n_trainings = 30000
# hidden_size = 100# size of hidden layer of neurons
# sequence_length = 25# number of steps to unroll the RNN for
# learning_rate = 1e-1

# if __name__ == "__main__":
#     # datasets
#     X_train, y_train, char_to_idx, idx_to_char, characters = parameter.get_datasets(
#         sequence_length=sequence_length,
#         debug=debug
#     )
    
#     # model parameters
#     vocab_size = len(characters)
#     rnn = RNN(vocab_size, hidden_size)
#     smooth_loss = -np.log(1.0 / vocab_size) * sequence_length # loss at iteration 0

#     for it in range(n_trainings):
#         # reset RNN memory
#         hprev = np.zeros((hidden_size,1)) 
            
#         # train model
#         for i in range(len(X_train)):
#             inputs = X_train[i]
#             targets = y_train[i]

#             # forward
#             loss = rnn.forward(inputs, targets, hprev)
#             smooth_loss = smooth_loss * 0.999 + loss * 0.001

#             # backward
#             hprev = rnn.backward(targets, learning_rate)

#         # print progress
#         print(f"--iteration: {it}, loss: {smooth_loss} --", end="\r", flush=True)
#         if it % 50 == 0:
#             samples = rnn.one_to_many(hprev, inputs[0], 200)
#             txt = "".join(idx_to_char[ix] for ix in samples)
#             print (f"\n{txt}\n")







# from datetime import datetime

# import torch
# from torch import nn, optim
# from torchtext.legacy import data
# from torchtext.legacy.data import BucketIterator

# from data_gen_utils import gen_df 
# from dataframe_dataset import DataFrameDataset


# import numpy as np
# import random



# # set random seeds for reproducibility
# torch.manual_seed(12)
# torch.cuda.manual_seed(12)
# np.random.seed(12)
# random.seed(12)

# # check if cuda is enabled
# USE_GPU=1
# # Device configuration
# device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')


# def tokenize(text):
#     # simple tokenizer
#     words = text.lower().split()
#     return words


# def accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """

#     # get max values along rows
#     _, indices = preds.max(dim=1)
#     # values, indices = torch.max(tensor, 0)

#     correct = (indices == y).float()  # convert into float for division
#     acc = correct.sum()/len(correct)
#     return acc




# # gen the trainning data
# min_seq_len = 100
# max_seq_len = 300

# # numer of tokenes in vocab to generate, max 10
# # it is equal the number of classes
# seq_tokens = 10

# n_train = 1000
# n_valid = 200

# train_df = gen_df(n=n_train, min_seq_len=min_seq_len,
#                       max_seq_len=max_seq_len, seq_tokens=seq_tokens)
# valid_df = gen_df(n=n_valid, min_seq_len=min_seq_len,
#                       max_seq_len=max_seq_len, seq_tokens=seq_tokens)


# print(train_df)
# print(valid_df)

# TEXT = data.Field(sequential=True, lower=True, tokenize=tokenize,fix_length=None)
# LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

# fields = {"text": TEXT, "label": LABEL}


# train_ds = DataFrameDataset(train_df, fields)
# valid_ds = DataFrameDataset(valid_df, fields)

# # numericalize the words
# TEXT.build_vocab(train_ds, min_freq=1)
# print(TEXT.vocab.freqs.most_common(20))

# vocab = TEXT.vocab
# vocab_size = len(vocab)

# batch_size = 4
# train_iter = BucketIterator(
#     train_ds, 
#     batch_size=batch_size, 
#     sort_key=lambda x: len(x.text), 
#     sort_within_batch=True, 
#     device=device)

# valid_iter = BucketIterator(
#     valid_ds, 
#     batch_size=batch_size, 
#     sort_key=lambda x: len(x.text), 
#     sort_within_batch=True,
#     device=device)

# #hidden size
# n_hid=200
# # embed size
# n_embed=10
# # number of layers
# n_layers=1




# class SeqLSTM(nn.Module):
#     """
#     LSTM example for long sequence
#     """

#     def __init__(self, vocab_size, output_size, embed_size, hidden_size, num_layers=1):
#         super().__init__()

#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers

#         self.embed = nn.Embedding(vocab_size, embed_size)

#         #after the embedding we can add dropout
#         self.drop = nn.Dropout(0.1)

#         self.network = nn.LSTM(embed_size, hidden_size,
#                             num_layers, batch_first=False)

#         self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, seq):
#         # Embed word ids to vectors
#         len_seq, bs = seq.shape
#         w_embed = self.embed(seq)
#         w_embed = self.drop(w_embed)

#         # https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
#         output, (hidden, cell) = self.network(w_embed)

#         # use dropout
#         # hidden = self.drop(hidden[-1,:,:])

#         # hidden has size [1,batch,hid dim]
#         # this does .squeeze(0) now hidden has size [batch, hid dim]
#         last_output = output[-1, :, :]
#         # last_output = self.drop(last_output)

#         out = self.linear(last_output)

#         return out


# # gen the trainning
# min_seq_len = 100
# max_seq_len = 300

# # numer of tokenes in vocab to generate, max 10
# # it is equal the number of classes
# seq_tokens = 10

# n_train = 1000
# n_valid = 200

# train_df = gen_df(n=n_train, min_seq_len=min_seq_len,
#                       max_seq_len=max_seq_len, seq_tokens=seq_tokens)
# valid_df = gen_df(n=n_valid, min_seq_len=min_seq_len,
#                       max_seq_len=max_seq_len, seq_tokens=seq_tokens)


# print(train_df)
# print(valid_df)

# TEXT = data.Field(sequential=True, lower=True, tokenize=tokenize,fix_length=None)
# LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

# fields = {"text": TEXT, "label": LABEL}


# train_ds = DataFrameDataset(train_df, fields)
# valid_ds = DataFrameDataset(valid_df, fields)

# # numericalize the words
# TEXT.build_vocab(train_ds, min_freq=1)

# #hidden size
# n_hid=200
# # embed size
# n_embed=20
# # number of layers
# n_layers=1

# print("-"*80)
# print(f'n_train={n_train}, n_valid={n_valid}')
# print(f'min_seq_len={min_seq_len}, max_seq_len={max_seq_len}')

# print(f'model params')
# print(f'vocab={vocab_size}, output={seq_tokens}')
# print(f'n_layers={n_layers}, n_hid={n_hid} embed={n_embed}')

# model = SeqLSTM(vocab_size=vocab_size, output_size=seq_tokens,
#                     embed_size=n_embed, hidden_size=n_hid)
# model.to(device)

# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())


# batch_size = 16
# train_iter = BucketIterator(
#     train_ds, 
#     batch_size=batch_size, 
#     sort_key=lambda x: len(x.text), 
#     sort_within_batch=True, 
#     device=device)

# valid_iter = BucketIterator(
#     valid_ds, 
#     batch_size=batch_size, 
#     sort_key=lambda x: len(x.text), 
#     sort_within_batch=True,
#     device=device)

# epoch_loss = 0
# epoch_acc = 0
# epoch = 60

# for e in range(epoch):

#     start_time = datetime.now()
#     # train loop
#     model.train()
#     for batch_idx, batch in enumerate(train_iter):

#         # get the inputs
#         inputs, labels = batch
#         # move data to device (GPU if enabled, else CPU do nothing)
#         inputs, labels = inputs.to(device), labels.to(device)

#         model.zero_grad()
#         #optimizer.zero_grad()

#         # get model output
#         predictions = model(inputs)

#         # prediction are [batch, out_dim]
#         # batch.label are [1,batch] <- should be mapped to  output vector
#         loss = loss_function(predictions, labels)
#         epoch_loss += loss.item()

#         # do backward and optimization step
#         loss.backward()
#         optimizer.step()

#     # mean epoch loss
#     epoch_loss = epoch_loss / len(train_iter)

#     time_elapsed = datetime.now() - start_time

#     # evaluation loop
#     model.eval()
#     for batch_idx, batch in enumerate(valid_iter):

#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # get model output
#         predictions = model(inputs)

#         # compute batch validation accuracy
#         acc = accuracy(predictions, labels)

#         epoch_acc += acc

#     epoch_acc = epoch_acc/len(valid_iter)

#     # show summary

#     print(
#         f'Epoch {e}/{epoch} loss={epoch_loss} acc={epoch_acc} time={time_elapsed}')
#     epoch_loss = 0
#     epoch_acc = 0