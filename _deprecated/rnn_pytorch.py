import numpy as np


text = ["hey how are you", "good i am fine", "have a nice day"]

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set("".join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char:ind for ind, char in int2char.items()}


# Finding the lenght of the longest string in our data
maxlen = len(max(text, key=len))

# Padding
# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length
# of the sentence matches with the length of the longest sentence
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '



# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])

    # remove first character for target sequence
    target_seq.append(text[i][1:])

    # print(f"Input Sequence: {input_seq[i]}\nTarget Sequence: {target_seq[i]}")

# print(input_seq)

for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

# print(input_seq)

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1

    return features


# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

import torch
from torch import nn

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()


# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# Instantiate the model with hyper parameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# define hyperparameters
n_epochs = 100
lr=0.01

# define Loss, Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()# Clears existing gadients from previous epoch
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropergation and calculates gradients
    optimizer.step()# Updates the weights accordingly

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}/{n_epochs}.........", end=" ")
        print(f"Loss: {loss.item()}")


# This function takes in the model and character as arguments 
# and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)

    out, hidden = model(character)
    prob = nn.functional.softmax(out[-1], dim=0).data

    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

# This function takes the desired output length and input characters as arguments, 
# returning the produced sentence
def sample(model, out, out_len, start="hey"):
    model.eval()# eval mode
    start = start.lower()

    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len = len(chars)

    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        char.append(char)

    return "".join(chars)


sample(model, 15, 'good')