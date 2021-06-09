from keras import datasets
import numpy as np
import re

def clean_str(string):
    string = re.sub("[^a-z\s\?\’\”\“\!\-\,\.]", "", string, 0, re.IGNORECASE | re.MULTILINE)
    return string.lower()

def get_datasets(sequence_length, debug):
    if debug:
        text = clean_str("Remember that “good night” normally means that you are saying goodbye. It is also commonly used right before going to bed. I’m sorry, i don’t understand. Could you please repeat that?")
    else:
        text = clean_str(open("../../../_EXTRA/data/LongestTextEver.txt", "r", encoding="utf8").read())

    # generate decoder & encoder
    characters  = list(set(text))
    vocab_size  = len(characters)
    data_size   = len(text)
    char_to_idx = {ch: i for i,ch in enumerate(characters)}
    idx_to_char = {i: ch for i,ch in enumerate(characters)}
    
    # generate training data
    n_total   = (data_size // sequence_length) - 1# remove last than never overflows
    text      = text[:n_total * sequence_length]
    data_size = len(text)
    inputs    = np.full(n_total, None)
    targets   = np.full(n_total, None)

    for i in range(0, data_size, sequence_length):
        j = i//sequence_length
        inputs[j]  = [char_to_idx[char] for char in text[(i  ):(i  )+sequence_length]]
        targets[j] = [char_to_idx[char] for char in text[(i+1):(i+1)+sequence_length]]

        if debug:    
            print(f"input :{[idx_to_char[i] for i in inputs[j]]}")
            print(f"target:{[idx_to_char[i] for i in targets[j]]}")

    return inputs, targets, char_to_idx, idx_to_char, characters

# Sample usages
if __name__ == "__main__":
    # PARAMETERS
    debug = True
    sequence_length = 25 

    X_train, y_train, char_to_idx, idx_to_char, characters = get_datasets(
        sequence_length=sequence_length,
        debug=debug
    )
