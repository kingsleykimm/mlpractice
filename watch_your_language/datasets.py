import collections
import random
import re
import torch
import math
from torch.utils.data import Dataset
# @path should just be the file naem
def get_file(path='', tokenization='char'):
    with open(path) as f:
        text = _preprocess(f.read())
    return text
    
def character_tokenization(text, min_freq):
    # need to construct the vocab
    text = text.split()
    counter = collections.Counter(text)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sort by highest frequency
    # now we need to get the actual vocab, sorted in alphabetical order
    token_to_freq = sorted(list(set(['<unk>'] + [token for token, freq in token_freqs if freq >= min_freq])))
    freq_to_token = {token : idx for idx, token in enumerate(token_to_freq)}
    return token_to_freq, freq_to_token, text
    
def word_tokenization(text, min_freq):
    # need to construct the vocab
    text = text.split(' ')
    counter = collections.Counter(text)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sort by highest frequency
    # now we need to get the actual vocab, sorted in alphabetical order
    token_to_freq = sorted(list(set(['<unk>'] + [token for token, freq in token_freqs if freq >= min_freq])))
    freq_to_token = {token : idx for idx, token in enumerate(token_to_freq)}
    return token_to_freq, freq_to_token


def _preprocess(text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

# remember to convert to Tensors when outputting
class PennTreebank(Dataset): # can do dataloader on this
    def __init__(self, root_dir, path, seq_len, min_freq=5, transform=None):
        self.root_dir = root_dir
        self.path = path
        self.transform = transform

        self.text = get_file(root_dir + path)
        self.token_to_freq, self.freq_to_token = character_tokenization(self.text, min_freq)
        # how do we do random sequencing? we don't we do all of them
        self.dataset = []
        for d in range(self.seq_len):
            new_seq_len = math.floor((len(self.text) - d) / self.seq_len)
            for i in range(new_seq_len, len(self.text), new_seq_len):
                self.dataset.append("".join(self.text[i:i+seq_len]))
        self.dataset = torch.tensor(self.dataset)
        pass
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]