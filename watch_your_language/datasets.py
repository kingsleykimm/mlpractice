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
    
def word_tokenization(text, min_freq):
    # need to construct the vocab
    text = text.split()
    counter = collections.Counter(text)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sort by highest frequency
    # now we need to get the actual vocab, sorted in alphabetical order
    token_to_freq = sorted(list(set(['<unk>'] + [token for token, freq in token_freqs if freq >= min_freq])))
    freq_to_token = {token : idx for idx, token in enumerate(token_to_freq)}
    return token_to_freq, freq_to_token, text
    
# def word_tokenization(text, min_freq):
#     # need to construct the vocab
#     text = text.split(' ')
#     counter = collections.Counter(text)
#     token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sort by highest frequency
#     # now we need to get the actual vocab, sorted in alphabetical order
#     token_to_freq = sorted(list(set(['<unk>'] + [token for token, freq in token_freqs if freq >= min_freq])))
#     freq_to_token = {token : idx for idx, token in enumerate(token_to_freq)}
#     return token_to_freq, freq_to_token


def _preprocess(text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

# remember to convert to Tensors when outputting
class PennTreebank(Dataset): # can do dataloader on this
    def __init__(self, root_dir, path, seq_len, min_freq=5, transform=None):
        self.root_dir = root_dir
        self.path = path
        self.transform = transform
        self.seq_len = seq_len
        self.text = get_file(root_dir + path)
        self.token_to_word, self.word_to_token, self.text = word_tokenization(self.text, min_freq)
        self.vocab_size = len(self.word_to_token)
        # how do we do random sequencing? we don't we do all of them
        self.dataset = []
        for d in range(self.seq_len):
            # new start is going to be at d
            for i in range(d, len(self.text), self.seq_len):
                self.dataset.append((self.text[i:i+self.seq_len], i, self.seq_len))
        pass
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        text, i, seq_len = self.dataset[index]
        # need to return the current text and then the next target as well
        targ = self.text[i+1:i+1+seq_len]
        # apply padding
        if len(targ) < self.seq_len:
            targ += ['unk'] * (self.seq_len - len(targ))
        # convert to tokens, freq-to-token goes from character -> token
        inp = []
        for i in range(len(text)):
            vocab = torch.zeros(self.vocab_size)
            vocab[self.word_to_token[text[i]]] = 1 # one-hot encoding
            inp.append(vocab)
        inp = torch.stack(inp)
        target = []
        for i in range(len(targ)):
            vocab = torch.zeros(self.vocab_size)
            vocab[self.word_to_token[targ[i]]] = 1
            target.append(vocab)
        target = torch.stack(target)
        return inp, target
    def tokens_to_letters(self, tokens : torch.Tensor) -> str:
        # apply token_to_freq on every letter
        result = ""
        for token in tokens:
            # each token will be a tensor of length vocab_size, just choose the max
            max_token = torch.argmax(token)
            letter = self.token_to_freq[max_token]
            result += letter
        return result