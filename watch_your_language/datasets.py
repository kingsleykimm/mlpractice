import collections
import random
import re
import torch
import math
from torch.utils.data import Dataset
from collections import defaultdict
# @path should just be the file naem
# TODO: Add in an '<end>' token to signify end, put it at the end of every line, and then don't let any sequences start with the '<end>' token.
# Add '<end>' token to end of each sentence when training on perplexity
def get_file(path=''):
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = _preprocess(line)
    return lines
    
def word_tokenization(lines, min_freq):
    # need to construct the vocab
    text = []
    for line in lines:
        line = line.split(' ')
        new_line = []
        for word in line:
            if word != '':
                if word == '\n':
                    new_line.append('<end>')
                else:
                    new_line.append(word)
        text += new_line
    counter = collections.Counter(text)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sort by highest frequency
    # now we need to get the actual vocab, sorted in alphabetical order
    token_to_freq = sorted(list(set(['<unk>'] + [token for token, freq in token_freqs if freq >= min_freq])))
    freq_to_token = {token : idx for idx, token in enumerate(token_to_freq)}
    return token_to_freq, freq_to_token, text

def translation_tokenization(lines, ds_size):
    source, target = [], []
    if ds_size == None: 
        ds_size = len(lines)
    for line in lines:
        # split up by tab
        if len(source) == ds_size: break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append([t for t in f'{parts[0]} <end>'.split(' ') if t != ''])
            target.append([token for token in f'<bos> {parts[1]} <end>'.split(' ') if token != '']) # beginning of sequence token for target decoding (transformers), 
            # the input to the decoder (target sequence) will start with <bos> so it can predict the next actual target token
    return source, target

def fix_seq_len(source, target, seq_len):
    for i in range(len(source)): # ds size
        # we have a fixed sequence length, if the length in targ
        if len(source[i]) < seq_len:
            source[i] += ['<pad>' for _ in range(seq_len - len(source[i]))] # padding
        elif len(source[i]) > seq_len:
            source[i] = source[i][:seq_len]# truncation

        if len(target[i]) < seq_len:
            target[i] += ['<pad>' for _ in range(seq_len - len(target[i]))]
        elif len(target[i]) > seq_len:
            target[i] = target[i][:seq_len]
    return source, target

def construct_dictionaries(sentences, min_freq):
    token_freqs = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            token_freqs[word] += 1
    token_freq_list = sorted(token_freqs.items(), key=lambda x : x[1], reverse=True) # sort by increasing freq
    token_to_word = sorted(list(set(['<unk>'] + [x[0] for x in token_freq_list if x[1] >= min_freq])))
    word_to_token = {w : idx for idx, w in enumerate(token_to_word)} # word to index
    return token_to_word, word_to_token
def _preprocess(text):
    text = text.replace('\u202f', ' ').replace('\xa0', ' ') # replace nonbreaking space with space
    no_space = lambda char, prev_char : char in ',.!?' and prev_char != ' ' # char has to be punctuation, and prev_char can't be space
    # add space in front of punctuation
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text.lower())]
    return ''.join(out)

# remember to convert to Tensors when outputting
class PennTreebank(Dataset): # can do dataloader on this
    def __init__(self, root_dir, path, seq_len, min_freq=3, transform=None, train_vocab=None):
        self.root_dir = root_dir
        self.path = path
        self.transform = transform
        self.seq_len = seq_len
        self.text = get_file(root_dir + path)
        self.token_to_word, self.word_to_token, self.text = word_tokenization(self.text, min_freq)
        self.vocab_size = len(self.word_to_token)
        if train_vocab: # if in the test case
            self.token_to_word, self.word_to_token = train_vocab
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
        # convert to tokens, freq-to-token goes from character -> token
        inp = []
        for i in range(len(text)):
            vocab = torch.zeros(self.vocab_size)
            if text[i] not in self.word_to_token:
                vocab[self.word_to_token['<unk>']] = 1
            else:
                vocab[self.word_to_token[text[i]]] = 1 # one-hot encoding
            inp.append(vocab)
        inp = torch.stack(inp)
        target = []
        if len(targ) < self.seq_len:
            targ += ['<unk>'] * (self.seq_len - len(targ))
            print(len(targ))
        for i in range(len(targ)):
            vocab = torch.zeros(self.vocab_size)
            if targ[i] not in self.word_to_token:
                vocab[self.word_to_token['<unk>']] = 1
            else:
                vocab[self.word_to_token[targ[i]]] = 1 # one-hot encoding
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
    
class MTFraEngDataset(Dataset):
    def __init__(self, min_freq, ds_size, seq_len):
        lines = get_file('../data/fra-eng/fra.txt')
        source, target = translation_tokenization(lines, ds_size)
        self.source, self.target = fix_seq_len(source, target, seq_len)
        self.src_token_to_word, self.src_word_to_token = construct_dictionaries(self.source, min_freq)
        self.targ_token_to_word, self.targ_word_to_token = construct_dictionaries(self.target, min_freq)

    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        return self.source[index], self.target[index]