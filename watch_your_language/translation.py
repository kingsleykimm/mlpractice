from torch.utils.data import DataLoader, RandomSampler, Subset
import torch
import argparse
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def bleu(pred_seq, target_seq, k):
    # target_seq can also be interpreted as the label seq
    # pred_seq, target_seq will come in the form of strings
    pred_len, label_len = len(pred_seq), len(target_seq)
    score = math.exp(min(0, 1 - label_len / pred_len))
    # BLEU is searching for the accuracy of n-grams, which is the n number of words in the sequence
    for n in range(1, min(pred_len, k) + 1):
        num_matches = 0
        word_store = defaultdict(int)
        for ind in range(label_len - n + 1): 
            word_store[" ".join(target_seq[ind:ind+n])] += 1
        for i in range(pred_len - n + 1):
            if word_store[" ".join(pred_seq[ind+ind:n])] > 0:
                num_matches += 1
        if num_matches == 0:
            return 0
        score *= math.pow((num_matches / (pred_len - n + 1)), math.pow(0.5, n))
    return score
    
            
def train_model(model_name, num_epochs, lr, batch_size, seq_len):
    dataset = MTFraEngDataset(1, 200000, seq_len)

    num_train_samples = int(dataset.ds_size * 0.8)
    train_ds = Subset(dataset, np.arange(num_train_samples))
    train_sampler = RandomSampler(train_ds)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)
    valid_ds = Subset(dataset, np.arange(num_train_samples, dataset.ds_size))
    valid_sampler = RandomSampler(valid_ds)
    valid_loader = DataLoader(valid_ds, sampler=valid_sampler, batch_size=batch_size)
    src_token_to_word, src_word_to_token = dataset.src_token_to_word, dataset.src_word_to_token
    targ_token_to_word, targ_word_to_token = dataset.targ_token_to_word, dataset.targ_word_to_token
    # how to handle unseen in test? Just use <unK>
    print('Starting training')
    print(targ_word_to_token['<eos>'])
    model = None
    if model_name == 'seq2seq':
        model = Seq2Seq(len(src_word_to_token), len(targ_word_to_token), 256, 256, 2, targ_word_to_token['<pad>'])
    model.apply(init_weights)
    model.to(device)
    bos_tensor = torch.tensor([targ_word_to_token['<bos>']])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(1, num_epochs + 1):
        for source, targets in train_loader:
            source, targets = source.to(device), targets.to(device)
            # source, target size: (batch_size, seq_len, vocab_size) (for each respective vocab)
            source = torch.argmax(source, 2)
            model_inputs = torch.argmax(targets, 2)[:, :-1]
            source = torch.transpose(source, 0, 1)
            targets = torch.transpose(targets, 0, 1) # (seq_len, batch_size, vocab_size)
            model_inputs = torch.transpose(model_inputs, 0, 1)
            # print(torch.equal(inputs[3][2], tmp)) check for 
            # inputs and targets will have a shape of (batch_size, num_steps/seq_len)
            # switch them to (num_steps, batch_size, vocab) for easier training, because we can iteratively do it per timestep
            Y = targets[1:, :, :].argmax(2).to(device)
            outputs, final_hiddens = model(source, model_inputs)
            outputs = torch.transpose(outputs, 0, 1)
            Y = torch.transpose(Y, 0, 1)
            
            mask = model.put_masks(Y).type(torch.long).to(device)
            Y *= mask
            loss = 0
            for seq_ind in range(Y.size(dim=1)):
                ce_matrix = F.cross_entropy(outputs[:, seq_ind], Y[:, seq_ind], reduction='none') * mask[:, seq_ind]
                loss += ce_matrix.sum()

            # loss = F.cross_entropy(outputs, Y, reduction='none') * mask
            loss = loss / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping, find a better way here
            optimizer.step()
        print(f"Epoch {i}, Training Loss: {loss}")
        batched_average_bleu = 0.0
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.argmax(inputs, 2)
            model_inputs = torch.argmax(targets, 2)[:, :-1]
            Y = targets[:, 1:].argmax(2)
            inputs = torch.transpose(inputs, 0, 1)
            targets = torch.transpose(inputs, 0, 1)
            model_inputs = torch.transpose(model_inputs, 0, 1)
            outputs = [bos_tensor.repeat(1, inputs.size(dim=1)).to(device)]
            with torch.no_grad():
                Y_hat = model.predict_steps(seq_len, (inputs, model_inputs), outputs)
                Y_hat = Y_hat.t()
            # outputs will be a tensor of num_steps, batch_size, vocab_size, we need to convert it to the tokens
            pred_string, target_string = None, None
            for targ, pred in zip(Y, Y_hat):
                target_string = [targ_token_to_word[targ[i]] for i in range(targ.size(dim=0))]
                pred_string = []
                for token in pred:
                    if token == '<eos>':
                        break
                    pred_string.append(targ_token_to_word[token])
                batched_average_bleu += bleu(pred_string, target_string, k=7)
        print(f"Batch Average Bleu: {batched_average_bleu}")
        

# gradient_clipping pattern:
"""
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
"""

train_model(model_name='seq2seq', num_epochs=30, lr=0.005, batch_size=128, seq_len = 9)