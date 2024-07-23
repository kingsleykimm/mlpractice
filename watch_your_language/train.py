from torch.utils.data import DataLoader
import torch
from models import *
from datasets import PennTreebank
import argparse
from transformer import Transformer
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def evaluate(validate_dataset, vocab_size, model, word_to_token, iteration):
    z = 0
    perplexity = 0
    for sentence in validate_dataset:
        # validate_dataset is a list of strings
        # calculate perplexity here
        # Perplexity measures how well the model represents the target data
        # Mold data for tensor input into model
        words = sentence.split(' ')
        processed = []
        for word in words:
            if word not in word_to_token:
                processed.append('<unk>')
            elif word == '\n':
                processed.append('<end>')
            else:
                processed.append(word)
        seq_length = len(processed)
        hidden_state = torch.zeros(1, model.hidden_dim)
        # processed = torch.Tensor(processed)
        # processed = torch.unsqueeze(processed, 0) # shape is now (1, vocab_size)
        # we need to create the last shape which is now (sentence_length, 1, vocab_size)

        # inputs should go up to but not including <end>, targets should include <end> since the inputs need to be able to predict it
        seq = []
        for i in range(seq_length):
            one_hot = torch.zeros(vocab_size)
            token_pos = word_to_token[processed[i]]
            one_hot[token_pos] = 1
            seq.append(one_hot)
        seq = torch.stack(seq)
        targets = seq[1:]
        inps = seq[:seq_length - 1]
        inps = torch.unsqueeze(inps, dim=1) # seq has size (seq_len, 1, vocab_size)
        targets = torch.unsqueeze(targets, dim=1)
        z = model.batch_train(inps, targets, 1, seq_length - 1)
        # for i in range(seq.size(dim=0)): # dim = 1 because processed is just (1, vocab_size) (batch_size = 1)
        #     print(seq[i].shape, hidden_state.shape)
        #     probs, hidden_state = model(seq[i], hidden_state)
        #     z += torch.log(probs[0][word_to_token[processed[i]]])
        z /= len(processed)
        perplexity += torch.exp(z)
    perplexity /= len(word_to_token)
    print(f"Perplexity at iteration {iteration} is {perplexity}.")



def train_model(model_name, num_epochs, lr, seq_len, eval_intervals, batch_size):
    train_dataset = PennTreebank('../data/', 'ptbdataset/ptb.train.txt', seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vocab_size = train_dataset.vocab_size
    with open('../data/ptbdataset/ptb.valid.txt') as f:
        validate_dataset = f.readlines()
    test_dataset = PennTreebank('../data/', 'ptbdataset/ptb.test.txt', seq_len, train_vocab=(train_dataset.token_to_word, train_dataset.word_to_token))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # how to handle unseen in test? Just use <unK>
    model = None
    if model_name == 'lstm':
        model = LSTM(vocab_size, 32, 32) # tune later
    elif model_name == 'recurrent':
        model = RecurrentLayer(vocab_size, 32, vocab_size)
    elif model_name == 'gru':
        model = GRU(vocab_size, 32)
    elif model_name == 'birnn':
        model = BiRNN(vocab_size, 32, 'tanh')
    elif model_name == 'drnn':
        model = DeepRNN(vocab_size, 32, 4)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(1, num_epochs + 1):
        if i % eval_intervals == 0 and i > 0:
            evaluate(validate_dataset, vocab_size, model, train_dataset.word_to_token, i)
        else:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # tmp = inputs[2][3]
                inputs = torch.permute(inputs, (1, 0, 2))
                targets = torch.permute(targets, (1, 0, 2))
                # print(torch.equal(inputs[3][2], tmp)) check for 
                # inputs and targets will have a shape of (batch_size, num_steps, vocab)
                # switch them to (num_steps, batch_size, vocab) for easier training, because we can iteratively do it per timestep
                loss = model.batch_train(inputs, targets, batch_size, seq_len)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 6) # gradient clipping, find a better way here
                optimizer.step()
                print(f"Epoch {i}, Loss: {loss}")

# gradient_clipping pattern:
"""
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--eval_intervals', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip', type=int, default=6)
    parser.add_argument('--model', type=str, default='lstm')
    args = parser.parse_args()
    train_model(model_name=args.model, num_epochs=args.num_epochs, lr=args.lr, seq_len=args.seq_len, eval_intervals=args.eval_intervals, batch_size=args.batch_size)