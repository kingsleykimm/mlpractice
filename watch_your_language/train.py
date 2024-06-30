from torch.utils.data import DataLoader
import torch
from models import *
from datasets import PennTreebank


def train_model(num_epochs=200, lr=1e-3, seq_len=5, eval_intervals=10, batch_size=64):
    train_dataset = PennTreebank('../data/', 'ptbdataset/ptb.train.txt', seq_len)
    test_dataset = PennTreebank('../data/', 'ptbdataset/ptb.test.txt', seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    vocab_size = train_dataset.vocab_size
    model = RecurrentLayer(vocab_size, 10, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_network = num_epochs
    for i in range(num_epochs):
        if i % eval_intervals == 0 and i > 0:
            # eval
            inputs, targets = next(iter(test_loader))
            print("Eval at iteration " + i + ". Loss is on batch size of ", 64)
            inputs = torch.permute(inputs, (1, 0, 2))
            hidden_state = torch.zeros(batch_size, model.activation_size)
            loss = 0
            for i in range(seq_len):
                output, hidden_state = model(inputs[i], hidden_state)
                loss += F.mse_loss(output, targets[i], reduction='mean')
            print("Loss :", loss) # add something here to print out the actual letters
        else:
            inputs, targets = next(iter(train_loader))
            # tmp = inputs[2][3]
            inputs = torch.permute(inputs, (1, 0, 2))
            # print(torch.equal(inputs[3][2], tmp)) check for 
            # inputs and targets will have a shape of (batch_size, num_steps, vocab)
            # switch them to (num_steps, batch_size, vocab) for easier training, because we can iteratively do it per timestep
            hidden_state = torch.zeros(batch_size, model.activation_size)
            loss = 0
            for i in range(seq_len):
                output, hidden_state = model(inputs[i], hidden_state)
                loss += F.mse_loss(output, targets[i], reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# gradient_clipping pattern:
"""
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
"""

if __name__ == '__main__':
    train_model()