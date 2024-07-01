from torch.utils.data import DataLoader
import torch
from models import *
from datasets import PennTreebank
def evaluate(validate_dataset, vocab_size, model, word_to_token, iteration):
    z = 0
    validate_batched = []
    perplexity = 0
    for sentence in validate_dataset:
        # calculate perplexity here
        # Perplexity measures how well the model represents the target data
        # Mold data for tensor input into model
        words = sentence.split(' ')
        processed = []
        for word in words:
            if word not in word_to_token:
                processed.append('<unk>')
            else:
                processed.append(word)
        hidden_state = torch.zeros(1, model.activation_size)
        # processed = torch.Tensor(processed)
        # processed = torch.unsqueeze(processed, 0) # shape is now (1, vocab_size)
        # we need to create the last shape which is now (sentence_length, 1, vocab_size)
        seq = []
        for i in range(len(processed)):
            one_hot = torch.zeros(vocab_size)
            token_pos = word_to_token[processed[i]]
            one_hot[token_pos] = 1
            seq.append(one_hot)
        seq = torch.stack(seq)
        seq = torch.unsqueeze(seq, dim=1)
        for i in range(seq.size(dim=0)): # dim = 1 because processed is just (1, vocab_size) (batch_size = 1)
            probs, hidden_state = model(seq[i], hidden_state)
            z += torch.log(probs[0][word_to_token[processed[i]]])

        z *= -1
        z /= vocab_size
        perplexity += torch.exp(z)
    perplexity /= len(validate_dataset)
    print(f"Perplexity at iteration {iteration} is {perplexity}.")

def custom_ce_loss(pred : torch.Tensor, targ : torch.Tensor):
    # both are sizes of (batch_size, vocab_size)
    batch_size = pred.size(dim=0)
    batch_loss = 0
    for i in range(batch_size):
        pred_word, targ_word = pred[i], targ[i]
        batch_loss += -torch.sum(targ_word @ torch.log(pred_word))
    return batch_loss / batch_size



def train_model(num_epochs=500, lr=1e-3, seq_len=5, eval_intervals=100, batch_size=64):
    train_dataset = PennTreebank('../data/', 'ptbdataset/ptb.train.txt', seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vocab_size = train_dataset.vocab_size
    with open('../data/ptbdataset/ptb.valid.txt') as f:
        validate_txt = f.read()
        validate_dataset = validate_txt.split('\n')
    test_dataset = PennTreebank('../data/', 'ptbdataset/ptb.test.txt', seq_len, train_vocab=(train_dataset.token_to_word, train_dataset.word_to_token))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # how to handle unseen in test? Just use <unK>
    model = RecurrentLayer(vocab_size, 10, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(1, num_epochs + 1):
        if i % eval_intervals == 0 and i > 0:
            evaluate(validate_dataset, vocab_size, model, train_dataset.word_to_token, i)
            # # eval
            # inputs, targets = next(iter(test_loader))
            # print("Eval at iteration " + str(i) + ". Loss is on batch size of ", str(64))
            # inputs = torch.permute(inputs, (1, 0, 2))
            # targets = torch.permute(targets, (1, 0, 2))
            # hidden_state = torch.zeros(batch_size, model.activation_size)
            # loss = 0
            # sample = []
            # for i in range(seq_len):
            #     # print(inputs[i].shape, hidden_state.shape)
            #     output, hidden_state = model(inputs[i], hidden_state)
            #     sample.append(output[0])
            #     loss += F.mse_loss(output, targets[i], reduction='mean')
            # sample = torch.stack(sample)
            # sample_phrase = train_dataset.tokens_to_letters(sample)
            # target_phrase = train_dataset.tokens_to_letters
            # print("Loss :", loss.item()) # add something here to print out the actual letters
        else:
            inputs, targets = next(iter(train_loader))
            # tmp = inputs[2][3]
            inputs = torch.permute(inputs, (1, 0, 2))
            targets = torch.permute(targets, (1, 0, 2))
            # print(torch.equal(inputs[3][2], tmp)) check for 
            # inputs and targets will have a shape of (batch_size, num_steps, vocab)
            # switch them to (num_steps, batch_size, vocab) for easier training, because we can iteratively do it per timestep
            hidden_state = torch.zeros(batch_size, model.activation_size)
            loss = 0
            for ind in range(seq_len):
                output, hidden_state = model(inputs[ind], hidden_state)
                # targets[i] already contains the one-hot vectors
                ce_loss = custom_ce_loss(output, targets[ind])
                loss += ce_loss
            optimizer.zero_grad()
            loss.backward()
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
    train_model()