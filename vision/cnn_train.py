from torch.utils.data import DataLoader
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import argparse
from cnns import *
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
def evaluate(test_loader, model):
    batch_images, batch_targets = next(iter(test_loader))
    with torch.no_grad():
        outs = model(batch_images)
    preds = torch.argmax(outs, dim=1) # across each row, get the max
    accuracy = (preds == batch_targets).sum() / batch_targets.size(dim=0)
    return accuracy



def train_model(model_name, num_epochs, lr, eval_intervals, batch_size):

    train_dataset = FashionMNIST('../data', train=True, download=True, transform=ToTensor())
    test_dataset = FashionMNIST('../data', train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = None
    if model_name == 'lenet':
        model = LeNet(len(train_dataset.classes))
    elif model_name == 'alexnet':
        model = AlexNet(len(train_dataset.classes)) # needs to be trained on GPUs/ COLAB
    elif model_name == 'vgg':
        model = VGGNet(len(train_dataset.classes), ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), True) # TODO: train
    elif model_name == 'nin':
        model = NetworkInNetwork(len(train_dataset.classes), True) # TODO: train
    elif model_name == 'googlelenet':
        pass
    elif model_name == 'resnet':
        pass
    
    model.to(device)
        # need to upsample all the images
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        if epoch % eval_intervals == 0 and epoch> 0:
            acc = evaluate(test_loader, model)
            print(f"Accuracy : {acc}")
        else:
            new_batch, new_targets = next(iter(train_loader))
            # process each of the new_targets
            one_hot_targets = []
            for i in range(new_targets.size(dim=0)):
                one_hot = torch.zeros(len(train_dataset.classes))
                one_hot[new_targets[i]] = 1
                one_hot_targets.append(one_hot)
            one_hot_targets = torch.stack(one_hot_targets)
            new_batch, one_hot_targets = new_batch.to(device), one_hot_targets.to(device)
            outputs = model(new_batch)
            loss = model.batch_loss(outputs, one_hot_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Epoch {epoch}, Loss: {loss}")

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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eval_intervals', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip', type=int, default=6)
    parser.add_argument('--model', type=str, default='lenet')
    args = parser.parse_args()
    train_model(model_name=args.model, num_epochs=args.num_epochs, lr=args.lr, eval_intervals=args.eval_intervals, batch_size=args.batch_size)