# Mostly unused, moved all the training onto Colab for GPUS.

from torch.utils.data import DataLoader
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import argparse
from cnns import *
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
def evaluate(test_loader, model):
    accuracy = []
    for batch_images, batch_targets in test_loader:
      batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
      with torch.no_grad():
          outs = model(batch_images)
      preds = torch.argmax(outs, dim=1) # across each row, get the max
      accuracy.append((preds == batch_targets).sum() / batch_targets.size(dim=0))
    return sum(accuracy) / len(accuracy)



def train_model(model_name, num_epochs, lr, batch_size):
    train_dataset = FashionMNIST('./data', train=True, download=True, transform=ToTensor())
    test_dataset = FashionMNIST('./data', train=False, download=True, transform=ToTensor())
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
        model = GoogleLeNet(len(train_dataset.classes))
    elif model_name == 'resnet':
        model = ResNet(((2, 64), (2, 128), (2, 256), (2, 512)), len(train_dataset.classes))
    elif model_name == 'dense':
        model = DenseNet(len(train_dataset.classes))
    
    model.to(device)
        # need to upsample all the images
    # dummy forward
    batch, _  = next(iter(train_loader))
    batch = batch.to(device)
    with torch.no_grad():
        model(batch)
    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    print('Starting training')
    start_time = time.time()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        train_accuracy = []
        for new_batch, new_targets in train_loader:

            new_batch, one_hot_targets = new_batch.to(device), new_targets.to(device)

            outputs = model(new_batch)
            loss = F.cross_entropy(outputs, one_hot_targets)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            train_accuracy.append((preds == one_hot_targets).sum() / one_hot_targets.size(dim=0))
        # Validation
        acc = evaluate(test_loader, model)
        print(f"Validation Accuracy: {acc}")
        print(f"Train Accuracy: {sum(train_accuracy) / len(train_accuracy)}")
    print(f'Training time {time.time() - start_time}')
    
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