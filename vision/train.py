from torchvision.datasets import MNIST
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import VAE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(num_epochs=1000, lr=1e-3):
    torch.manual_seed(42)

    train_mnist = MNIST(root='../data', train=True, download=True, transform=ToTensor())
    test_mnist = MNIST(root='../data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=64, shuffle=True)
    # model = Autoencoder(3, 16, 784).to(device)
    model = VAE(784, 16, 3, 1).to(device)
    # train _ loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    evaluate_intervals = 50
    mse = torch.nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        images, targets = next(iter(train_loader))
        images = torch.flatten(images, start_dim=2)
        images = torch.squeeze(images, 1)
        # print(images.shape, targets.shape)
        # need to reshape images from 1, (28, 28) to 1, 784
        outputs, mus, sigmas = model(images)

        loss = model.custom_loss(images, outputs, mus, sigmas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % evaluate_intervals == 0:
            # print(f'Epoch {epoch}/{num_epochs} Loss: {loss.item()}')
            images, targets = next(iter(test_loader))
            images = torch.flatten(images, start_dim=2)
            images = torch.squeeze(images, 1)
            outputs, mus, sigmas = model(images)
            loss = model.custom_loss(images, outputs, mus, sigmas)
            print(f'Epoch {epoch}/{num_epochs} Loss: {loss.item()}')
        # how do we know the dimensions?


    pass



if __name__ == '__main__':
    train_model()
