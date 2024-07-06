from torchvision.datasets import MNIST
import torch
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FastMNIST(MNIST):
    """
    The base of this class is taken from GitHub Gist, then I adapted it to my needs.
    Author: Joost van Amersfoort (y0ast)
    link: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Since I'm working with autoencoders, 'targets' becomes a copy of the data. The labels are now stored
        # in the variable 'labels'. Also put everything on GPU in advance, since the dataset is small.
        # This lets me bypass PyTorch's DataLoaders and speed up the training.
        self.data = self.data.to(device)
        self.labels = self.targets.to(device)
        self.targets = torch.flatten(copy.deepcopy(self.data), start_dim=1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, label = self.data[index], self.targets[index], self.labels[index]
        return img, target, label
