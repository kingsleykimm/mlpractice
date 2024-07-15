import torch
import torch.nn as nn
import torch.functional as F
from cnns import VGGNet
from utils import init_weights, BatchNorm

"""ResNet originated from the idea of thinking of neural networks as function classes. If we were searching for the function
f*, which perfectly maps the features X -> labels y, usually neural networks will find the best approximation by minimizing a loss function.
One of the main ideas of deep neural networks is that by expanding the reach of our function classes, i.e by adding parameters/layers, we would get closer
and closer to the true function. However, there were two types of function classes, nested and non-nested. Non-nested function classes are bad because
there is no gaurantee was are getting closer to the function class, since we can't know if we even contain the previous one, i.e the funciton class
could go on a random walk. Nested function classes are good because we are essentially creating bigger and bigger rings around the starting function class,
guaranteeing that we are 'covering' more function space.

This is where Residual Networks come in, because we want to ensure that each new layer can repliacte the identity function, i.e it can replicate its previous nested
function class. """

class ResidualBlock(nn.Module):
    def __init__(self, out_channels : int, stride=1, change_channels : bool = False):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels, num_dim=4),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels, num_dim=4),
            nn.ReLU(),
        )
        self.change_channels = change_channels
        # if we want to use different out_channels from input's channels, do a 1x1 conv (not included here)
        if change_channels:
            self.change_conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=1, stride=stride)
        self.apply(init_weights)
    def forward(self, inp):
        residual = inp
        g_x = self.net(inp)
        if self.change_channels:
            residual = self.change_conv(residual)
        g_x += residual
        return F.relu(g_x)




class ResNet(nn.Module): # ResNet-18 arch: (2, 64), (2, 128), (2, 256), (2, 512)
    def __init__(self, arch : tuple[int], num_classes : int): # arch only describes the 
        
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNorm(num_features=64, num_dim=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # to create one of these
        resblocks = []
        for ind, (num_blocks, num_channels) in enumerate(arch):
            if ind == 0:
                resblocks.append(self.create_resnet_block(num_blocks, num_channels, True))
            else:
                resblocks.append(self.create_resnet_block(num_blocks, num_channels, False))
        self.body = nn.Sequential(*resblocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
        self.net = nn.Sequential(
            self.stem, self.body, self.head
        )
        self.apply(init_weights)
    def create_resnet_block(self, num_blocks, out_channels, first_block):
        blocks = []
        for i in range(num_blocks):
            if i == 0 and not first_block: # number of channels is doubled and height and width are halved
                blocks.append(ResidualBlock(out_channels, stride=2, change_channels=True))
            else:
                blocks.append(ResidualBlock(out_channels))
        return nn.Sequential(*blocks)
    def forward(self, inp):
        inp = self.net(inp)
        return inp
    
class ResNextBlock(nn.Module):
    def __init__(self, mid_channels : int, out_channels : int, groups : int, strides, use_channel : bool):
        super(ResNextBlock, self).__init__()
        # break into g groups
        self.body = nn.Sequential(
            nn.LazyConv2d(out_channels=mid_channels, kernel_size=1, stride=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(out_channels=mid_channels, kernel_size=3, padding=1, stride=strides, groups=groups),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            nn.LazyBatchNorm2d(), nn.ReLU()
        )
        self.use_channel = use_channel
        if use_channel:
            self.reformatter = nn.Sequential(
                nn.LazyConv2d(out_channels, kernel_size=1, stride=strides),
                nn.LazyBatchNorm2d(), nn.ReLU()
            )

    def forward(self, img):
        residual = img
        img = self.body(img)
        if self.use_channel:
            residual = self.reformatter(residual)
        return F.relu(residual + img)