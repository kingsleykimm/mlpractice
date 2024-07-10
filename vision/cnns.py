import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CNN Notes:
Kernel = moving filter
Padding = how much to help the filter capture features of entire image
Stride = how much to move the filter step
Channels = more descriptive features of the image, kernal_channels == input_image_channels 
Kernel sizes with channels -> (c_i, c_o, k_h, k_w), since each input channel needs to map to every single output channel. When cross-correlation operations are done, each output channel
will sum up the contribution from each of the input channels

"""
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # padding = 2 because (kernel_size - 1) / 2
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
            nn.Softmax()
        )
    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.flatten(img)
        img = self.fc_layers(img)
        return img
    def batch_loss(self, y_hat, y):
        # feed it in, get the outputs
        # across the entire batch, get the 
        log_probs = torch.log(y_hat)
        loss = y * log_probs
        sum_loss_per = loss.sum(dim=1) # sums the losses for each tensor in the minibatch of length num_classes
        return -1 * (sum_loss_per.sum() / loss.size(dim=0))

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        self.upsampler = nn.Upsample(size=(224, 224))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(),
            nn.LazyLinear(4096),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.Dropout(),
            nn.LazyLinear(num_classes),
        )
        # remember to put the softmax at the end
    def forward(self, imgs):
        imgs = self.upsampler(imgs)
        imgs = self.conv1(imgs)
        imgs = self.conv2(imgs)
        imgs = self.conv3(imgs)
        imgs = self.conv4(imgs)
        imgs = self.conv5(imgs)
        imgs_linear = self.fc_layers(imgs)
        return F.softmax(imgs_linear, dim=1) # because of batch_size
    def batch_loss(self, y_hat, y):
        # feed it in, get the outputs
        # across the entire batch, get the 
        log_probs = torch.log(y_hat)
        loss = y * log_probs
        sum_loss_per = loss.sum(dim=1) # sums the losses for each tensor in the minibatch of length num_classes
        return -1 * (sum_loss_per.sum() / loss.size(dim=0))
    

class VGGNet(nn.Module): # idea to chain together multiple convolutional blocks between downsampling to avoid the log limit, keep the channels the same
    # original VGGNet arch: ((1,64), (1, 128), (2, 256), (2, 512), (2, 512)) - VGG-11, 8 conv
    # our model will use: (1, 16), (1, 32), (2, 64), (2, 128), (2, 128) - VGG-11 with smaller out_channels
    def __init__(self, num_classes : int, arch, mnist : bool = False):
        self.using_mnist = mnist
        layers = []
        if mnist: # need to upsample for mnist images to fit original 224 x 224 Imagenet models
            layers.append(nn.Upsample(size=(224, 224)))

        for num_convs, out_channels in arch:
            for i in range(num_convs):
                layers.append(nn.LazyConv2d(out_channels=out_channels))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(
            *layers, 
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(num_classes), nn.ReLU()
        )
    def forward(self, imgs):
        return self.net(imgs)
    def batch_loss(self, y_hat, y):
        # feed it in, get the outputs
        # across the entire batch, get the 
        log_probs = torch.log(y_hat)
        loss = y * log_probs
        sum_loss_per = loss.sum(dim=1) # sums the losses for each tensor in the minibatch of length num_classes
        return -1 * (sum_loss_per.sum() / loss.size(dim=0))

class NetworkInNetwork(nn.Module):
    """ NiN's quirk is that it is removing the need for the fully connected layers at the end after the convolutions.
    Instead, to add nonlinearity and an element of FC layers, they did 1x1 convolutions, effectively emulating the effect
    of a FC layer on each pixel, and then a global average pooling.
    """
    def __init__(self, num_classes : int, mnist : bool = True):
        super(NetworkInNetwork, self).__init__()
        self.layers = []
        if mnist:
            self.layers.append(nn.Upsample((224, 224)))
        self.layers.extend(list(self.nin_block(11, 96, 4)))
        self.layers.append(nn.MaxPool2d(3, 2))
        self.layers.extend(list(self.nin_block(5, 256, padding=2)))
        self.layers.append(nn.MaxPool2d(3, 2))
        self.layers.extend(list(self.nin_block(3, 384, padding=1)))
        self.layers.append(nn.MaxPool2d(3, 2))
        self.layers.extend(list(self.nin_block(3, num_classes, padding=1)))
        self.layers.append(nn.AvgPool2d(num_classes))
        self.net = nn.Sequential(
            *self.layers
        )
        self.apply(init_weights)

    
    def nin_block(self, kernel_size, out_channels, stride=1, padding=1):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1), nn.ReLU(),
        )
    def forward(self, imgs):
        return self.net(imgs)
    def batch_loss(self, y_hat, y):
        # feed it in, get the outputs
        # across the entire batch, get the 
        log_probs = torch.log(y_hat)
        loss = y * log_probs
        sum_loss_per = loss.sum(dim=1) # sums the losses for each tensor in the minibatch of length num_classes
        return -1 * (sum_loss_per.sum() / loss.size(dim=0))

class GoogleLeNet(nn.Module):
    """The first model to emphasize the use of a stem (data ingest / mainly low level features), a body (with multiple convolutions),
    and a head (prediction) in a CNN"""
    def __init__(self, num_classes : int):
        super(GoogleLeNet, self).__init__()
        self.stem = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # padding for kernel_size - 1 // 2, and stride is to reduce dimensionality by 2
        )
        self.body = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.head = nn.Sequential(
            InceptionBlock(64, (96, 128), (16, 32), 32), # we need to downsample the second and third branches because the in_channels=192, so 192->96->128
            # out_channels is 256
            InceptionBlock(128, (128, 192), (32, 96), 64),
            #out_channels = 480
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            InceptionBlock(192, (96, 208), (16, 48), 64),
            InceptionBlock(160, (112, 224), (24, 64), 64),
            InceptionBlock(128, (128, 256), (24, 64), 64),
            InceptionBlock(112, (144, 288), (32, 64), 64),
            InceptionBlock(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # fifth module
            InceptionBlock(256, (160, 320), (32, 128), 128),
            InceptionBlock(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), # flattens all the output channels
            nn.LazyLinear(num_classes)
        )
        self.net = nn.Sequential(self.stem, self.body, self.head)
        self.apply(init_weights)
    def forward(self, imgs):
        return self.net(imgs)
    # can implement CE loss later
class InceptionBlock(nn.Module):
    """Inception block used in GoogleLeNet"""
    def __init__(self, c0, c1, c2, c3, **kwargs): # use the channels of each of the four parallel branches
        super(InceptionBlock, self).__init__(**kwargs)
        self.branch1 = nn.Sequential(nn.LazyConv2d(c0, kernel_size=1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(c1[0], kernel_size=1),
            nn.LazyConv2d(c1[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(c2[0], 1),
            nn.LazyConv2d(c2[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.LazyConv2d(c3, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, imgs):
        # with respect to batching, we will cat along dim=1
        return torch.cat(
            [self.branch1(imgs), self.branch2(imgs), self.branch3(imgs), self.branch4(imgs)], dim=1
        )

class BatchNorm(nn.Module):
    """Batch Norm between training and predicting is different.
    During training, it only uses the statistics of the minibatches, while during """
    def __init__(self, input_size : tuple): # input_size will be a tuple
        super(BatchNorm, self).__init__()
        # if len(input_size) == 2:
        #     input_size = (1, input_size[1])
        # elif len(input_size) == 4:
        #     input_size = (1, input_size[1], 1, 1) # for broadcasting just make it fit to the number of channels
        self.gamma = nn.Parameter(torch.zeros(*input_size))
        self.beta = nn.Parameter(torch.ones(*input_size))
        self.moving_avg, self.moving_var = torch.zeros(*input_size), torch.ones(*input_size)
        self.epsilon = 1e5
        self.momentum = 0.1
    def forward(self, x, predicting):
        return self.batch_norm(self.epsilon, x, self.gamma, self.beta, predicting, self.momentum)

    def batch_norm(self, epsilon, batch, gamma, beta, predicting, momentum):
        # gamma and beta are the same size as element in batch
        # so if we
        if predicting:
            # if we're predicting, we're doing it over the entire dataset
            return gamma * (batch - self.moving_avg) / torch.sqrt(self.moving_var + epsilon) + beta
        # if we're not in predicting (i.e training), we're going to want to just find the means across the batch_size dim
        else:
            mean, var = None, None
            if len(batch.size) == 2: # i.e fc, (batch_size, fc_layer)
                mean = torch.mean(batch, dim=0)
                var = ((batch - mean) ** 2).mean(dim=0) # variance per batch row
                # apply batch norm equation
            elif len(batch.size) == 4: # i.e conv (batch_size, num_channels, height, width)
                mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True) # now mean will be of size (1, channels, 1, 1)
                var = torch.mean((batch - mean) ** 2, dim=(0, 2, 3), keepdim=True) # broadcasting allows it to just subtract
            self.moving_avg = self.moving_avg * (1 - momentum) + mean * momentum
            self.moving_var = self.moving_var * (1 - momentum) + var * momentum
            batch_normalized = gamma * (batch - mean) / torch.sqrt(var + epsilon) + beta
            return batch_normalized
        
class BatchLeNet(nn.Module):
    def __init__(self, num_classes):
        super(BatchLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # padding = 2 because (kernel_size - 1) / 2
            BatchNorm((1, 6, 1, 1)),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
            nn.Softmax()
        )
    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.flatten(img)
        img = self.fc_layers(img)
        return img