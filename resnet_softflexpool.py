
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from math import ceil


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

##### Addition of FlexPool ################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_dims, out_dims, stride=1, option='A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dims)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dims)

        self.shortcut = nn.Sequential()

        # If shape was not preserved (not the same shape as input feature map) reduce the identity's (x) shape to match that of the processed units:
        if stride != 1 or in_dims != out_dims:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_dims // 4, out_dims // 4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dims, self.expansion * out_dims, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * out_dims))

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.bn2(F.relu(out))


# Apply softmax function on flexpool weights (self.flex_pool3) before multiplying it with the last feature map, then sum accross height x width to get a single pixel output
# After we applied softmax on flexpool weights (self.flex_pool3), now we need to multiply them with the last featuremap (out)
# After you multiply (element wise), you must add the last featuremap accros height and width


class ResNet(nn.Module):
    def __init__(self, in_shape, in_dims, num_blocks, num_classes):
        super().__init__()

        # Initialize relevant parameters and convolutional residual blocks:
        self.initial_dims = 16
        self.bn1 = nn.BatchNorm2d(self.initial_dims)
        self.conv1 = nn.Conv2d(in_dims, self.initial_dims, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        layer3_shape = ceil(in_shape / 4)
        self.flex_pool3 = nn.Parameter(torch.ones(64, layer3_shape, layer3_shape) / layer3_shape ** 2)
        self.linear3 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, out_dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.initial_dims, out_dims, stride))
            self.initial_dims = out_dims * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.layer1(out)  # (BS, 16, 32, 32)
        out = self.layer2(out)  # (BS, 32, 16, 16)
        out = self.layer3(out)  # (BS, 64, 8, 8)  (last feature map)
        fp = self.flex_pool3.view(64, -1).softmax(1).view(self.flex_pool3.shape)  # FlexPool weights
        import pdb;
        pdb.set_trace
        out = (out * fp).sum((2, 3))  # (BS, 64)
        return self.linear3(out)


def resnet20(inshape, in_dims=3, num_classes=10):
    return ResNet(inshape, in_dims, [3, 3, 3], num_classes)  # 3 stages, with 3 res-blocks per stage
