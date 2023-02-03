import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import ceil
import torchvision.datasets as datasets
import torchvision.transforms as transforms
VGG_types = {
'VGG11': [64, 'M', 128, 'M', 256, 256,'M', 512, 512, "M", 512, 512, "M"],
'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,'M', 512, 512, 512, 512,'M']

}


class VGG_net(nn.Module):
    def __init__(self,in_shape, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        layer3_shape = ceil(in_shape/32)
        self.flex_pool3 = nn.Parameter(torch.ones(512, layer3_shape, layer3_shape) / layer3_shape ** 2) ## Flexpooling where we define the matrix and divide by shape*2
        
        self.fcs = nn.Sequential(
            nn.Linear(512 , 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
      #  x = x.reshape(x.shape[0], -1)## Here we are flattening h,w,d into one long vector just keeping the batch size
       # x = F.avg_pool2d(x, x.size(2)) ##bs,depth,1
        
       # x = x.view(-1,x.size(1))### chaning view to reduce one dimension     shape(1*512)
       
       
        fp = self.flex_pool3.view(512, -1).softmax(1).view(self.flex_pool3.shape)  # FlexPool weights : shape of fp is 512*7*7
        out = (x * fp).sum((2, 3))  # (BS, 512)
        x = self.fcs(out)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))] ## reduce the shape by /2
        return nn.Sequential(*layers)
#device = 'cuda' if torch.cuda.is_available else 'cpu'

