import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
from avg.resnet import resnet20
# from resnet_softflexpool import resnet20
from torchvision.datasets import ImageFolder
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


root_dir = '/l/users/muhammad.ali/Flexpool_waste/dataset-original'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = ImageFolder(root=f'{root_dir}')
train_size = int(len(data)*0.9)
test_size = len(data) - train_size


T = transforms.Compose([transforms.Resize((244, 244)), transforms.ToTensor(), normalize])
train_subset, test_subset = torch.utils.data.random_split(data, [train_size, test_size])

train_data = DatasetFromSubset(train_subset, transform=transforms.Compose([transforms.RandomHorizontalFlip(), T]))
test_data = DatasetFromSubset(test_subset, transform=T)

train_loader = DataLoader(train_data, 10, shuffle = True)
for x,y in train_loader:
    print(x,y)
    break