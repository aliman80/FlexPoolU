
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
from avg.resnet import resnet20
# from resnet_softflexpool import resnet20
from torchvision.datasets import ImageFolder
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm
# from torchvision.models import resnet18t

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


run_name = 'new_split wastedata avg_32bsize'
logging.basicConfig(filename=f"/l/users/muhammad.ali/Flexpool_waste/logs/{run_name}.log", format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
print(f'This is {run_name}')
logger = logging.getLogger()
logger.info(f'This is {run_name}\n')

device = torch.device('cuda')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.cuda.empty_cache()
# testing 
root_dir = '/l/users/muhammad.ali/Flexpool_waste/dataset-original'
data = ImageFolder(root=f'{root_dir}')
train_size = int(len(data)*0.9)
test_size = len(data) - train_size

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.Resize((244, 244)), transforms.ToTensor(), normalize])
train_subset, test_subset = torch.utils.data.random_split(data, [train_size, test_size])

train_data = DatasetFromSubset(train_subset, transform=transforms.Compose([transforms.RandomHorizontalFlip(), T]))
test_data = DatasetFromSubset(test_subset, transform=T)


logger.info(f'\nDataset: {train_data}\n')

BS = 32
EPOCHS = 100
LR = 0.05

train_loader = DataLoader(train_data, BS, shuffle = True, num_workers=8)
val_loader = DataLoader(test_data, BS, shuffle = False , num_workers= 8 )

logger.info(f'\nTraining parameters out:\nBS={BS}, EPOCHS={EPOCHS}, LR={LR}\n')


m = resnet20(num_classes=6).to(device)
opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), base_momentum=0.9)


for _ in trange(EPOCHS):
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

m.eval()
acc = []
with torch.no_grad():
    for x, y in tqdm(val_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = m(x)
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())

accuracy = 100 * sum(acc) / len(acc)
print(f'\nAccuracy: {accuracy:.4}%')
logger.info(f'Accuracy: {accuracy:.4}%\n')
logger.info(f'\nModel: {m}')
torch.save(m.state_dict(), "/l/users/muhammad.ali/Flexpool_waste/weights/weights_avg_pool")
m = m.to(torch.device('cpu'))

class_names =["cardboard", "glass" , "metal", "paper" , "plastic", "trash"]

for e, batch in enumerate(tqdm(val_loader)):
    images, y = batch  # (BS, 3, 244, 244)
    out = m(images)  # (BS, 6)
    prob, pred = out.softmax(1).max(1)
    for i, img in enumerate(images):
        img = img.moveaxis(0, -1)
        plt.imshow(img)
        if pred[i] == y[i]:
            plt.xlabel(f"{class_names[pred[i]]} with {prob[i]*100:.3}% probability", color='green')
        else:
            plt.xlabel(f"{class_names[pred[i]]} with {prob[i]*100:.3}% probability", color='red')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"avg_images_new_bigl/img-{e}-{i}.png")
