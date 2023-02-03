
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import numpy as np
# from avg.resnet import resnet20
#from resnet_softflexpool import resnet20
from vgg_net.vgg_flexpool import VGG_net
from torchvision.datasets import ImageFolder
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
from tqdm import tqdm
# from torchvision.models import resnet18

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


id = 2
run_name = f'soft_VGG_0.05_fp_run_{id}'
logging.basicConfig(filename=f"/l/users/muhammad.ali/Flexpool_waste/logs/{run_name}.log", format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
print(f'This is {run_name}')
logger = logging.getLogger()
logger.info(f'This is {run_name}\n')

device = torch.device('cuda')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.cuda.empty_cache()
root_dir = '/l/users/muhammad.ali/Flexpool_waste/dataset-original'
data = ImageFolder(root=f'{root_dir}')
train_size = int(len(data)*0.9)
test_size = len(data) - train_size
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.Resize((244,244)),transforms.ToTensor(), normalize])
train_subset, test_subset = torch.utils.data.random_split(data, [train_size, test_size])
train_data = DatasetFromSubset(train_subset, transform=transforms.Compose([transforms.RandomHorizontalFlip(), T]))
test_data = DatasetFromSubset(test_subset, transform=T)
print(len(train_data))
logger.info(f'\nDataset: {train_data}\n')

BS = 16
EPOCHS = 100
LR = 0.05

logger.info(f'\nTraining parameters out:\nBS={BS}, EPOCHS={EPOCHS}, LR={LR}\n')
train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=8)
val_loader = DataLoader(test_data, batch_size=BS, num_workers=0)

#m = resnet20(244, num_classes=6).to(device)
m = VGG_net(224, 3 ,6).to(device)
opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))


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
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = m(x)
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())
        
accuracy = 100 * sum(acc) / len(acc)
print(f'\nAccuracy: {accuracy:.4}%')
logger.info(f'Accuracy: {accuracy:.4}%\n')
logger.info(f'\nModel: {m}')
torch.save(m.state_dict(), f"/l/users/muhammad.ali/Flexpool_waste/weights/weights_flex_pool_{id}")
m = m.to(torch.device('cpu'))
class_names =["Cardboard", "Glass" , "Metal", "Paper" , "Plastic", "Trash"]

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
        # plt.savefig(f"images_{id}/img-{e}-{i}.png")
        plt.savefig(f"fpool_images_vgg-16/img-{e}-{i}.png")
