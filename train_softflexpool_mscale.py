
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
# from avg.resnet import resnet20
from resnet_mscale import resnet20
from torchvision.datasets import ImageFolder
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
# from torchvision.models import resnet18


run_name = 'wastedata flexpool_mscale soft'
logging.basicConfig(filename=f"/l/users/muhammad.ali/Flexpool_waste/logs/{run_name}.log", format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
print(f'This is {run_name}')
logger = logging.getLogger()
logger.info(f'This is {run_name}\n')

device = torch.device("cuda")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.cuda.empty_cache()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((244, 244)), transforms.ToTensor(), normalize])
T_val = transforms.Compose([transforms.Resize((244, 244)), transforms.ToTensor(), normalize])
logger.info(f'\nTrain_transformations: {T_train}\nTest_transformations: {T_val}\n')

# root_dir = '/l/users/muhammad.ali/imagenet-1k'
root_dir = '/l/users/muhammad.ali/Flexpool_waste/dataset-original'

# print(len(os.listdir(f"{root_dir}/val")))
# train_data = datasets.ImageNet(root=f'{root_dir}', transform=T_train, download = False)
# val_data = datasets.ImageNet(root=f'{root_dir}', transform=T_val, train =  False)
train_data = ImageFolder(root=f'{root_dir}', transform=T_train)
val_data = ImageFolder(root=f'{root_dir}', transform=T_val)

logger.info(f'\nDataset: {train_data}\n')

BS = 16
EPOCHS = 50 
LR = 0.05

logger.info(f'\nTraining parameters out:\nBS={BS}, EPOCHS={EPOCHS}, LR={LR}\n')
train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=BS, num_workers=0)

m = resnet20(244, num_classes=6).to(device)
opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), base_momentum=0.9)


for _ in trange(EPOCHS):
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        o1, o2, o3 = m(x)
        # Try: 0.05, 0.1, 0.85
        loss = 0.01*F.cross_entropy(o1, y) + 0.11*F.cross_entropy(o2, y) +  0.88*F.cross_entropy(o3, y)
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
torch.save(m.state_dict(), "/l/users/muhammad.ali/Flexpool_waste/weights/weights_flexpool_mscale")
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
        plt.savefig(f"flex_mscale_imgs/img-{e}-{i}.png")
