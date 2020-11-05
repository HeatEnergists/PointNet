import os

import numpy as np
import torch
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.notebook import tqdm

from dataset import MolecularDataset
from pointnet import PointNetCls


writer = SummaryWriter()

# hyper parameters
LR = 0.001
EPOCH = 500
BATCH_SIZE = 128

# load dataset
dataset = MolecularDataset(transform=transforms.ToTensor(), data_augmentation=True)
train_size = int(0.8 * len(dataset))
test_size = int(len(dataset) - train_size)
train_data, test_data = data.random_split(dataset, [train_size, test_size])

train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)

test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE,)

# define something about training...
mynet = PointNetCls()
optimizer = torch.optim.Adam(mynet.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
loss_func = torch.nn.MSELoss()

# train
myepoch = tqdm(range(1,500))
for epoch in myepoch:
    loss_list = []
    valid_loss_list = []
    for step, (features, targets) in enumerate(train_loader):
        mynet.cuda()
        mynet.train()
        features = features.transpose(2, 1)
        features, targets = features.cuda(), targets.cuda()
        predicted_targets, feature_transform_matrix = mynet(features)

        loss = loss_func(targets, predicted_targets)
        loss = (
            loss + mynet.feature_transform_regularizer(feature_transform_matrix) * 0.001
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.cpu().data.numpy())

    ave_loss = np.array(loss_list).mean()
    writer.add_scalar("loss", ave_loss, epoch)

    if epoch % 10 == 0:
        for step, (features, targets) in enumerate(test_loader):
            mynet.cpu()
            mynet.eval()
            features = features.transpose(2, 1)
            predicted_targets, feature_transform_matrix = mynet(features)
            valid_loss_list.append(loss_func(targets, predicted_targets).cpu().data.numpy())

        ave_valid_loss = np.array(valid_loss_list).mean()
        writer.add_scalar("valid_loss", ave_valid_loss, epoch)

    myepoch.set_description("loss:{:.2f}#####".format(ave_loss))
    scheduler.step()
mynet.eval()
torch.save(mynet, "mynet.pkl")

writer.close()

train_loss_list = []
for step, (features, targets) in enumerate(test_loader):
    features = features.transpose(2, 1)
    predicted_targets, feature_transform_matrix = mynet(features)
    train_loss_list.append((torch.abs(predicted_targets.data-targets)/targets*100).mean())

train_loss = -np.array(train_loss_list).mean()

valid_loss_list = []
for step, (features, targets) in enumerate(test_loader):
    features = features.transpose(2, 1)
    predicted_targets, feature_transform_matrix = mynet(features)
    valid_loss_list.append((torch.abs(predicted_targets.data-targets)/targets*100).mean())

valid_loss = -np.array(valid_loss_list).mean()

print('训练集误差：{:.4f}%'.format(train_loss))
print('测试集误差：{:.4f}%'.format(valid_loss))
