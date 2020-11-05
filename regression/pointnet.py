import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable


class STNkd(nn.Module):

    # input: input_point_clouds : (Batch_size * point_features(default 4) * point_numbers(default 29))
    # output: transform_matrix : (Batch_size * k * k)

    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(self.k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k * self.k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):

    # input: input_point_clouds : (Batch_size * point_features(default 4) * point_numbers(default 29))
    # output: transform_matrix : (Batch_size * k * k)

    def __init__(self):
        super().__init__()
        self.stn = STNkd(k=3)

        self.conv1 = nn.Conv1d(4, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.fstn = STNkd(k=64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        input_transform_matrix = self.stn(x[:, :3, :].clone())
        x[:, :3, :] = torch.bmm(
            x[:, :3, :].clone().transpose(2, 1), input_transform_matrix
        ).transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        feature_transform_matrix = self.fstn(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform_matrix).transpose(2, 1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, feature_transform_matrix


class PointNetCls(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat = PointNetfeat()

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x, feature_transform_matrix = self.feat(x)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, feature_transform_matrix

    def feature_transform_regularizer(self, feature_transform_matrix):
        d = feature_transform_matrix.size()[1]
        batchsize = feature_transform_matrix.size()[0]
        I = torch.eye(d)[None, :, :]
        if feature_transform_matrix.is_cuda:
            I = I.cuda()
        loss = torch.mean(
            torch.norm(
                torch.bmm(
                    feature_transform_matrix, feature_transform_matrix.transpose(2, 1)
                )
                - I,
                dim=(1, 2),
            )
        )
        return loss
