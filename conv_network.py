import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3)
        self.dropout3 = nn.Dropout2d(0.3)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3)
        self.dropout4 = nn.Dropout2d(0.4)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(256*7*7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, t):
        t = F.relu(self.bn1(self.dropout1(self.conv1(t))))
        t = F.relu(self.bn2(self.dropout2(self.conv2(t))))
        t = self.pool1(t)
        t = F.relu(self.bn3(self.dropout3(self.conv3(t))))
        t = F.relu(self.bn4(self.dropout4(self.conv4(t))))
        t = self.pool2(t)
        t = t.view(-1, 256*7*7)
        t = F.relu(self.bn5(self.fc1(t)))
        t = torch.sigmoid(self.fc2(t))
        return t
