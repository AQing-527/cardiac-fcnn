import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.fc1_1 = nn.Linear(32 * 8 * 8 * 8, 64)
        self.fc1_2 = nn.Linear(64, 3)
        self.fc2_1 = nn.Linear(32 * 8 * 8 * 8, 64)
        self.fc2_2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = torch.flatten(x, start_dim=1)
        x1 = F.elu(self.fc1_1(x))
        x1 = self.fc1_2(x1)
        x2 = F.elu(self.fc2_1(x))
        x2 = torch.sigmoid(self.fc2_2(x2))
        return x1, x2


fcnn = FCNN()