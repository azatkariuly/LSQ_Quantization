import torch.nn as nn
import torch.nn.functional as F
from lsq import ActLSQ, Conv2dLSQ, LinearLSQ

class Net(nn.Module, nbits=8):
    def __init__(self):
        super(Net, self).__init__()
        self.act1 = ActLSQ()
        self.conv1 = Conv2dLSQ(in_channels=1, out_channels=20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu = nn.ReLU(inplace=True)
        self.act2 = ActLSQ()
        self.conv2 = Conv2dLSQ(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.act3 = ActLSQ()
        self.fc1 = LinearLSQ(4*4*50, 500)
        self.fc2 = LinearLSQ(500, 10)

    def forward(self, x):
        #Activation
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #Activation
        x = self.act2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #Activation
        x = self.act3(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
