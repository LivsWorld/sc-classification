import torch.nn as nn
from torch import amax

CONV_NAMES = ['conv3', 'conv4']

class Conv3(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__()
        # input channels, output channels, kernel size, stride, padding
        self.layer1 = nn.Sequential(nn.Conv2d(3, c1, 7, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(c1),
                                    nn.AvgPool2d(2, 2))

        self.layer2 = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(c2),
                                    nn.AvgPool2d(2, 2))

        self.layer3 = nn.Sequential(nn.Conv2d(c2, c3, 3, 1, 1),
                                    nn.BatchNorm2d(c3),
                                    nn.ReLU())

        self.output_size = c3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Return global avg pool (average of each 2D feature channel)
        return x.mean([2,3])

class Conv4(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # input channels, output channels, kernel size, stride, padding
        self.layer1 = nn.Sequential(nn.Conv2d(3, c1, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(c1),
                                    nn.AvgPool2d(2, 2))

        self.layer2 = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(c2),
                                    nn.AvgPool2d(2, 2))

        self.layer3 = nn.Sequential(nn.Conv2d(c2, c3, 3, 1, 1),
                                    nn.BatchNorm2d(c3),
                                    nn.ReLU(),
                                    nn.AvgPool2d(2, 2))

        self.layer4 = nn.Sequential(nn.Conv2d(c3, c4, 3, 1, 1),
                                    nn.BatchNorm2d(c4),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Return global max pool (max of each 2D feature channel)
        return amax(x, (2,3))