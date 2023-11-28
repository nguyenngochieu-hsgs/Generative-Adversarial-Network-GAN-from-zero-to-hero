import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True)

        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor):
        x = self.leaky_relu(self.batchnorm_1(self.conv_1(x)))
        x = self.leaky_relu(self.batchnorm_2(self.conv_2(x)))
        x = self.leaky_relu(self.batchnorm_3(self.conv_3(x)))
        x = self.conv_4(x)
        x = self.sigmoid(x)
        return x.view(-1, 1)
    