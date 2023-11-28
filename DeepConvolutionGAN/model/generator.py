import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels: int = 100, out_channels: int = 1):
        super(Generator, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv_5 = nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1, padding=2, bias=False)

        self.batchnorm_1 = nn.BatchNorm2d(512)
        self.batchnorm_2 = nn.BatchNorm2d(256)
        self.batchnorm_3 = nn.BatchNorm2d(128)
        self.batchnorm_4 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x : torch.Tensor):
        x = self.relu(self.batchnorm_1(self.deconv_1(x)))
        x = self.relu(self.batchnorm_2(self.deconv_2(x)))
        x = self.relu(self.batchnorm_3(self.deconv_3(x)))
        x = self.relu(self.batchnorm_4(self.deconv_4(x)))
        x = self.deconv_5(x)

        x = self.tanh(x)
        return x