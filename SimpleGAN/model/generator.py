import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.fc_3 = nn.Linear(256, 512)
        self.fc_4 = nn.Linear(512, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor):
        x = self.leaky_relu(self.fc_1(x))
        x = self.leaky_relu(self.fc_2(x))
        x = self.leaky_relu(self.fc_3(x))
        x = torch.tanh(self.fc_4(x))
        return x


