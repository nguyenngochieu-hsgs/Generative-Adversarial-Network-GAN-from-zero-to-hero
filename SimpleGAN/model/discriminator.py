import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super(Discriminator, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256,128)
        self.fc_4 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor):
        x = self.leaky_relu(self.fc_1(x))
        x = self.leaky_relu(self.fc_2(x))
        x = self.leaky_relu(self.fc_3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc_4(x))
        return x