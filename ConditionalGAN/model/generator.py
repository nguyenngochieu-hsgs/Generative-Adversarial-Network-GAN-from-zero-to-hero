import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim: int = 100, num_classes: int = 10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=50)
        self.label_linear = nn.Linear(50, 49)
        self.linear_input = nn.Linear(input_dim, 64*7*7)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=65, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.last_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.batchnorm_1 = nn.BatchNorm2d(32)
        self.batchnorm_2 = nn.BatchNorm2d(16)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        emb_labels = self.label_emb(labels)
        linear_labels = self.leaky_relu(self.label_linear(emb_labels))
        linear_labels = linear_labels.view(linear_labels.size(0), 1, 7, 7)
        x = self.leaky_relu(self.linear_input(x))
        x = x.view(x.size(0), 64, 7, 7) 
        x = torch.cat([x, linear_labels], 1)
        x = self.leaky_relu(self.batchnorm_1(self.deconv_1(x)))
        x = self.leaky_relu(self.batchnorm_2(self.deconv_2(x)))
        x = self.last_conv(x)
        x = self.tanh(x)
        return x

#test
# if __name__ == '__main__':
#     labels = torch.randint(0, 10, (64,))
#     x = torch.randn(64, 100)
#     G = Generator(100, 10)
#     out_g = G(x, labels)
#     print(out_g.size())
