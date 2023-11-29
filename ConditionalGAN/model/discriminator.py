import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=50)
        self.label_linear = nn.Linear(50, 28*28)
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.last_linear = nn.Linear(128*7*7, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        emb_labels = self.label_emb(labels)
        linear_labels = self.label_linear(emb_labels)
        linear_labels = linear_labels.view(linear_labels.size(0), 1, 28, 28)
        x = torch.cat([x, linear_labels], 1)
        x = self.leaky_relu(self.batchnorm(self.conv_1(x)))
        x = self.leaky_relu((self.conv_2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = self.sigmoid(x)
        return x

#test     
# if __name__ == '__main__':
#     x = torch.randn(64, 1, 28, 28)
#     labels = torch.randint(0, 10, (64,))
#     D = Discriminator()
#     D(x, labels)