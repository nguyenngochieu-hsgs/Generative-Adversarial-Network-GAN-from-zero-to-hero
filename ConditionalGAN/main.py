import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.generator import Generator
from model.discriminator import Discriminator
from torchvision.utils import save_image


def G_train(G: Generator, D:Discriminator, real_input: torch.Tensor, z_dim: int, loss: nn.Module, g_optimizer: torch.optim.Optimizer, device=torch.device):
    G.zero_grad()
    random_noise_z = Variable(torch.randn(real_input.size(0), z_dim)).to(device)
    fake_labels = Variable(torch.LongTensor(torch.randint(0, 10, (real_input.size(0),)))).to(device)
    g_output = G(random_noise_z, fake_labels)
    d_output = D(g_output, fake_labels) 
    
    #loss
    y_fake = Variable(torch.ones(real_input.size(0), 1)).cuda()
    g_loss = loss(d_output, y_fake)
    
    #backprop
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


def D_train(G: Generator, D:Discriminator, real_input: torch.Tensor, real_label: torch.Tensor, z_dim: int, loss: nn.Module, d_optimzier: torch.optim.Optimizer, device=torch.device):
    D.zero_grad()
    real_input = real_input.to(device)
    real_label = torch.LongTensor(real_label).to(device)
    y_real = Variable(torch.ones(real_input.size(0), 1)).to(device)
    d_output_real = D(real_input, real_label)
    d_real_loss = loss(d_output_real, y_real)

    #discriminator on fake
    random_noise_z = Variable(torch.randn(real_input.size(0), z_dim)).to(device)
    fake_labels = Variable(torch.LongTensor(torch.randint(0, 10, (real_input.size(0),)))).to(device)
    fake_input = G(random_noise_z, fake_labels)
    y_fake = Variable(torch.zeros(real_input.size(0), 1)).to(device)
    d_output_fake = D(fake_input, fake_labels)
    d_fake_loss = loss(d_output_fake, y_fake)

    #loss
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()

if __name__ == '__main__':
    #Config
    batchsize = 64
    learning_rate = 2e-4
    z_dim = 100
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE : ", device)

    #Get dataset
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batchsize, shuffle=True)

    #Init Model
    G = Generator(input_dim=z_dim, num_classes=10).to(device)
    D = Discriminator(in_channels=1, num_classes=10).to(device)

    #Loss and optimizer
    loss = nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    #Train
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for i, (real_input, real_label) in enumerate(dataloader):
            d_losses.append(D_train(G, D, real_input, real_label, z_dim=z_dim, loss=loss, d_optimzier=d_optimizer, device=device))
            g_losses.append(G_train(G, D, real_input, z_dim=z_dim, loss=loss, g_optimizer=g_optimizer, device=device))
        
        with torch.no_grad():
            g_loss = torch.mean(torch.FloatTensor(g_losses))
            d_loss = torch.mean(torch.FloatTensor(d_losses))
            print("Epoch {} / {} : d_loss: {}, g_loss: {}".format(epoch+1, epochs, d_loss, g_loss))
            test_z = torch.randn(100, z_dim).to(device)
            test_labels = torch.LongTensor(torch.arange(0, 10, 1).repeat(10)).to(device)
            fake_x = G(test_z, test_labels)
            if not os.path.exists("./sample_imgs"):
                os.makedirs("./sample_imgs")
            save_image(fake_x.view(fake_x.size(0),1 ,28, 28), "./sample_imgs/test_sample_at_epoch_{}.png".format(epoch+1))
            