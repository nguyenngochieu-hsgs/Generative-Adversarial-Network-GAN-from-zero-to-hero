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
    y_fake = Variable(torch.ones(real_input.size(0),1)).to(device)
    g_output = G(random_noise_z)
    d_output = D(g_output) 
    
    #loss
    g_loss = loss(d_output, y_fake)
    
    #backprop
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

def D_train(G: Generator, D:Discriminator, real_input: torch.Tensor, flat_dim: int, z_dim: int, loss: nn.Module, d_optimzier: torch.optim.Optimizer, device=torch.device):
    D.zero_grad()
    real_input = real_input.view(-1, flat_dim).to(device)
    y_real = torch.ones(real_input.size(0), 1).to(device)
    d_output_real = D(real_input)
    d_real_loss = loss(d_output_real, y_real)

    #discriminator on fake
    random_noise_z = Variable(torch.randn(real_input.size(0), z_dim)).to(device)
    fake_input = G(random_noise_z)
    y_fake = Variable(torch.zeros(real_input.size(0), 1)).to(device)
    d_output_fake = D(fake_input)
    d_fake_loss = loss(d_output_fake, y_fake)

    #loss
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()


if __name__ == '__main__':
    #Config
    batchsize = 64
    z_dim = 100
    learning_rate = 0.0002
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE : ", device)


    # Get Dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5),
            std=(0.5)
        )
    ])

    train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
    #Init model
    mnist_flat_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2) # 28 * 28
    G = Generator(input_dim = z_dim, output_dim=mnist_flat_dim).to(device)
    D = Discriminator(input_dim=mnist_flat_dim).to(device)

    #Loss
    loss = nn.BCELoss()
    g_loss_divide_d_loss = 100

    #Optimizer
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for i, (real_input, _) in enumerate(train_loader):
            d_losses.append(D_train(G, D, real_input, flat_dim=mnist_flat_dim, z_dim=z_dim, loss=loss, d_optimzier=d_optimizer, device=device))
            g_losses.append(G_train(G, D, real_input, z_dim=z_dim, loss=loss, g_optimizer=g_optimizer, device=device))
        
        g_loss = torch.mean(torch.FloatTensor(g_losses))
        d_loss = torch.mean(torch.FloatTensor(d_losses))
        print("Epoch {} / {} : d_loss: {}, g_loss: {}".format(epoch+1, epochs, d_loss, g_loss))
        if g_loss / (d_loss+1e-8) < g_loss_divide_d_loss:
            g_loss_divide_d_loss = g_loss / (d_loss+1e-8)
            torch.save(D.state_dict(), "D_at_{}.pth".format(epoch+1))
            torch.save(G.state_dict(), "G_at_{}.pth".format(epoch+1))


    
    