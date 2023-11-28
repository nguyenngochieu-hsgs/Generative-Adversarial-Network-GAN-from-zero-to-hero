import torch
from model.generator import Generator
from model.discriminator import Discriminator
from torchvision.utils import save_image

z_dim = 100
G = Generator(input_dim = z_dim, output_dim=784)
D = Discriminator(input_dim=784)
G.load_state_dict(torch.load("G_at_77.pth"))
D.load_state_dict(torch.load("D_at_77.pth"))

with torch.no_grad():
    test_z = torch.randn(50, z_dim)
    fake_x = G(test_z)
    save_image(fake_x.view(fake_x.size(0),1 ,28, 28), "test_sample.png")