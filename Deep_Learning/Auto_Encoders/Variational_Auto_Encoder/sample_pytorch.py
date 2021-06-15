import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

n_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 10
channels = 1
sample_interval = 400
img_size = 32
img_shape = (channels, img_size, img_size)

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

if __name__ == "__main__":
    os.makedirs("../../data", exist_ok=True)
    os.makedirs("sample_pytorch_output", exist_ok=True)

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST( 
            "../../data", 
            train=True, 
            download=True,
            transform=transforms.Compose( [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])] ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits"""
        # Sample noise
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
        gen_imgs = decoder(z)
        save_image(gen_imgs.data, f"./sample_pytorch_output/image_{batches_done}.png", nrow=n_row, normalize=True)


    # ----------
    #  Training
    # ----------

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(f"[{epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

# import argparse
# import torch
# import os
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image

# torch.manual_seed(1)

# device = "cpu"
# batch_size = 128
# epochs = 10
# log_interval = 10

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor()), 
#     batch_size=batch_size, 
#     shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data', train=False, transform=transforms.ToTensor()), 
#     batch_size=batch_size, 
#     shuffle=True
# )

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD


# def train(epoch, epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):

#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()

#         if batch_idx % log_interval == 0:
#             data_now = (batch_idx * len(data))
#             data_total = len(train_loader.dataset)
#             procent_passed = (100. * batch_idx / len(train_loader))
#             loss_value = (loss.item() / len(data))
#             print(f"\r[{epoch}/{epochs}] [{data_now}/{data_total}] \t loss: {loss_value}", end="")
#     print(f"  ====> Epoch: {epoch} Average loss: {(train_loss / len(train_loader.dataset))}")

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(), './sample_pytorch_output/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss}')

# def make_dir():
#     image_dir = "./sample_pytorch_output"
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

# make_dir()
# for epoch in range(1, epochs + 1):
#     train(epoch, epochs)
#     test(epoch)
#     with torch.no_grad():
#         sample = torch.randn(64, 20).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28), './sample_pytorch_output/sample_' + str(epoch) + '.png')






