import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 20
batch_size = 128
learning_rate = 1e-3


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        noisy_img = add_noise(img)
        noisy_img = Variable(noisy_img).cuda()
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(noisy_img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        x_noisy = to_img(noisy_img.cpu().data)
        weights = to_img(model.encoder[0].weight.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        save_image(weights, './filters/epoch_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')



# import torch
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torchvision import datasets, transforms
# from torch.autograd import Variable
 
# import numpy as np
# import math
# from utils import Dataset, masking_noise
# from ops import MSELoss, BCELoss
 
# def adjust_learning_rate(init_lr, optimizer, epoch):
#     lr = init_lr * (0.1 ** (epoch//100))
#     toprint = True
#     for param_group in optimizer.param_groups:
#         if param_group["lr"]!=lr:
#             param_group["lr"] = lr
#             if toprint:
#                 print("Switching to learning rate %f" % lr)
#                 toprint = False
 
# class DenoisingAutoencoder(nn.Module):
#     def __init__(self, in_features, out_features, activation="relu", 
#         dropout=0.2, tied=False):
#         super(self.__class__, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if tied:
#             self.deweight = self.weight.t()
#         else:
#             self.deweight = Parameter(torch.Tensor(in_features, out_features))
#         self.bias = Parameter(torch.Tensor(out_features))
#         self.vbias = Parameter(torch.Tensor(in_features))
        
#         if activation=="relu":
#             self.enc_act_func = nn.ReLU()
#         elif activation=="sigmoid":
#             self.enc_act_func = nn.Sigmoid()
#         elif activation=="none":
#             self.enc_act_func = None
#         self.dropout = nn.Dropout(p=dropout)
 
#         self.reset_parameters()
 
#     def reset_parameters(self):
#         stdv = 0.01
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)
#         stdv = 0.01
#         self.deweight.data.uniform_(-stdv, stdv)
#         self.vbias.data.uniform_(-stdv, stdv)
 
#     def forward(self, x):
#         if self.enc_act_func is not None:
#             return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
#         else:
#             return self.dropout(F.linear(x, self.weight, self.bias))
 
#     def encode(self, x, train=True):
#         if train:
#             self.dropout.train()
#         else:
#             self.dropout.eval()
#         if self.enc_act_func is not None:
#             return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
#         else:
#             return self.dropout(F.linear(x, self.weight, self.bias))
 
#     def encodeBatch(self, dataloader):
#         use_cuda = torch.cuda.is_available()
#         encoded = []
#         for batch_idx, (inputs, _) in enumerate(dataloader):
#             inputs = inputs.view(inputs.size(0), -1).float()
#             if use_cuda:
#                 inputs = inputs.cuda()
#             inputs = Variable(inputs)
#             hidden = self.encode(inputs, train=False)
#             encoded.append(hidden.data.cpu())
 
#         encoded = torch.cat(encoded, dim=0)
#         return encoded
 
#     def decode(self, x, binary=False):
#         if not binary:
#             return F.linear(x, self.deweight, self.vbias)
#         else:
#             return F.sigmoid(F.linear(x, self.deweight, self.vbias))
 
#     def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.3,
#         loss_type="mse"):
#         """
#         data_x: FloatTensor
#         valid_x: FloatTensor
#         """
#         use_cuda = torch.cuda.is_available()
#         if use_cuda:
#             self.cuda()
#         print("=====Denoising Autoencoding layer=======")
#         # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
#         optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
#         if loss_type=="mse":
#             criterion = MSELoss()
#         elif loss_type=="cross-entropy":
#             criterion = BCELoss()
 
#         # validate
#         total_loss = 0.0
#         total_num = 0
#         for batch_idx, (inputs, _) in enumerate(validloader):
#             # inputs = inputs.view(inputs.size(0), -1).float()
#             # if use_cuda:
#             #     inputs = inputs.cuda()
#             inputs = Variable(inputs)
#             hidden = self.encode(inputs)
#             if loss_type=="cross-entropy":
#                 outputs = self.decode(hidden, binary=True)
#             else:
#                 outputs = self.decode(hidden)
 
#             valid_recon_loss = criterion(outputs, inputs)
#             total_loss += valid_recon_loss.data * len(inputs)
#             total_num += inputs.size()[0]
 
#         valid_loss = total_loss / total_num
#         print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))
 
#         self.train()
#         for epoch in range(num_epochs):
#             # train 1 epoch
#             train_loss = 0.0
#             adjust_learning_rate(lr, optimizer, epoch)
#             for batch_idx, (inputs, _) in enumerate(trainloader):
#                 # inputs = inputs.view(inputs.size(0), -1).float()
#                 inputs_corr = masking_noise(inputs, corrupt)
#                 # if use_cuda:
#                 #     inputs = inputs.cuda()
#                 #     inputs_corr = inputs_corr.cuda()
#                 optimizer.zero_grad()
#                 inputs = Variable(inputs)
#                 inputs_corr = Variable(inputs_corr)
 
#                 hidden = self.encode(inputs_corr)
#                 if loss_type=="cross-entropy":
#                     outputs = self.decode(hidden, binary=True)
#                 else:
#                     outputs = self.decode(hidden)
#                 recon_loss = criterion(outputs, inputs)
#                 train_loss += recon_loss.data*len(inputs)
#                 recon_loss.backward()
#                 optimizer.step()
 
#             # validate
#             valid_loss = 0.0
#             for batch_idx, (inputs, _) in enumerate(validloader):
#                 # inputs = inputs.view(inputs.size(0), -1).float()
#                 # if use_cuda:
#                 #     inputs = inputs.cuda()
#                 inputs = Variable(inputs)
#                 hidden = self.encode(inputs, train=False)
#                 if loss_type=="cross-entropy":
#                     outputs = self.decode(hidden, binary=True)
#                 else:
#                     outputs = self.decode(hidden)
 
#                 valid_recon_loss = criterion(outputs, inputs)
#                 valid_loss += valid_recon_loss.data * len(inputs)
 
#             print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f" % (
#                 epoch+1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))
 
#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )