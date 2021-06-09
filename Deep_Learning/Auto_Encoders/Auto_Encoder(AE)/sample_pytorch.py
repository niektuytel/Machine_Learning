import os 
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# constants
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

def save_decoded_image(img, epoch):
    save_image(img.view(img.size(0), 1, 28, 28), f"./data/sample_pytorch_output/linear_ae_image{epoch}.png")

def make_dir():
    image_dir = "./data/sample_pytorch_output"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

# image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.FashionMNIST(
    root='./data',
    train=True, 
    download=True,
    transform=transform
)
testset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decoder
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))

        return x

def fit(net, trainloader, n_epochs, optimizer, loss_function=nn.MSELoss()):
    train_loss = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()

            outputs = net(img)
            loss = loss_function(outputs, img)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print(f"Epoch {epoch+1} of {n_epochs}, Train Loss: {loss}")

        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)
    return train_loss

def test_image_reconstruction(net, testloader):
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)

        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, "./data/sample_pytorch_output/fashionmnist_reconstruction.png")
        break


device = "cuda:0" if torch.cuda.is_available() else "cpu"
make_dir()

# define network
net = AutoEncoder()
net.to(device)
print(net)

# train network
train_loss = fit(
    net=net, 
    trainloader=trainloader, 
    n_epochs=NUM_EPOCHS, 
    optimizer=optim.Adam(net.parameters(), lr=LEARNING_RATE)
)

# network results
plt.figure()
plt.plot(train_loss)
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("./data/sample_pytorch_output/deep_ae_fashionmnist_loss.png")

# test the network
test_image_reconstruction(net, testloader)








