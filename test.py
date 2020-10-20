from models.Beta_VAE import BetaVAE
from models.Baseline_VAE import BaselineVAE

from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import csv
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
batch_size = 2

transform = transforms.Compose([
    transforms.ToTensor(),
])

# train and validation data
train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)

# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)


lr = 0.0001

model = BetaVAE().to(device)
#model = BaselineVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
#criterion = BetaVAE.loss_function()


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(batch_size, 1, 28, 28)
        optimizer.zero_grad()
        z, output, loss = model(data)
        # loss = BetaVAE.loss_function(x=data, posterior_x_z=x_dist, posterior_z_x=z_dist)
        running_loss += loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss



train_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    # val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    # val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    # print(f"Val Loss: {val_epoch_loss:.4f}")



"""
dataloader = train_loader
for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data

x = data[1].view(-1,1,28,28)
x = x.to(device)
model.generate(x)
"""

x = model.sample(2,current_device=device)

x = x.view(-1, 1, 28, 28)
result = torch.Tensor.cpu(x).detach()
result = result.numpy()
img = result[0]
img = img.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()